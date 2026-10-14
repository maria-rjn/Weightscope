'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const ollama = require('../src/ollama');
const backend = require('../src/backend');
const F = require('./fixtures');
const { store } = require('./ollama-fixtures');
async function fixture(t) { const root = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-ollama-')); t.after(() => fs.rm(root, { recursive: true, force: true })); return store(root); }

test('Ollama lookup respects custom paths, environment, home and Linux service stores', async t => {
  assert.deepEqual(ollama.candidateRoots('~/custom', { OLLAMA_MODELS: '/env' }, 'linux', '/home/test'), ['/home/test/custom']);
  assert.deepEqual(ollama.candidateRoots('', { OLLAMA_MODELS: '/env' }, 'darwin', '/home/test'), ['/env']);
  assert.deepEqual(ollama.candidateRoots('', {}, 'win32', '/home/test'), ['/home/test/.ollama/models']);
  assert.ok(ollama.candidateRoots('', {}, 'linux', '/home/test').includes('/usr/share/ollama/.ollama/models'));
  const s = await fixture(t); assert.equal((await ollama.findRoot(s.root, {})).root, s.root);
  assert.equal((await ollama.findRoot(path.join(s.root, 'missing'), { OLLAMA_MODELS: s.root })).root, null);
});

test('Ollama GGUF manifests open original hash blobs, retain tags and support numerical diffs', async t => {
  const s = await fixture(t), a = await s.blob('model', [{ name: 'blk.0.weight', shape: [2], values: [1, 2] }]);
  const template = await s.blob('template', '{{ .Prompt }}'), params = await s.blob('params', { temperature: 0.4 });
  const file = await s.manifest('tiny', 'q4', [a, template, params]); await s.manifest('tiny', 'latest', [a], {}, 'team');
  const entries = await ollama.listModels(s.root); assert.deepEqual(entries.map(e => e.name), ['team/tiny:latest', 'tiny:q4']);
  const model = await backend.inspect(file), original = ollama.blobPath(s.root, a.digest);
  assert.equal(model.path, original); assert.equal(backend.publicModel(model).name, 'tiny:q4');
  assert.equal(model.source.template.text, '{{ .Prompt }}'); assert.equal(model.source.params.temperature, 0.4);
  assert.equal((await backend.tensorStatistics(model, 'blk.0.weight', { sampleSize: 256 })).mean, 1.5);
  assert.equal((await backend.inspect(original)).format, 'gguf');
  const other = path.join(s.root, 'target.safetensors'); await F.safetensors(other, [{ name: 'blk.0.weight', values: [1, 3] }]);
  assert.equal((await backend.compareTensor(model, await backend.inspect(other), 'blk.0.weight')).maxAbs, 1);
  assert.ok((await fs.readdir(path.join(s.root, 'blobs'))).every(name => !name.includes('.gguf')));
  assert.equal(ollama.manifestLocation(original), null);
});

test('Ollama safetensors shards form one model and sample the right underlying files', async t => {
  const s = await fixture(t), a = await s.blob('tensor', [{ name: 'layer.0.weight', values: [1, 2] }]), b = await s.blob('tensor', [{ name: 'layer.1.weight', values: [4, 8] }]);
  const file = await s.manifest('split', 'latest', [a, b, a]); const model = await backend.inspect(file);
  assert.equal(model.format, 'safetensors'); assert.equal(model.summary.parameters, 4); assert.equal(model.tensors.length, 2);
  assert.equal(model.fileBytes, a.size + b.size); assert.equal(model.source.totalSize, a.size + b.size);
  assert.equal((await backend.tensorStatistics(model, 'layer.1.weight', { sampleSize: 256 })).mean, 6);
  await fs.appendFile(ollama.blobPath(s.root, b.digest), 'changed');
  await assert.rejects(backend.tensorStatistics(model, 'layer.1.weight'), /blob changed/);
});

test('packed Ollama quantization exposes its scheme and refuses misleading numerical analysis', async t => {
  const s = await fixture(t), raw = Buffer.alloc(8); raw.writeUInt32LE(0x12345678);
  const a = await s.blob('tensor', [{ name: 'packed.weight', shape: [2], dtype: 'U32', data: raw }], { metadata: { quant_type: 'mxfp8' } });
  const file = await s.manifest('packed', 'latest', [a]); const model = await backend.inspect(file);
  assert.equal(model.tensors[0].quantization.scheme, 'mxfp8'); assert.equal(model.summary.quantizedTensors, 1);
  assert.match(model.warnings[0], /stored codes/);
  await assert.rejects(backend.tensorStatistics(model, 'packed.weight'), /Numerical decoding.*unavailable/);
});

test('Ollama picker entries explain missing, corrupt, non-GGUF, cloud and invalid-digest models', async t => {
  const s = await fixture(t), a = await s.blob('model', [{ name: 'w', shape: [1], values: [1] }]);
  await s.manifest('missing', 'latest', [{ ...a, digest: `sha256:${'0'.repeat(64)}` }]);
  await s.manifest('invalid', 'latest', [{ ...a, digest: '../../outside' }]);
  await s.manifest('cloud', 'latest', []);
  const bad = await s.manifest('broken', 'latest', []); await fs.writeFile(bad, '{broken');
  const text = await s.blob('model', [{ name: 'w', shape: [1], values: [1] }]); await fs.writeFile(ollama.blobPath(s.root, text.digest), 'not GGUF');
  await s.manifest('wrong-format', 'latest', [text]);
  const entries = await ollama.listModels(s.root), problems = Object.fromEntries(entries.map(e => [e.model, e.problem]));
  assert.match(problems.missing, /missing/); assert.match(problems.invalid, /invalid digest/); assert.match(problems.cloud, /cloud-only/);
  assert.ok(problems.broken); assert.match(problems['wrong-format'], /not a supported/);
  for (const e of entries) await assert.rejects(backend.inspect(e.manifestPath));
  assert.throws(() => ollama.blobPath(s.root, 'sha256:../../outside'), /Invalid/);
});

test('Ollama text previews are bounded, optional failures stay visible and config blobs are protected', async t => {
  const s = await fixture(t), a = await s.blob('model', [{ name: 'w', shape: [1], values: [1] }]);
  const template = await s.blob('template', 'x'.repeat(70000)), params = await s.blob('params', '{broken'), license = await s.blob('license', 'test license'), config = await s.blob('config', { model_format: 'gguf' });
  const file = await s.manifest('text', 'latest', [a, template, params, license], { config });
  const model = await backend.inspect(file); assert.equal(model.source.template.text.length, 65536); assert.equal(model.source.template.truncated, true);
  assert.equal(model.source.licenses[0].text, 'test license'); assert.match(model.source.warnings[0], /params/);
  assert.ok(model.protectedPaths.includes(ollama.blobPath(s.root, config.digest)));
  await fs.appendFile(file, ' '); await assert.rejects(backend.tensorStatistics(model, 'w'), /manifest changed/);
});

test('Ollama manifest lists resolve local digest references and reject cycles', async t => {
  const s = await fixture(t), a = await s.blob('model', [{ name: 'w', shape: [1], values: [7] }]);
  const child = await s.blob('manifest', { layers: [a], runner: 'ggml', format: 'gguf' });
  const file = await s.manifest('variant', 'latest', undefined, { manifests: [{ digest: `sha256:${'0'.repeat(64)}`, format: 'gguf' }, { format: 'gguf', layers: [{ ...a, digest: `sha256:${'0'.repeat(64)}` }] }, { digest: child.digest, format: 'gguf' }] });
  const model = await backend.inspect(file); assert.equal(model.source.runner, 'ggml'); assert.equal((await backend.tensorStatistics(model, 'w')).mean, 7);
  const cycleDigest = `sha256:${'a'.repeat(64)}`, cyclePath = ollama.blobPath(s.root, cycleDigest);
  await fs.writeFile(cyclePath, JSON.stringify({ digest: cycleDigest }));
  const loop = await s.manifest('cycle', 'latest', undefined, { digest: cycleDigest });
  await assert.rejects(backend.inspect(loop), /nesting limit/);
});

test('Ollama duplicate tensor names and cancelled inspection fail explicitly', async t => {
  const s = await fixture(t), a = await s.blob('tensor', [{ name: 'w', values: [1] }]), b = await s.blob('tensor', [{ name: 'w', values: [2] }]);
  const file = await s.manifest('duplicate', 'latest', [a, b]); await assert.rejects(backend.inspect(file), /Duplicate Ollama tensor/);
  const controller = new AbortController(); controller.abort(); await assert.rejects(backend.inspect(file, { signal: controller.signal }), /cancelled/);
});

test('Ollama manifest names identify generic data tensors, shared blobs retain aliases and packed image weights stay explicit', async t => {
  const s = await fixture(t), a = await s.blob('tensor', [{ name: 'data', values: [1, 2] }]);
  const raw = Buffer.alloc(4), packed = await s.blob('tensor', [{ name: 'data', dtype: 'U32', shape: [1], data: raw }]);
  const index = await s.blob('json', { quantization: 'FP4' }, { name: 'model_index.json' });
  const file = await s.manifest('image', 'latest', [{ ...a, name: 'encoder.a.weight' }, { ...a, name: 'encoder.b.weight' }, { ...packed, name: 'encoder.c.weight' }, { ...a, name: 'encoder.c.weight_scale' }, index]);
  const model = await backend.inspect(file);
  assert.deepEqual(model.tensors.map(t => t.name), ['encoder.a.weight', 'encoder.b.weight', 'encoder.c.weight', 'encoder.c.weight_scale']);
  assert.equal(model.summary.parameters, 7); assert.equal(model.fileBytes, a.size + packed.size);
  assert.equal((await backend.tensorStatistics(model, 'encoder.b.weight')).mean, 1.5);
  assert.equal(model.byName.get('encoder.c.weight').quantization.scheme, 'FP4');
  await assert.rejects(backend.tensorStatistics(model, 'encoder.c.weight'), /Numerical decoding.*unavailable/);
  assert.equal(model.source.configs['model_index.json'].quantization, 'FP4');
});
