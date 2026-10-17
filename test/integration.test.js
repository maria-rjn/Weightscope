'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { promisify } = require('node:util');
const execFile = promisify(require('node:child_process').execFile);
const backend = require('../src/backend');
const { sampleWindows } = require('../src/binary');
const { translate, ARCHITECTURES } = require('../src/tensor-names');
const pythonPath = process.env.WEIGHTSCOPE_TEST_PYTHON;
let dir;
before(async () => {
  if (!pythonPath) return;
  dir = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-framework-'));
  await execFile(pythonPath, ['scripts/framework-fixtures.py', dir], { timeout: 120000 });
});
after(async () => { if (dir) await fs.rm(dir, { recursive: true, force: true }); });
const options = { pythonPath, trusted: true, sampleSize: 32768 };
// Writes real torch.save shards from { 'relative/file.bin': { name: [values] } } in one Python process.
async function torchShards(root, files) {
  const script = 'import json, sys, torch\nfrom pathlib import Path\nfor name, tensors in json.loads(sys.argv[2]).items():\n    file = Path(sys.argv[1], name)\n    file.parent.mkdir(parents=True, exist_ok=True)\n    torch.save({k: torch.tensor(v) for k, v in tensors.items()}, file)\n';
  await execFile(pythonPath, ['-I', '-c', script, root, JSON.stringify(files)], { timeout: 120000 });
}

test('upstream safetensors and ONNX serialization agrees with native readers', { skip: !pythonPath }, async () => {
  const safe = await backend.inspect(path.join(dir, 'official.safetensors'));
  const onnx = await backend.inspect(path.join(dir, 'official.onnx'));
  assert.equal(safe.summary.parameters, 27); assert.equal(onnx.summary.parameters, 15);
  assert.equal((await backend.tensorStatistics(onnx, 'packed')).mean, 2);
  assert.equal((await backend.compareTensor(safe, onnx, 'layer.weight')).status, 'equal');
  const external = await backend.inspect(path.join(dir, 'external.onnx'));
  assert.equal((await backend.tensorStatistics(external, 'layer.weight')).mean, 5.5);
});
test('upstream safetensors shards with a transformers-style index match the single-file checkpoint', { skip: !pythonPath }, async () => {
  const model = await backend.inspect(path.join(dir, 'sharded', 'model-00002-of-00002.safetensors'));
  const single = await backend.inspect(path.join(dir, 'official.safetensors'));
  assert.equal(model.source.shards.length, 2); assert.equal(model.summary.parameters, single.summary.parameters);
  assert.equal(model.summary.tensorBytes, model.metadata.index.total_size); assert.equal(model.byName.get('layer.bias').dtype, 'BF16');
  for (const t of single.tensors) assert.equal(backend.structureDiff(model, single).find(r => r.name === t.name).status, 'unverified', t.name);
  for (const name of ['layer.weight', 'layer.bias', 'noncontiguous']) assert.equal((await backend.compareTensor(model, single, name)).status, 'equal', name);
});
test('upstream PyTorch .bin shards with a transformers-style index match the safetensors shards beside them', { skip: !pythonPath }, async () => {
  const index = path.join(dir, 'sharded', 'pytorch_model.bin.index.json');
  const model = await backend.inspect(path.join(dir, 'sharded', 'pytorch_model-00002-of-00002.bin'), options);
  const safe = await backend.inspect(path.join(dir, 'sharded', 'model-00001-of-00002.safetensors'));
  assert.equal(model.format, 'pytorch'); assert.equal(model.source.indexPath, index); assert.equal(model.source.name, 'sharded/pytorch_model.bin');
  assert.deepEqual(model.source.shards.map(s => s.tensors), [2, 2]); assert.equal(model.warnings.length, 1);
  assert.equal(model.summary.parameters, safe.summary.parameters); assert.equal(model.summary.tensorBytes, model.metadata.index.total_size);
  assert.equal(model.byName.get('layer.bias').dtype, 'BF16'); assert.equal(model.byName.get('noncontiguous').shard, 'pytorch_model-00002-of-00002.bin');
  assert.deepEqual(await backend.sample(model, 'layer.bias', options), [-1, 0, 1]);
  assert.deepEqual(await backend.sample(model, 'noncontiguous', options), [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]);
  assert.ok(backend.structureDiff(model, safe).every(r => r.status === 'unverified'));
  for (const name of ['layer.weight', 'layer.bias', 'noncontiguous']) assert.equal((await backend.compareTensor(model, safe, name, options)).status, 'equal', name);
  assert.equal((await backend.inspect(index, options)).tensors.length, 4);
  await assert.rejects(backend.inspect(index, { ...options, trusted: false }), /trusted VS Code workspace/);
});
test('real PyTorch shard sets are validated, refuse executable pickle, detect changes and fall back to one shard', { skip: !pythonPath }, async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-pt-shards-')); t.after(() => fs.rm(root, { recursive: true, force: true }));
  const shard = i => `pytorch_model-0000${i}-of-00002.bin`, file = (set, name) => path.join(root, set, name);
  await torchShards(root, { [`set/${shard(1)}`]: { 'a.weight': [1, 2] }, [`set/${shard(2)}`]: { 'b.weight': [3, 4] }, [`duplicate/${shard(1)}`]: { 'a.weight': [1, 2] }, [`duplicate/${shard(2)}`]: { 'a.weight': [5, 6], 'b.weight': [3, 4] } });
  const index = async (set, map) => { await fs.writeFile(file(set, 'pytorch_model.bin.index.json'), JSON.stringify({ weight_map: map })); return file(set, 'pytorch_model.bin.index.json'); };
  const fallsBack = async (set, expected) => {
    const alone = await backend.inspect(file(set, shard(1)), options);
    assert.equal(alone.source, undefined); assert.match(alone.warnings.join(' '), expected);
    assert.deepEqual(await backend.sample(alone, 'a.weight', options), [1, 2]);
  };
  const complete = await index('set', { 'a.weight': shard(1), 'b.weight': shard(2) });
  const model = await backend.inspect(complete, options);
  assert.deepEqual(await backend.sample(model, 'b.weight', options), [3, 4]);
  await index('set', { 'a.weight': shard(2), 'b.weight': shard(1) });
  await assert.rejects(backend.sample(model, 'a.weight', options), /The checkpoint changed on disk/);
  await assert.rejects(backend.inspect(complete, options), /maps a\.weight to pytorch_model-00002-of-00002\.bin, but it is stored in pytorch_model-00001-of-00002\.bin/);
  await fallsBack('set', /Showing this shard only: The index maps a\.weight/);
  await index('set', { 'a.weight': shard(1), 'b.weight': shard(2), 'c.weight': shard(2) });
  await assert.rejects(backend.inspect(complete, options), /lists c\.weight in pytorch_model-00002-of-00002\.bin, but that shard does not contain it/);
  await assert.rejects(backend.inspect(await index('duplicate', { 'a.weight': shard(1), 'b.weight': shard(2) }), options), /a\.weight is stored in both/);
  await index('set', { 'a.weight': shard(1), 'b.weight': shard(2) });
  const current = await backend.inspect(file('set', shard(1)), options);
  await fs.appendFile(file('set', shard(2)), Buffer.from([0]));
  await assert.rejects(backend.sample(current, 'b.weight', options), /Shard pytorch_model-00002-of-00002\.bin changed on disk/);
  // A shard carrying an executable pickle is rejected by restricted loading; the rest of its set still opens alone.
  await fs.copyFile(path.join(dir, 'rejected.pt'), file('set', shard(2)));
  await assert.rejects(backend.inspect(complete, options), /Cannot read shard pytorch_model-00002-of-00002\.bin: Restricted weights-only loading rejected/);
  await fallsBack('set', /Cannot read shard pytorch_model-00002-of-00002\.bin/);
  await assert.rejects(fs.stat(path.join(dir, 'should-not-exist')), { code: 'ENOENT' });
  await fs.rm(file('set', shard(2)));
  await assert.rejects(backend.inspect(complete, options), /1 of 2 shards listed in pytorch_model\.bin\.index\.json are missing/);
  await fallsBack('set', /are missing, including pytorch_model-00002-of-00002\.bin/);
});
test('upstream GGUF writer and official dequantizers agree with scalar, primary and K decoders', { skip: !pythonPath }, async () => {
  const m = await backend.inspect(path.join(dir, 'official.gguf'));
  const expected = JSON.parse(await fs.readFile(path.join(dir, 'expected.json'), 'utf8'));
  assert.deepEqual(m.byName.get('float.weight').shape, [3, 4]);
  for (const [name, values] of Object.entries(expected)) {
    const actual = await backend.sample(m, name);
    assert.equal(actual.length, values.length);
    actual.forEach((v, i) => assert.ok(Math.abs(v - values[i]) < 1e-5, `${name} at ${i}: ${v} vs ${values[i]}`));
    const bounded = await backend.sample(m, name, { sampleSize: 256 });
    const reference = sampleWindows(values.length, 256).flatMap(w => values.slice(w.start, w.start + w.count));
    bounded.forEach((v, i) => assert.ok(Math.abs(v - reference[i]) < 1e-5));
  }
});
test('a llama.cpp-style Q8_0 conversion pairs with its Hugging Face source tensor by tensor', { skip: !pythonPath }, async () => {
  const hf = await backend.inspect(path.join(dir, 'hf-llama.safetensors')), gguf = await backend.inspect(path.join(dir, 'llama-q8_0.gguf'));
  const expected = JSON.parse(await fs.readFile(path.join(dir, 'cross-format.json'), 'utf8')), rows = backend.structureDiff(hf, gguf);
  assert.equal(backend.tensorMatching(hf, gguf).matched, Object.keys(expected).length); assert.equal(rows.length, Object.keys(expected).length);
  for (const row of rows) {
    const e = expected[row.name]; assert.equal(row.targetName, e.target); assert.equal(row.status, 'unverified');
    const d = await backend.compareTensor(hf, gguf, row.name, options);
    assert.equal(d.exact, true); assert.equal(d.status, e.maxAbs ? 'changed' : 'equal', row.name);
    assert.ok(Math.abs(d.meanAbs - e.meanAbs) < 1e-9, `${row.name} mean |Δ| ${d.meanAbs} vs ${e.meanAbs}`);
    assert.ok(Math.abs(d.maxAbs - e.maxAbs) < 1e-9, `${row.name} max |Δ| ${d.maxAbs} vs ${e.maxAbs}`);
    if (e.maxAbs) assert.ok(d.cosine > .999, `${row.name} cosine ${d.cosine}`);
  }
});
test('name translation agrees with gguf-py’s tensor name map for every supported architecture', { skip: !pythonPath }, async () => {
  const expected = JSON.parse(await fs.readFile(path.join(dir, 'gguf-names.json'), 'utf8'));
  assert.deepEqual(Object.keys(expected).sort(), Object.keys(ARCHITECTURES).sort());
  // gguf-py omits tensors an architecture never has; every name it does define must agree.
  for (const [arch, names] of Object.entries(expected)) for (const [name, target] of Object.entries(names)) if (target) assert.equal(translate(name, ARCHITECTURES[arch])?.name, target, `${arch}: ${name}`);
});
test('MXFP8 agrees with PyTorch for every FP8 code and E8M0 scale, in full and bounded samples', { skip: !pythonPath }, async () => {
  const model = await backend.inspect(path.join(dir, 'ollama', 'manifests', 'registry.ollama.ai', 'library', 'mxfp8', 'latest'));
  const expected = JSON.parse(await fs.readFile(path.join(dir, 'mxfp8-expected.json'), 'utf8')).map(v => v === 'NaN' ? NaN : v === 'Infinity' ? Infinity : v === '-Infinity' ? -Infinity : v);
  assert.equal(expected.length, 65536);
  assert.deepEqual(await backend.sample(model, 'packed.weight', { sampleSize: 65536 }), expected);
  const bounded = await backend.sample(model, 'packed.weight', { sampleSize: 32768 });
  assert.deepEqual(bounded, sampleWindows(expected.length, 32768).flatMap(w => expected.slice(w.start, w.start + w.count)));
});
test('real PyTorch metadata excludes optimizer state, samples BF16 and noncontiguous tensors', { skip: !pythonPath }, async () => {
  const m = await backend.inspect(path.join(dir, 'official.pt'), options);
  assert.equal(m.summary.parameters, 27); assert.equal(m.metadata.epoch, 7); assert.equal(m.byName.get('layer.bias').dtype, 'BF16');
  assert.deepEqual(await backend.sample(m, 'layer.bias', options), [-1, 0, 1]);
  assert.deepEqual(await backend.sample(m, 'noncontiguous', options), [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]);
  const legacy = await backend.inspect(path.join(dir, 'legacy.pth'), options);
  assert.equal((await backend.tensorStatistics(legacy, 'layer.weight', options)).mean, 5.5);
  const safe = await backend.inspect(path.join(dir, 'official.safetensors'));
  assert.equal((await backend.compareTensor(m, safe, 'noncontiguous', options)).status, 'equal');
});
test('PyTorch rejects executable pickle and TorchScript, preserves non-finite diagnostics', { skip: !pythonPath }, async () => {
  await assert.rejects(backend.inspect(path.join(dir, 'rejected.pt'), options), /Restricted weights-only/);
  await assert.rejects(backend.inspect(path.join(dir, 'scripted.pt'), options), /TorchScript/);
  await assert.rejects(fs.stat(path.join(dir, 'should-not-exist')), { code: 'ENOENT' });
  const m = await backend.inspect(path.join(dir, 'nonfinite.pt'), options);
  const s = await backend.tensorStatistics(m, 'tensor', options);
  assert.equal(s.nan, 1); assert.equal(s.inf, 1); assert.equal(s.mean, 1);
});
