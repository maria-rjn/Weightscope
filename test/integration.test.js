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
const pythonPath = process.env.WEIGHTSCOPE_TEST_PYTHON;
let dir;
before(async () => {
  if (!pythonPath) return;
  dir = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-framework-'));
  await execFile(pythonPath, ['scripts/framework-fixtures.py', dir], { timeout: 120000 });
});
after(async () => { if (dir) await fs.rm(dir, { recursive: true, force: true }); });
const options = { pythonPath, trusted: true, sampleSize: 32768 };

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
