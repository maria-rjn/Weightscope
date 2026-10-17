'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { promisify } = require('node:util');
const execFile = promisify(require('node:child_process').execFile);
const backend = require('../src/backend');
const { saveCheckpoint } = require('../src/checkpoint');
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

test('upstream libraries can reopen metadata-edited checkpoints and validated PyTorch copies', { skip: !pythonPath }, async () => {
  for (const ext of ['safetensors', 'gguf', 'onnx']) {
    const model = await backend.inspect(path.join(dir, `official.${ext}`));
    await saveCheckpoint(model, path.join(dir, `edited.${ext}`), { metadata: [{ key: 'weightscope.test', value: 'round-trip' }] });
  }
  const torch = await backend.inspect(path.join(dir, 'official.pt'), options);
  await saveCheckpoint(torch, path.join(dir, 'copy.pt'), options);
  await execFile(pythonPath, ['-c', `
import os, sys, torch, onnx, gguf
from safetensors import safe_open
root = sys.argv[1]
with safe_open(os.path.join(root, 'edited.safetensors'), framework='pt') as f:
    assert f.metadata()['weightscope.test'] == 'round-trip'
model = onnx.load(os.path.join(root, 'edited.onnx'))
onnx.checker.check_model(model)
assert dict((p.key, p.value) for p in model.metadata_props)['weightscope.test'] == 'round-trip'
reader = gguf.GGUFReader(os.path.join(root, 'edited.gguf'))
assert 'weightscope.test' in reader.fields
assert len(reader.tensors) > 0
original = torch.load(os.path.join(root, 'official.pt'), weights_only=True)
copy = torch.load(os.path.join(root, 'copy.pt'), weights_only=True)
assert original['epoch'] == copy['epoch']
`, dir], { timeout: 120000 });
});

test('tensor edits agree with PyTorch rounding and official GGUF dequantization', { skip: !pythonPath }, async () => {
  await execFile(pythonPath, ['-c', `
import os, sys, torch
from safetensors.torch import save_file
root = sys.argv[1]
values = {}
for name, dtype in [('fp16', torch.float16), ('bf16', torch.bfloat16)]:
    tensor = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    tensor[~torch.isfinite(tensor)] = 0
    values[name] = tensor
    expected = (tensor.to(torch.float64) * .5).to(dtype).view(torch.int16).numpy().tobytes()
    open(os.path.join(root, name + '.bin'), 'wb').write(expected)
save_file(values, os.path.join(root, 'rounding.safetensors'))
`, dir], { timeout: 120000 });
  const rounding = await backend.inspect(path.join(dir, 'rounding.safetensors'));
  const roundedPath = path.join(dir, 'rounded.safetensors');
  await saveCheckpoint(rounding, roundedPath, { tensors: rounding.tensors.map(t => ({ name: t.name, operation: 'scale', factor: .5 })) });
  const rounded = await backend.inspect(roundedPath), bytes = await fs.readFile(roundedPath);
  for (const tensor of rounded.tensors) assert.deepEqual(bytes.subarray(tensor.storage.offset, tensor.storage.offset + tensor.bytes), await fs.readFile(path.join(dir, `${tensor.name}.bin`)));
  const { tensorCapabilities } = require('../src/checkpoint');
  const gguf = await backend.inspect(path.join(dir, 'official.gguf'));
  await saveCheckpoint(gguf, path.join(dir, 'scaled.gguf'), { tensors: gguf.tensors.filter(t => tensorCapabilities(gguf, t).operations.includes('scale')).map(t => ({ name: t.name, operation: 'scale', factor: .5 })) });
  await execFile(pythonPath, ['-c', `
import os, sys, gguf, numpy as np
root = sys.argv[1]
original = gguf.GGUFReader(os.path.join(root, 'official.gguf'))
changed = gguf.GGUFReader(os.path.join(root, 'scaled.gguf'))
for a, b in zip(original.tensors, changed.tensors):
    assert a.name == b.name
    expected = gguf.quants.dequantize(a.data, a.tensor_type) * .5
    actual = gguf.quants.dequantize(b.data, b.tensor_type)
    np.testing.assert_array_equal(actual, expected)
`, dir], { timeout: 120000 });
});

test('PyTorch Save As edits preserve wrappers, unselected aliases, and original files', { skip: !pythonPath }, async () => {
  await execFile(pythonPath, ['-c', `
import os, sys, torch
root = sys.argv[1]
t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
torch.save({'state_dict': {'a': t, 'alias': t, 'transpose': t.T}, 'epoch': 7, 'optimizer': {'step': 3}}, os.path.join(root, 'aliases.pt'))
torch.save({'state_dict': {'replacement': torch.full((2, 3), 9.)}}, os.path.join(root, 'donor.pt'))
`, dir], { timeout: 120000 });
  const model = await backend.inspect(path.join(dir, 'aliases.pt'), options), donor = await backend.inspect(path.join(dir, 'donor.pt'), options);
  const original = await fs.readFile(model.path), output = path.join(dir, 'aliases-edited.pt');
  await saveCheckpoint(model, output, { ...options, tensors: [{ name: 'a', operation: 'replace', source: { model: donor, name: 'replacement' } }, { name: 'transpose', operation: 'zero' }] });
  const edited = await backend.inspect(output, options);
  assert.deepEqual(await backend.sample(edited, 'a', options), [9, 9, 9, 9, 9, 9]);
  assert.deepEqual(await backend.sample(edited, 'alias', options), [0, 1, 2, 3, 4, 5]);
  assert.deepEqual(await backend.sample(edited, 'transpose', options), [0, 0, 0, 0, 0, 0]);
  assert.equal(edited.metadata.epoch, 7); assert.deepEqual(await fs.readFile(model.path), original);
  await execFile(pythonPath, ['-c', `import torch,sys; assert torch.load(sys.argv[1],weights_only=True)['optimizer']['step'] == 3`, output]);
  const scaled = path.join(dir, 'aliases-scaled.pt');
  await saveCheckpoint(model, scaled, { ...options, tensors: [{ name: 'a', operation: 'scale', factor: 2 }] });
  assert.deepEqual(await backend.sample(await backend.inspect(scaled, options), 'a', options), [0, 2, 4, 6, 8, 10]);
});

test('upstream safetensors and ONNX serialization agrees with native readers', { skip: !pythonPath }, async () => {
  const safe = await backend.inspect(path.join(dir, 'official.safetensors'));
  const onnx = await backend.inspect(path.join(dir, 'official.onnx'));
  assert.equal(safe.summary.parameters, 27); assert.equal(onnx.summary.parameters, 15);
  assert.equal((await backend.tensorStatistics(onnx, 'packed')).mean, 2);
  assert.equal((await backend.compareTensor(safe, onnx, 'layer.weight')).status, 'equal');
  const external = await backend.inspect(path.join(dir, 'external.onnx'));
  assert.equal((await backend.tensorStatistics(external, 'layer.weight')).mean, 5.5);
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
