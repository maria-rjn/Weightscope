'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const backend = require('../src/backend');
const { saveCheckpoint, tensorCapabilities } = require('../src/checkpoint');
const { writeScalar } = require('../src/tensor-edits');
const { TYPES } = require('../src/parsers/gguf');
const { scalar } = require('../src/binary');
const f = require('./fixtures');
async function directory(t) { const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-tensor-edit-')); t.after(() => fs.rm(dir, { recursive: true, force: true })); return dir; }
test('native edits combine metadata, scaling, zeroing and compatible replacement without changing original tensors', async t => {
  const dir = await directory(t);
  for (const ext of ['safetensors', 'gguf', 'onnx']) {
    const base = path.join(dir, `base.${ext}`), donor = path.join(dir, `donor.${ext}`), output = path.join(dir, `edited.${ext}`);
    await f[ext](base, ['scale', 'zero', 'replace', 'keep'].map(name => ({ name, shape: [2], values: [2, -4] })));
    await f[ext](donor, [{ name: 'replacement', shape: [2], values: [7, 8] }]);
    const model = await backend.inspect(base), source = await backend.inspect(donor), original = await fs.readFile(base);
    await saveCheckpoint(model, output, { metadata: [{ key: 'description', value: 'Changed header length and values' }], tensors: [
      { name: 'scale', operation: 'scale', factor: -.5 }, { name: 'zero', operation: 'zero' }, { name: 'replace', operation: 'replace', source: { model: source, name: 'replacement' } }
    ] });
    const edited = await backend.inspect(output);
    assert.deepEqual(await backend.sample(edited, 'scale'), [-1, 2]); assert.deepEqual(await backend.sample(edited, 'zero'), [0, 0]);
    assert.deepEqual(await backend.sample(edited, 'replace'), [7, 8]); assert.deepEqual(await backend.sample(edited, 'keep'), [2, -4]);
    assert.deepEqual(await fs.readFile(base), original);
  }
});
test('F16 and BF16 encoding rounds ties to even, preserves signed zero, and handles subnormals', () => {
  for (const [dtype, mantissa, minimum] of [['F16', 10, -24], ['BF16', 7, -133]]) {
    const buffer = Buffer.alloc(2);
    for (const [value, expected] of [[1 + 2 ** (-mantissa - 1), 1], [1 + 3 * 2 ** (-mantissa - 1), 1 + 2 ** (1 - mantissa)], [2 ** minimum, 2 ** minimum], [2 ** (minimum - 1), 0], [-0, -0]]) {
      writeScalar(buffer, 0, dtype, value); assert.ok(Object.is(scalar(buffer, 0, dtype), expected), `${dtype}: ${value}`);
    }
  }
  assert.throws(() => writeScalar(Buffer.alloc(2), 0, 'F16', 65520), /overflows/);
});
test('F16 and BF16 tensor scaling writes their stored precision and zeroing clears non-finite payloads', async t => {
  const dir = await directory(t), file = path.join(dir, 'base.safetensors');
  const fp16 = Buffer.alloc(6); [1, -2, 2 ** -24].forEach((v, i) => writeScalar(fp16, i * 2, 'F16', v));
  const bf16 = Buffer.alloc(4); [1.5, -3].forEach((v, i) => writeScalar(bf16, i * 2, 'BF16', v));
  await f.safetensors(file, [{ name: 'fp16', shape: [3], dtype: 'F16', data: fp16 }, { name: 'bf16', shape: [2], dtype: 'BF16', data: bf16 }, { name: 'nonfinite', values: [NaN, Infinity] }]);
  const model = await backend.inspect(file), output = path.join(dir, 'edited.safetensors');
  await saveCheckpoint(model, output, { tensors: [{ name: 'fp16', operation: 'scale', factor: 2 }, { name: 'bf16', operation: 'scale', factor: -2 }, { name: 'nonfinite', operation: 'zero' }] });
  const edited = await backend.inspect(output);
  assert.deepEqual(await backend.sample(edited, 'fp16'), [2, -4, 2 ** -23]); assert.deepEqual(await backend.sample(edited, 'bf16'), [-3, 6]); assert.deepEqual(await backend.sample(edited, 'nonfinite'), [0, 0]);
});
test('supported GGUF quantization blocks scale and zero via their scales across streaming boundaries', async t => {
  const dir = await directory(t), file = path.join(dir, 'base.gguf');
  const scales = { Q4_0: [0], Q4_1: [0, 2], Q5_0: [0], Q5_1: [0, 2], Q8_0: [0], Q2_K: [80, 82], Q3_K: [108], Q4_K: [0, 2], Q5_K: [0, 2], Q6_K: [208], Q8_K: [0] };
  const tensors = Object.entries(TYPES).filter(([, [name]]) => Object.hasOwn(scales, name)).map(([type, [name, block, bytes]]) => {
    const blocks = name === 'Q4_0' ? 60000 : 2, data = Buffer.alloc(bytes * blocks, 0x21);
    for (let start = 0; start < data.length; start += bytes) for (const offset of scales[name]) writeScalar(data, start + offset, name === 'Q8_K' ? 'F32' : 'F16', 1);
    return { name, type: Number(type), shape: [blocks * block], data };
  });
  await f.gguf(file, tensors); const model = await backend.inspect(file);
  for (const [operation, factor] of [['scale', -2], ['zero', undefined]]) {
    const output = path.join(dir, `${operation}.gguf`);
    await saveCheckpoint(model, output, { tensors: tensors.map(t => ({ name: t.name, operation, factor })) }); const edited = await backend.inspect(output);
    for (const tensor of tensors) {
      const original = await backend.sample(model, tensor.name), values = await backend.sample(edited, tensor.name);
      values.forEach((v, i) => assert.ok(v === (operation === 'zero' ? 0 : original[i] * factor), `${tensor.name}[${i}]`));
    }
  }
});
test('ONNX repeated typed floating-point payloads can be edited and replaced from raw storage', async t => {
  const dir = await directory(t), file = path.join(dir, 'typed.onnx'), donor = path.join(dir, 'donor.safetensors');
  const tensor = Buffer.concat([f.field(1, f.vint(3)), f.vfield(2, 1), f.field(8, 'w'), f.field(4, f.f32([1])), f.field(4, f.f32([2, 3]))]);
  await fs.writeFile(file, f.field(7, f.field(5, tensor))); await f.safetensors(donor, [{ name: 'v', values: [4, 5, 6] }]);
  const model = await backend.inspect(file);
  await saveCheckpoint(model, path.join(dir, 'scale.onnx'), { tensors: [{ name: 'w', operation: 'scale', factor: 2 }] });
  assert.deepEqual(await backend.sample(await backend.inspect(path.join(dir, 'scale.onnx')), 'w'), [2, 4, 6]);
  await saveCheckpoint(model, path.join(dir, 'replace.onnx'), { tensors: [{ name: 'w', operation: 'replace', source: { model: await backend.inspect(donor), name: 'v' } }] });
  assert.deepEqual(await backend.sample(await backend.inspect(path.join(dir, 'replace.onnx')), 'w'), [4, 5, 6]);
});
test('unsafe tensor plans, stale donors, overflow and non-finite scaling leave originals and destinations intact', async t => {
  const dir = await directory(t), file = path.join(dir, 'base.safetensors'), donor = path.join(dir, 'donor.safetensors');
  await f.safetensors(file, [{ name: 'w', values: [3e38] }, { name: 'nan', values: [NaN] }]); await f.safetensors(donor, [{ name: 'w', values: [2] }]);
  const model = await backend.inspect(file), source = await backend.inspect(donor), original = await fs.readFile(file), output = path.join(dir, 'failed.safetensors');
  for (const tensors of [[{ name: 'w', operation: 'scale', factor: 2 }], [{ name: 'nan', operation: 'scale', factor: 1 }], [{ name: 'w', operation: 'scale', factor: '2' }], [{ name: 'w', operation: 'unknown' }], [{ name: 'missing', operation: 'zero' }]]) await assert.rejects(saveCheckpoint(model, output, { tensors }));
  await fs.appendFile(donor, ' ');
  await assert.rejects(saveCheckpoint(model, output, { tensors: [{ name: 'w', operation: 'replace', source: { model: source, name: 'w' } }] }), /changed on disk/);
  await assert.rejects(fs.stat(output), { code: 'ENOENT' }); assert.deepEqual(await fs.readFile(file), original);
  assert.equal((await fs.readdir(dir)).some(n => n.startsWith('.weightscope-')), false);
});
test('external ONNX and unknown GGUF block layouts expose editing limitations', async t => {
  const dir = await directory(t), file = path.join(dir, 'external.onnx');
  await f.onnx(file, [{ name: 'w', shape: [1], external: { location: 'data.bin' } }]);
  const model = await backend.inspect(file); assert.deepEqual(tensorCapabilities(model, model.tensors[0]).operations, []);
});
