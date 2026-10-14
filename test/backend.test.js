'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const backend = require('../src/backend');
const { statistics, difference, structureDiff } = require('../src/math');
const { half, scalar, sampleWindows } = require('../src/binary');
const F = require('./fixtures');
let dir;
before(async () => { dir = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-test-')); });
after(async () => { await fs.rm(dir, { recursive: true, force: true }); });
const file = name => path.join(dir, name);

test('safetensors metadata, scalar and empty tensors, parameter totals', async () => {
  await F.safetensors(file('basic.safetensors'), [{ name: 'encoder.0.weight', shape: [2, 2], values: [-1, 0, 1, 2] }, { name: 'scalar', shape: [], values: [7] }, { name: 'empty', shape: [0, 8], values: [] }], { architecture: 'Tiny' });
  const m = await backend.inspect(file('basic.safetensors'));
  assert.equal(m.summary.parameters, 5); assert.equal(m.summary.tensorBytes, 20);
  assert.equal(m.metadata.architecture, 'Tiny'); assert.equal(backend.publicModel(m).tensors[0].storage, undefined);
  const stats = await backend.tensorStatistics(m, 'encoder.0.weight');
  assert.equal(stats.mean, .5); assert.equal(stats.min, -1); assert.equal(stats.std, Math.sqrt(1.25)); assert.equal(stats.exact, true);
  assert.equal((await backend.tensorStatistics(m, 'empty')).min, null);
});
test('safetensors rejects wrong size, truncated data and invalid offsets', async () => {
  await F.safetensors(file('bad.safetensors'), [{ name: 'wrong', shape: [9], values: [1, 2] }]);
  await assert.rejects(backend.inspect(file('bad.safetensors')), /byte length/);
  await fs.writeFile(file('short.safetensors'), F.u64(99999));
  await assert.rejects(backend.inspect(file('short.safetensors')), /Truncated/);
  const json = Buffer.from(JSON.stringify({ a: { dtype: 'F32', shape: [1], data_offsets: [4, 8] } }));
  await fs.writeFile(file('hole.safetensors'), Buffer.concat([F.u64(json.length), json, F.f32([1, 2])]));
  await assert.rejects(backend.inspect(file('hole.safetensors')), /uncovered range/);
});
test('F16, BF16 and non-finite values are decoded without frameworks', async () => {
  assert.equal(half(0x3c00), 1); assert.equal(half(0x8000), -0); assert.equal(half(1), 2 ** -24); assert.equal(half(0x7c00), Infinity); assert.ok(Number.isNaN(half(0x7e00)));
  const data = Buffer.from([0x80, 0x3f, 0, 0, 0x80, 0x7f, 0xc0, 0x7f]);
  await F.safetensors(file('brain.safetensors'), [{ name: 'weight', dtype: 'BF16', shape: [4], data }]);
  const s = await backend.tensorStatistics(await backend.inspect(file('brain.safetensors')), 'weight');
  assert.equal(s.mean, .5); assert.equal(s.nan, 1); assert.equal(s.inf, 1); assert.equal(s.finite, 2);
});
test('float8 special cases and signed integer decoding', () => {
  assert.equal(scalar(Buffer.from([0x38]), 0, 'F8_E4M3FN'), 1);
  assert.ok(Number.isNaN(scalar(Buffer.from([0x7f]), 0, 'F8_E4M3FN')));
  assert.equal(scalar(Buffer.from([0x7c]), 0, 'F8_E5M2'), Infinity);
  assert.ok(Number.isNaN(scalar(Buffer.from([0x80]), 0, 'F8_E5M2FNUZ')));
  assert.equal(scalar(Buffer.from([255]), 0, 'I8'), -1);
});
test('GGUF scalar metadata, array previews, dimension order and F32 stats', async () => {
  const arr = Buffer.concat([F.u32(8), F.u64(40), ...Array.from({ length: 40 }, (_, i) => F.str(`token${i}`))]);
  await F.gguf(file('basic.gguf'), [{ name: 'blk.0.attn.weight', shape: [2, 2], values: [1, 2, 3, 4] }], [['tokenizer.ggml.tokens', 9, arr], ['general.alignment', 4, F.u32(32)]]);
  const m = await backend.inspect(file('basic.gguf'));
  assert.deepEqual(m.tensors[0].shape, [2, 2]); assert.equal(m.metadata['tokenizer.ggml.tokens'].count, 40); assert.equal(m.metadata['tokenizer.ggml.tokens'].preview.length, 32);
  assert.equal((await backend.tensorStatistics(m, m.tensors[0].name)).mean, 2.5);
});
test('GGUF Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 use decoded weights', async () => {
  for (const [type, size, expected] of [[2, 18, -7.5], [3, 20, 2.5], [6, 22, -15.5], [7, 24, 2.5], [8, 34, -2]]) {
    const b = Buffer.alloc(size); b.writeUInt16LE(0x3c00, 0);
    if ([3, 7].includes(type)) b.writeUInt16LE(0x4000, 2);
    const start = type === 2 ? 2 : type === 3 ? 4 : type === 6 ? 6 : type === 7 ? 8 : type === 8 ? 2 : 4;
    b.fill(type >= 8 ? 254 : 0x10, start);
    await F.gguf(file(`q${type}.gguf`), [{ name: 'weight', type, shape: [32], data: b }]);
    const m = await backend.inspect(file(`q${type}.gguf`));
    assert.equal(m.tensors[0].quantization.blockSize, 32);
    assert.ok(Math.abs((await backend.tensorStatistics(m, 'weight')).mean - expected) < 1e-12);
  }
});
test('GGUF quantization metadata works for K types; IQ has explicit stats limitation', async () => {
  await F.gguf(file('k.gguf'), [{ name: 'weight', type: 12, shape: [256], data: Buffer.alloc(144) }]);
  const m = await backend.inspect(file('k.gguf'));
  assert.equal(m.tensors[0].quantization.bitsPerWeight, 4.5);
  assert.equal((await backend.tensorStatistics(m, 'weight')).mean, 0);
  await F.gguf(file('iq.gguf'), [{ name: 'weight', type: 16, shape: [256], data: Buffer.alloc(66) }]);
  await assert.rejects(backend.tensorStatistics(await backend.inspect(file('iq.gguf')), 'weight'), /not available/);
});
test('GGUF rejects malformed signatures, block sizes and unsafe counts', async () => {
  await fs.writeFile(file('bad.gguf'), Buffer.concat([Buffer.from('NOPE'), Buffer.alloc(100)]));
  await assert.rejects(backend.inspect(file('bad.gguf')), /magic/);
  await F.gguf(file('partial.gguf'), [{ name: 'weight', type: 2, shape: [33], data: Buffer.alloc(36) }]);
  await assert.rejects(backend.inspect(file('partial.gguf')), /blocks/);
  await fs.writeFile(file('huge.gguf'), Buffer.concat([Buffer.from('GGUF'), F.u32(3), F.u64(2n ** 60n), F.u64(0)]));
  await assert.rejects(backend.inspect(file('huge.gguf')), /safe integer/);
});
test('GGUF Q8_K values follow signed codes; ambiguous Q8_1 layout is explicit', async () => {
  const raw = Buffer.alloc(292); raw.writeFloatLE(.25, 0);
  for (let i = 0; i < 256; i++) raw.writeInt8(i - 128, i + 4);
  await F.gguf(file('q8k.gguf'), [{ name: 'weight', type: 15, shape: [256], data: raw }]);
  assert.deepEqual(await backend.sample(await backend.inspect(file('q8k.gguf')), 'weight'), Array.from({ length: 256 }, (_, i) => (i - 128) * .25));
  await F.gguf(file('q81.gguf'), [{ name: 'weight', type: 9, shape: [32], data: Buffer.alloc(40) }]);
  const model = await backend.inspect(file('q81.gguf')); assert.equal(model.byName.get('weight').bytes, null);
  await assert.rejects(backend.sample(model, 'weight'), /not available/);
});
test('native ONNX reads raw and packed float initializers, operators and quantization', async () => {
  await F.onnx(file('basic.onnx'), [{ name: 'layer.weight', shape: [2, 2], values: [1, 2, 3, 4] }, { name: 'bias', values: [5, 6], typed: 4 }, { name: 'codes', dtype: 3, shape: [2], data: Buffer.from([1, 255]) }], [{ type: 'DequantizeLinear', inputs: ['codes', 'scale', 'zero'] }]);
  const m = await backend.inspect(file('basic.onnx'));
  assert.equal(m.summary.parameters, 8); assert.equal(m.metadata.irVersion, 9); assert.equal(m.nodes[0].opType, 'DequantizeLinear');
  assert.equal(m.byName.get('codes').quantization.scaleTensor, 'scale');
  assert.equal((await backend.tensorStatistics(m, 'layer.weight')).mean, 2.5);
  assert.equal((await backend.tensorStatistics(m, 'bias')).mean, 5.5);
  assert.deepEqual(await backend.sample(m, 'codes'), [1, -1]);
});
test('native ONNX packed integers correctly handle negative varints and F16 bit patterns', async () => {
  await F.onnx(file('ints.onnx'), [{ name: 'ints', dtype: 6, shape: [3], typed: 5, data: Buffer.concat([-2, 0, 9].map(F.vint)) }, { name: 'halves', dtype: 10, shape: [2], typed: 5, data: Buffer.concat([0x3c00, 0xc000].map(F.vint)) }]);
  const m = await backend.inspect(file('ints.onnx'));
  assert.deepEqual(await backend.sample(m, 'ints'), [-2, 0, 9]); assert.deepEqual(await backend.sample(m, 'halves'), [1, -2]);
});
test('ONNX tensor-valued Constant operators count as stored tensors', async () => {
  const attr = F.field(5, Buffer.concat([F.field(1, 'value'), F.field(5, F.onnxTensor({ name: 'constant', values: [7] }))]));
  await F.onnx(file('constant.onnx'), [], [{ type: 'Constant', attrs: [attr] }]);
  const m = await backend.inspect(file('constant.onnx'));
  assert.equal(m.summary.parameters, 1); assert.equal((await backend.tensorStatistics(m, 'constant')).mean, 7);
});
test('ONNX external weights are loaded on request and traversal/symlink escapes rejected', async () => {
  await fs.writeFile(file('weights.bin'), F.f32([3, 5]));
  await F.onnx(file('external.onnx'), [{ name: 'weight', shape: [2], external: { location: 'weights.bin', length: 8 } }]);
  const m = await backend.inspect(file('external.onnx')); assert.equal((await backend.tensorStatistics(m, 'weight')).mean, 4);
  await F.onnx(file('escape.onnx'), [{ name: 'weight', shape: [2], external: { location: '../weights.bin' } }]);
  await assert.rejects(backend.sample(await backend.inspect(file('escape.onnx')), 'weight'), /inside/);
  await fs.symlink(os.tmpdir(), file('escape'));
  await F.onnx(file('symlink.onnx'), [{ name: 'weight', shape: [2], external: { location: 'escape' } }]);
  await assert.rejects(backend.sample(await backend.inspect(file('symlink.onnx')), 'weight'), /outside/);
});
test('ONNX malformed protobuf and tensor sizes are rejected', async () => {
  await fs.writeFile(file('short.onnx'), Buffer.from([58, 20, 1])); await assert.rejects(backend.inspect(file('short.onnx')), /Truncated/);
  await F.onnx(file('badshape.onnx'), [{ name: 'weight', shape: [10], values: [1] }]); await assert.rejects(backend.inspect(file('badshape.onnx')), /byte length/);
});
test('sampling is bounded, deterministic, spans the tensor and labels partial coverage', async () => {
  const values = Array.from({ length: 20000 }, (_, i) => i);
  await F.safetensors(file('sample.safetensors'), [{ name: 'weight', values }]);
  const m = await backend.inspect(file('sample.safetensors'));
  const sampled = await backend.sample(m, 'weight', { sampleSize: 256 });
  assert.equal(sampled.length, 256); assert.equal(sampled[0], 0); assert.equal(sampled.at(-1), 19999);
  assert.deepEqual(sampled, await backend.sample(m, 'weight', { sampleSize: 256 }));
  assert.equal((await backend.tensorStatistics(m, 'weight', { sampleSize: 256 })).exact, false);
  assert.deepEqual(sampleWindows(0, 500), []);
});
test('cross-format checkpoint diffs quantify drift and shape/dtype changes', async () => {
  await F.safetensors(file('base.safetensors'), [{ name: 'weight', values: [0, 1, 2, 3] }, { name: 'removed', values: [1] }, { name: 'shape', values: [1] }]);
  await F.onnx(file('target.onnx'), [{ name: 'weight', values: [1, 2, 3, 4] }, { name: 'added', values: [1] }, { name: 'shape', values: [1, 2] }]);
  const a = await backend.inspect(file('base.safetensors')), b = await backend.inspect(file('target.onnx'));
  const rows = structureDiff(a, b); assert.equal(rows.find(r => r.name === 'removed').status, 'removed'); assert.equal(rows.find(r => r.name === 'shape').status, 'shape-changed');
  const d = await backend.compareTensor(a, b, 'weight'); assert.equal(d.meanAbs, 1); assert.equal(d.maxAbs, 1); assert.equal(d.rmse, 1); assert.equal(d.changedValues, 4); assert.equal(d.exact, true);
  await assert.rejects(backend.compareTensor(a, b, 'shape'), /incompatible/);
});
test('sample matches never imply exact equality, tolerance is honored', () => {
  assert.equal(difference([1, 2], [1, 2], 100).status, 'sample-match');
  assert.equal(difference([1, 2], [1.01, 2], 2, .02).status, 'equal');
  assert.equal(difference([1, NaN, Infinity], [1, NaN, -Infinity], 3).nonFiniteMismatch, 1);
  assert.equal(difference([], [], 0).meanAbs, null);
  assert.equal(statistics([NaN, Infinity], 2).mean, null);
});
test('integer precision overflow is reported, not silently rounded', async () => {
  const b = Buffer.alloc(8); b.writeBigInt64LE(2n ** 60n);
  await F.safetensors(file('bigint.safetensors'), [{ name: 'weight', dtype: 'I64', shape: [1], data: b }]);
  await assert.rejects(backend.sample(await backend.inspect(file('bigint.safetensors')), 'weight'), /precision/);
});
test('checkpoint mutation invalidates analysis and cancellation stops operations', async () => {
  await F.safetensors(file('mutable.safetensors'), [{ name: 'weight', values: [1] }]); const m = await backend.inspect(file('mutable.safetensors'));
  await fs.appendFile(file('mutable.safetensors'), Buffer.from([0])); await assert.rejects(backend.sample(m, 'weight'), /changed on disk/);
  await assert.rejects(backend.inspect(file('basic.safetensors'), { signal: AbortSignal.abort() }), /cancelled/);
});
test('PyTorch process execution requires workspace trust', async () => {
  await fs.writeFile(file('trusted.pt'), 'test');
  await assert.rejects(backend.inspect(file('trusted.pt')), /trusted/);
});
