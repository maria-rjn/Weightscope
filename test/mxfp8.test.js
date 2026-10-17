'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const backend = require('../src/backend');
const { Reader, sampleWindows } = require('../src/binary');
const { store } = require('./ollama-fixtures');
const F = require('./fixtures');
async function fixture(t) { const root = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-mxfp8-')); t.after(() => fs.rm(root, { recursive: true, force: true })); return store(root); }
function weight(name, codes, shape = [codes.length / 4]) { return { name, dtype: 'U32', shape, data: Buffer.from(codes) }; }
function scale(name, codes, shape = [codes.length]) { return { name: name + '.scale', dtype: 'U8', shape, data: Buffer.from(codes) }; }

test('MXFP8 decodes packed bytes in order, applies block scales and retains stored counts', async t => {
  const s = await fixture(t), name = 'model.embed_tokens.weight';
  const codes = [0x00, 0x80, 0x01, 0x07, 0x08, 0x38, 0xb8, 0x7e];
  const values = [0, -0, 2 ** -9, 7 * 2 ** -9, 2 ** -6, 1, -1, 448];
  const raw = [...codes, ...Array(24).fill(0), ...Array(32).fill(0x38), ...Array(32).fill(0xb8), ...Array(32).fill(0x40)];
  const blob = await s.blob('tensor', [weight(name, raw, [2, 16]), scale(name, [127, 128, 126, 129], [2, 2]), { name: 'norm.weight', values: [3, 5] }], { metadata: { quant_type: 'mxfp8' } });
  const model = await backend.inspect(await s.manifest('packed', 'latest', [blob])), tensor = model.byName.get(name);
  assert.deepEqual(tensor.shape, [2, 16]); assert.equal(tensor.elements, 32);
  assert.deepEqual(tensor.logicalShape, [2, 64]); assert.equal(tensor.logicalElements, 128);
  assert.equal(model.summary.parameters, 38); assert.equal(model.summary.quantizedTensors, 1); assert.deepEqual(model.warnings, []);
  assert.equal(tensor.quantization.decoded, true); assert.equal(tensor.quantization.bitsPerWeight, 8); assert.equal(tensor.quantization.blockSize, 32);
  assert.equal(tensor.quantization.effectiveBitsPerWeight, 8.25);
  assert.deepEqual(await backend.sample(model, name), [...values, ...Array(24).fill(0), ...Array(32).fill(2), ...Array(32).fill(-0.5), ...Array(32).fill(8)]);
  const stats = await backend.tensorStatistics(model, name);
  assert.equal(stats.total, 128); assert.equal(stats.sampled, 128); assert.equal(stats.exact, true); assert.equal(stats.max, 448);
  assert.equal((await backend.tensorStatistics(model, 'norm.weight')).mean, 4);
  assert.deepEqual(await backend.sample(model, name + '.scale'), [127, 128, 126, 129]);
  assert.match((await backend.tensorStatistics(model, name + '.scale')).basis, /scale codes/);
  const publicTensor = backend.publicModel(model).tensors.find(t => t.name === name);
  assert.deepEqual(publicTensor.logicalShape, [2, 64]); assert.equal(publicTensor.storage, undefined);
});

test('MXFP8 preserves signed zero and NaN codes, tiny E8M0 scales and float32 overflow', async t => {
  const s = await fixture(t), name = 'w';
  const raw = Buffer.alloc(128, 0x38); raw[0] = 0x7f; raw[1] = 0xff; raw[2] = 0x80; raw[96] = 0x7e;
  const blob = await s.blob('tensor', [weight(name, raw, [4, 8]), scale(name, [127, 0, 255, 254], [4, 1])], { metadata: { quant_type: 'mxfp8' } });
  const model = await backend.inspect(await s.manifest('special', 'latest', [blob]));
  const values = await backend.sample(model, name);
  assert.ok(Number.isNaN(values[0])); assert.ok(Number.isNaN(values[1])); assert.ok(Object.is(values[2], -0));
  assert.equal(values[32], 2 ** -127); assert.ok(Number.isNaN(values[64])); assert.equal(values[96], Infinity);
  const stats = await backend.tensorStatistics(model, name);
  assert.equal(stats.nan, 34); assert.equal(stats.inf, 1); assert.equal(stats.zero, 1);
});

test('MXFP8 bounded sampling uses logical indices across word, block, row and expert boundaries', async t => {
  const s = await fixture(t), name = 'experts.weight', raw = Buffer.alloc(2 * 3 * 64);
  const expected = Array.from(raw, (_, i) => (i % 4 + 1) * 2 ** (Math.floor(i / 32) % 5 - 2));
  raw.forEach((_, i) => { raw[i] = [0x38, 0x40, 0x44, 0x48][i % 4]; });
  const scales = Array.from({ length: raw.length / 32 }, (_, i) => 125 + i % 5);
  const blob = await s.blob('tensor', [weight(name, raw, [2, 3, 16]), scale(name, scales, [2, 3, 2])], { metadata: { quant_type: 'MXFP8', group_size: '32' } });
  const model = await backend.inspect(await s.manifest('experts', 'latest', [blob]));
  const calls = [], read = Reader.prototype.read;
  Reader.prototype.read = async function(start, length) { calls.push({ start, length }); return read.call(this, start, length); };
  t.after(() => { Reader.prototype.read = read; });
  for (const limit of [1, 7, 33, 65, 256]) {
    calls.length = 0;
    const sampled = await backend.sample(model, name, { sampleSize: limit });
    assert.deepEqual(sampled, sampleWindows(expected.length, limit).flatMap(w => expected.slice(w.start, w.start + w.count)));
    assert.ok(sampled.length <= limit);
    assert.ok(calls.every(c => c.length <= Math.ceil(limit / Math.min(32, limit))));
  }
  const controller = new AbortController(); controller.abort();
  await assert.rejects(backend.sample(model, name, { signal: controller.signal }), /cancelled/);
});

test('MXFP8 compares decoded shapes and values against floating-point and packed checkpoints', async t => {
  const s = await fixture(t), name = 'w', raw = Buffer.alloc(64, 0x38);
  const blob = await s.blob('tensor', [weight(name, raw, [2, 8]), scale(name, [127, 128], [2, 1])], { metadata: { quant_type: 'mxfp8' } });
  const model = await backend.inspect(await s.manifest('base', 'latest', [blob]));
  const file = path.join(s.root, 'float.safetensors');
  await F.safetensors(file, [{ name, shape: [2, 32], values: [...Array(32).fill(1), ...Array(32).fill(2)] }]);
  const other = await backend.inspect(file), row = backend.structureDiff(model, other).find(r => r.name === name);
  assert.equal(row.status, 'unverified'); assert.equal(row.dtypeChanged, true); assert.deepEqual(row.left.shape, [2, 32]);
  assert.equal((await backend.compareTensor(model, other, name)).status, 'equal');
  assert.equal((await backend.compareTensor(other, model, name)).total, 64);
  const changed = await s.blob('tensor', [weight(name, raw, [2, 8]), scale(name, [127, 129], [2, 1])], { metadata: { quant_type: 'mxfp8' } });
  const target = await backend.inspect(await s.manifest('target', 'latest', [changed]));
  const diff = await backend.compareTensor(model, target, name);
  assert.equal(diff.total, 64); assert.equal(diff.changedValues, 32); assert.equal(diff.maxAbs, 2);
});

test('MXFP8 scale companions in other blobs work and changes are rejected before and during sampling', async t => {
  const s = await fixture(t), name = 'w';
  const a = await s.blob('tensor', [weight(name, Buffer.alloc(32, 0x38))], { metadata: { quant_type: 'mxfp8' } });
  const b = await s.blob('tensor', [scale(name, [128])]);
  const model = await backend.inspect(await s.manifest('split', 'latest', [b, a]));
  assert.equal((await backend.tensorStatistics(model, name)).mean, 2);
  const scaleFile = model.byName.get(name + '.scale').storage.file, read = Reader.prototype.read;
  Reader.prototype.read = async function(start, length) { const bytes = await read.call(this, start, length); await fs.appendFile(scaleFile, 'changed'); return bytes; };
  try { await assert.rejects(backend.sample(model, name), /scale blob changed during/); } finally { Reader.prototype.read = read; }
  await assert.rejects(backend.sample(model, name), /scale blob changed on disk/);
});

test('MXFP8 honors per-tensor metadata in mixed blobs and leaves unsupported schemes explicit', async t => {
  const s = await fixture(t), blob = await s.blob('tensor', [weight('a', Buffer.alloc(32, 0x38)), scale('a', [127]), weight('b', Buffer.alloc(32)), scale('b', [127]), { name: 'norm.weight', values: [5] }],
    { metadata: { 'a.quant_type': 'mxfp8', 'a.group_size': '32', 'b.quant_type': 'mxfp4', 'b.group_size': '32' } });
  const model = await backend.inspect(await s.manifest('mixed', 'latest', [blob]));
  assert.equal((await backend.tensorStatistics(model, 'a')).mean, 1);
  assert.equal(model.byName.get('b').quantization.decoded, false);
  await assert.rejects(backend.sample(model, 'b'), /Numerical decoding.*unavailable/);
  assert.equal(model.byName.get('b.scale').quantization, undefined);
  assert.equal((await backend.tensorStatistics(model, 'norm.weight')).mean, 5);
});

test('MXFP8 resolves manifest-named data aliases and model-index quantization with separate scales', async t => {
  const s = await fixture(t);
  const a = await s.blob('tensor', [weight('data', Buffer.alloc(32, 0x38))]);
  const b = await s.blob('tensor', [{ name: 'data', dtype: 'U8', shape: [1], data: Buffer.from([128]) }]);
  const config = await s.blob('json', { quantization: 'mxfp8' }, { name: 'model_index.json' });
  const model = await backend.inspect(await s.manifest('aliases', 'latest', [
    { ...a, name: 'encoder.a.weight' }, { ...a, name: 'encoder.b.weight' },
    { ...b, name: 'encoder.a.weight_scale' }, { ...b, name: 'encoder.b.weight.scale' }, config
  ]));
  assert.equal(model.summary.quantizedTensors, 2); assert.equal(model.summary.parameters, 18);
  assert.equal(model.fileBytes, a.size + b.size);
  for (const name of ['encoder.a.weight', 'encoder.b.weight']) {
    assert.deepEqual(model.byName.get(name).logicalShape, [32]);
    assert.equal((await backend.tensorStatistics(model, name)).mean, 2);
  }
});

test('MXFP8 malformed layouts stay metadata-only with actionable explanations', async t => {
  const cases = [
    { name: 'missing scale', tensors: [weight('w', Buffer.alloc(32))], reason: /missing.*scale/ },
    { name: 'wrong scale dtype', tensors: [weight('w', Buffer.alloc(32)), { name: 'w.scale', dtype: 'F32', values: [1] }], reason: /U8 scale/ },
    { name: 'wrong scale shape', tensors: [weight('w', Buffer.alloc(64), [2, 8]), scale('w', [127, 127], [1, 2])], reason: /shape \[2, 1\]/ },
    { name: 'unaligned row', tensors: [weight('w', Buffer.alloc(16)), scale('w', [127])], reason: /multiple of 32/ },
    { name: 'bad group size', tensors: [weight('w', Buffer.alloc(32)), scale('w', [127])], metadata: { group_size: '64' }, reason: /group size of 32/ },
    { name: 'unexpected bias', tensors: [weight('w', Buffer.alloc(32)), scale('w', [127]), { name: 'w.bias', values: [0] }], reason: /bias tensor/ },
    { name: 'scalar weight', tensors: [weight('w', Buffer.alloc(4), [])], reason: /non-scalar U32/ },
    { name: 'wrong weight dtype', tensors: [{ name: 'w', values: Array(32).fill(1) }, scale('w', [127])], metadata: { 'w.quant_type': 'mxfp8' }, reason: /U32/ }
  ];
  for (const c of cases) await t.test(c.name, async t => {
    const s = await fixture(t), blob = await s.blob('tensor', c.tensors, { metadata: { quant_type: 'mxfp8', ...c.metadata } });
    const model = await backend.inspect(await s.manifest('invalid', 'latest', [blob])), tensor = model.byName.get('w');
    assert.equal(tensor.quantization.decoded, false); assert.equal(tensor.logicalShape, undefined); assert.match(tensor.analysisUnavailable, c.reason);
    await assert.rejects(backend.sample(model, 'w'), c.reason);
  });
});
