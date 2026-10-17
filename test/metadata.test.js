'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const backend = require('../src/backend');
const { saveCheckpoint } = require('../src/checkpoint');
const { validateForModel } = require('../src/metadata');
const f = require('./fixtures');
async function directory(t) { const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-metadata-')); t.after(() => fs.rm(dir, { recursive: true, force: true })); return dir; }
test('safetensors metadata adds, updates and removes keys while preserving every tensor byte', async t => {
  const dir = await directory(t), file = path.join(dir, 'base.safetensors'), output = path.join(dir, 'edited.safetensors');
  await f.safetensors(file, [{ name: 'w', values: [1, -2, NaN, Infinity] }], { name: 'old', remove: 'yes' });
  const model = await backend.inspect(file), original = await fs.readFile(file);
  await saveCheckpoint(model, output, { metadata: [{ key: 'name', value: 'new 🦊' }, { key: 'remove', value: null }, { key: '__proto__', value: 'literal key' }] });
  const edited = await backend.inspect(output);
  assert.deepEqual(edited.metadata, JSON.parse('{"name":"new 🦊","__proto__":"literal key"}'));
  const data = await fs.readFile(output);
  assert.deepEqual(data.subarray(edited.tensors[0].storage.offset), original.subarray(model.tensors[0].storage.offset));
  assert.deepEqual(await fs.readFile(file), original);
  await assert.rejects(saveCheckpoint(model, path.join(dir, 'bad.safetensors'), { metadata: [{ key: 'number', value: 1 }] }), /must be strings/);
});
test('GGUF metadata retains exact scalar types, full tokenizer arrays, and aligned tensor payloads', async t => {
  const dir = await directory(t), file = path.join(dir, 'base.gguf'), output = path.join(dir, 'edited.gguf');
  const tokens = Buffer.concat([f.u32(8), f.u64(100), ...Array.from({ length: 100 }, (_, i) => f.str(`token ${i}`))]);
  await f.gguf(file, [{ name: 'w', shape: [2], values: [1, 2] }], [['general.name', 8, f.str('old')], ['count', 10, f.u64(18446744073709551615n)], ['tokenizer.ggml.tokens', 9, tokens], ['remove', 7, Buffer.from([1])]]);
  const model = await backend.inspect(file), original = await fs.readFile(file);
  await saveCheckpoint(model, output, { metadata: [{ key: 'general.name', value: 'A much longer model name' }, { key: 'count', value: '18446744073709551614' }, { key: 'remove', value: null }, { key: 'custom.flag', type: 'bool', value: true }] });
  const edited = await backend.inspect(output), data = await fs.readFile(output);
  assert.equal(edited.metadata.count, '18446744073709551614'); assert.equal(edited.metadata['custom.flag'], true); assert.equal(edited.metadata.remove, undefined);
  assert.equal(edited.metadata['tokenizer.ggml.tokens'].count, 100);
  const before = model.layout.metadataEntries.find(e => e.key === 'tokenizer.ggml.tokens'), after = edited.layout.metadataEntries.find(e => e.key === before.key);
  assert.deepEqual(data.subarray(after.start, after.end), original.subarray(before.start, before.end));
  assert.deepEqual(data.subarray(edited.layout.dataStart), original.subarray(model.layout.dataStart));
  assert.equal(edited.tensors[0].storage.offset % 32, 0);
  assert.deepEqual(await backend.sample(edited, 'w'), [1, 2]);
  for (const change of [{ key: 'general.alignment', value: 64 }, { key: 'tokenizer.ggml.tokens', value: 'preview' }, { key: 'count', value: Number.MAX_SAFE_INTEGER + 1 }, { key: 'general.name', type: 'i32', value: 4 }, { key: 'custom.small', type: 'u8', value: 256 }]) assert.throws(() => validateForModel(model, [change]));
});
test('ONNX metadata property edits preserve graph, unknown fields and external references', async t => {
  const dir = await directory(t), file = path.join(dir, 'base.onnx'), output = path.join(dir, 'edited.onnx');
  await f.onnx(file, [{ name: 'w', values: [3] }], [{ type: 'Identity', inputs: ['w'] }]);
  const unknown = f.field(500, 'opaque extension');
  await fs.appendFile(file, Buffer.concat([f.field(14, Buffer.concat([f.field(1, 'author'), f.field(2, 'before')])), unknown]));
  const model = await backend.inspect(file);
  await saveCheckpoint(model, output, { metadata: [{ key: 'author', value: null }, { key: 'description', value: 'after' }] });
  const edited = await backend.inspect(output);
  assert.equal(edited.metadata.author, undefined); assert.equal(edited.metadata.description, 'after'); assert.deepEqual(edited.nodes, model.nodes);
  assert.deepEqual(await backend.sample(edited, 'w'), [3]); assert.ok((await fs.readFile(output)).includes(unknown));
});
test('metadata plans reject malformed, duplicate, oversized and non-finite changes', () => {
  const model = { format: 'safetensors' };
  for (const changes of [null, [{ key: '', value: 'v' }], [{ key: 'w', value: {} }], [{ key: 'w', value: Infinity }], [{ key: 'w', value: 'a' }, { key: 'w', value: 'b' }], [{ key: 'w', value: 'x'.repeat(2 * 1024 * 1024) }]]) assert.throws(() => validateForModel(model, changes));
});
