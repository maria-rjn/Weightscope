'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const backend = require('../src/backend');
const { saveCheckpoint, transaction } = require('../src/checkpoint');
const fixtures = require('./fixtures');

async function directory(t) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-save-'));
  t.after(() => fs.rm(dir, { recursive: true, force: true })); return dir;
}
test('Save As creates byte-identical, readable native checkpoints and keeps originals intact', async t => {
  const dir = await directory(t);
  for (const ext of ['safetensors', 'gguf', 'onnx']) {
    const file = path.join(dir, `source.${ext}`), destination = path.join(dir, `copy.${ext}`);
    await fixtures[ext](file, [{ name: 'layer.weight', shape: [2], values: [1, -2] }]);
    const original = await fs.readFile(file), model = await backend.inspect(file);
    assert.equal((await saveCheckpoint(model, destination)).path, destination);
    assert.deepEqual(await fs.readFile(destination), original); assert.deepEqual(await fs.readFile(file), original);
    assert.equal((await backend.tensorStatistics(await backend.inspect(destination), 'layer.weight')).mean, -.5);
  }
  assert.equal((await fs.readdir(dir)).some(n => n.startsWith('.weightscope-')), false);
});
test('Save As never overwrites files or aliases, and rejects stale inputs and incorrect extensions', async t => {
  const dir = await directory(t), file = path.join(dir, 'source.safetensors');
  await fixtures.safetensors(file, [{ name: 'w', values: [1] }]);
  const model = await backend.inspect(file), original = await fs.readFile(file);
  for (const alias of ['source', 'symlink', 'hardlink']) {
    const destination = alias === 'source' ? file : path.join(dir, `${alias}.safetensors`);
    if (alias === 'symlink') await fs.symlink(file, destination);
    if (alias === 'hardlink') await fs.link(file, destination);
    // Creating a hardlink changes ctime, so inspect again before exercising destination protection.
    await assert.rejects(saveCheckpoint(await backend.inspect(file), destination), /never overwrites/);
  }
  assert.deepEqual(await fs.readFile(file), original);
  await assert.rejects(saveCheckpoint(model, path.join(dir, 'bad.gguf')), /extension/);
  await fs.appendFile(file, ' ');
  await assert.rejects(saveCheckpoint(model, path.join(dir, 'stale.safetensors')), /changed on disk/);
  assert.equal((await fs.readdir(dir)).some(n => n.startsWith('.weightscope-')), false);
});
test('cancelled or invalid output leaves no partial checkpoint or staging file', async t => {
  const dir = await directory(t), file = path.join(dir, 'source.safetensors'), destination = path.join(dir, 'copy.safetensors');
  await fixtures.safetensors(file, [{ name: 'w', values: [1] }]); const model = await backend.inspect(file);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(saveCheckpoint(model, destination, { signal: controller.signal }), /cancelled/);
  await assert.rejects(transaction(model, destination, {}, async (_reader, output) => { await output.write(Buffer.from('broken')); }), /Truncated/);
  await assert.rejects(fs.stat(destination), { code: 'ENOENT' });
  assert.deepEqual(await fs.readdir(dir), ['source.safetensors']);
});
test('ONNX external references stay valid and cannot be silently relocated', async t => {
  const dir = await directory(t), file = path.join(dir, 'source.onnx');
  await fs.writeFile(path.join(dir, 'data.bin'), fixtures.f32([7]));
  await fixtures.onnx(file, [{ name: 'w', shape: [1], external: { location: 'data.bin' } }]);
  const model = await backend.inspect(file), copy = path.join(dir, 'copy.onnx'); await saveCheckpoint(model, copy);
  assert.equal((await backend.tensorStatistics(await backend.inspect(copy), 'w')).mean, 7);
  const other = path.join(dir, 'other'); await fs.mkdir(other);
  await assert.rejects(saveCheckpoint(model, path.join(other, 'copy.onnx')), /original model directory/);
});
