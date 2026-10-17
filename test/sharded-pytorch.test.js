'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const backend = require('../src/backend');
const sharded = require('../src/sharded');
const { capabilities } = require('../src/checkpoint');
const F = require('./fixtures');
const manifest = require('../package.json');

const SHARDS = [
  { 'model.embed_tokens.weight': [1, 2, 3, 4], 'model.layers.0.mlp.weight': [1, 1] },
  { 'model.layers.1.mlp.weight': [5, 7] },
  { 'model.norm.weight': [.5, .5], 'lm_head.weight': [-2, 2] }
];
const pad = n => String(n).padStart(5, '0');
const binShard = (i, n = SHARDS.length) => `pytorch_model-${pad(i + 1)}-of-${pad(n)}.bin`;
// The fake helper is an executable script, so these tests need a shebang-capable platform.
const fake = { skip: (process.platform === 'win32' || /\s/.test(process.execPath)) && 'the fake PyTorch helper needs a shebang script' };
async function folder(t, name = 'tiny-llm') {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-pt-sharded-')); t.after(() => fs.rm(root, { recursive: true, force: true }));
  const dir = path.join(root, name); await fs.mkdir(dir, { recursive: true }); return { root, dir };
}
// Stands in for python/inspect_pt.py: fake shards are JSON tensor maps, and every request is logged next to the script.
async function fakePython(root) {
  const file = path.join(root, 'fake-python');
  await fs.writeFile(file, `#!${process.execPath}
const fs = require('node:fs'), path = require('node:path');
let input = '';
process.stdin.on('data', chunk => { input += chunk; }).on('end', () => {
  const request = JSON.parse(input);
  fs.appendFileSync(path.join(__dirname, 'requests.jsonl'), JSON.stringify(request) + '\\n');
  try {
    const shard = JSON.parse(fs.readFileSync(request.path, 'utf8'));
    const data = request.action === 'inspect'
      ? { format: 'pytorch', metadata: { torchVersion: 'fake' }, warnings: ['Counts describe stored tensors.'], tensors: Object.entries(shard).map(([name, values]) => ({ name, shape: [values.length], dtype: 'F32', elements: values.length, bytes: values.length * 4, storage: { kind: 'pytorch' } })) }
      : request.windows.flatMap(w => shard[request.name].slice(w.start, w.start + w.count));
    process.stdout.write(JSON.stringify({ ok: true, data }));
  } catch (e) { process.stdout.write(JSON.stringify({ ok: false, error: e.message })); }
});
`, { mode: 0o755 });
  const log = path.join(root, 'requests.jsonl');
  return { options: { trusted: true, pythonPath: file }, requests: async () => (await fs.readFile(log, 'utf8').catch(() => '')).split('\n').filter(Boolean).map(JSON.parse) };
}
// Writes fake .bin shards; `map` adjusts the index weight_map, `index: false` omits it.
async function checkpoint(dir, { shards = SHARDS, map = m => m, index = true } = {}) {
  const weightMap = {};
  for (const [i, tensors] of shards.entries()) {
    await fs.writeFile(path.join(dir, binShard(i, shards.length)), JSON.stringify(tensors));
    for (const name of Object.keys(tensors)) weightMap[name] = binShard(i, shards.length);
  }
  const file = path.join(dir, 'pytorch_model.bin.index.json');
  if (index) await fs.writeFile(file, JSON.stringify({ metadata: { total_size: 48 }, weight_map: map(weightMap) }));
  return file;
}

test('a PyTorch shard index opens as one checkpoint and the helper samples each tensor from its own shard', fake, async t => {
  const { root, dir } = await folder(t), index = await checkpoint(dir), python = await fakePython(root);
  const model = await backend.inspect(index, python.options);
  assert.equal(model.format, 'pytorch'); assert.equal(model.tensors.length, 5); assert.equal(model.summary.parameters, 12); assert.equal(model.summary.tensorBytes, 48);
  assert.equal(model.source.kind, 'sharded'); assert.equal(model.source.name, 'tiny-llm/pytorch_model.bin'); assert.equal(model.source.indexPath, index);
  assert.deepEqual(model.source.shards.map(s => s.tensors), [2, 1, 2]); assert.equal(model.metadata.index.total_size, 48);
  assert.deepEqual(model.warnings, ['Counts describe stored tensors.']);
  assert.equal(model.byName.get('lm_head.weight').shard, binShard(2));
  assert.deepEqual(await backend.sample(model, 'model.layers.1.mlp.weight', python.options), [5, 7]);
  assert.equal((await backend.tensorStatistics(model, 'lm_head.weight', python.options)).mean, 0);
  const shardPaths = SHARDS.map((_, i) => path.join(dir, binShard(i)));
  assert.deepEqual((await python.requests()).map(r => [r.action, r.path]), [...shardPaths.map(p => ['inspect', p]), ['sample', shardPaths[1]], ['sample', shardPaths[2]]]);
  assert.deepEqual(model.protectedPaths.sort(), [index, ...shardPaths].sort());
  assert.equal(backend.publicModel(model).tensors[0].storage, undefined);
  assert.equal(capabilities(model).save, false); assert.match(capabilities(model).reason, /Sharded PyTorch checkpoints/);
  // The same weights as safetensors shards compare tensor by tensor.
  const single = path.join(root, 'merged.safetensors');
  await F.safetensors(single, SHARDS.flatMap(s => Object.entries(s).map(([name, values]) => ({ name, values }))));
  const target = await backend.inspect(single);
  assert.ok(backend.structureDiff(model, target).every(r => r.status === 'unverified'));
  assert.equal((await backend.compareTensor(model, target, 'model.embed_tokens.weight', python.options)).status, 'equal');
  const fromShard = await backend.inspect(path.join(dir, binShard(1)), python.options);
  assert.equal(fromShard.tensors.length, 5); assert.equal(fromShard.source.indexPath, index);
});

test('PyTorch and safetensors shards in one folder each use the index of their own format', fake, async t => {
  const { root, dir } = await folder(t), binIndex = await checkpoint(dir), python = await fakePython(root);
  const weightMap = {};
  for (const [i, tensors] of SHARDS.entries()) {
    const file = binShard(i).replace('pytorch_model', 'model').replace('.bin', '.safetensors');
    await F.safetensors(path.join(dir, file), Object.entries(tensors).map(([name, values]) => ({ name, values })));
    for (const name of Object.keys(tensors)) weightMap[name] = file;
  }
  const safeIndex = path.join(dir, 'model.safetensors.index.json');
  await fs.writeFile(safeIndex, JSON.stringify({ weight_map: weightMap }));
  const pt = await backend.inspect(path.join(dir, binShard(0)), python.options), safe = await backend.inspect(path.join(dir, 'model-00001-of-00003.safetensors'));
  assert.equal(pt.format, 'pytorch'); assert.equal(pt.source.indexPath, binIndex); assert.equal(pt.tensors.length, 5);
  assert.equal(safe.format, 'safetensors'); assert.equal(safe.source.indexPath, safeIndex); assert.equal(safe.source.name, 'tiny-llm/model.safetensors');
  assert.equal((await backend.compareTensor(pt, safe, 'model.norm.weight', python.options)).status, 'equal');
  // A differently named PyTorch index is found, but a safetensors index is never used for .bin shards.
  await fs.rename(binIndex, path.join(dir, 'weights.bin.index.json'));
  assert.equal((await sharded.locate(path.join(dir, binShard(2)))).indexPath, path.join(dir, 'weights.bin.index.json'));
  await fs.rm(path.join(dir, 'weights.bin.index.json'));
  const grouped = await sharded.locate(path.join(dir, binShard(2)));
  assert.equal(grouped.format, 'pytorch'); assert.equal(grouped.indexPath, null); assert.deepEqual(grouped.files, SHARDS.map((_, i) => binShard(i)));
});

test('PyTorch shard sets need a trusted workspace and start no helper otherwise', fake, async t => {
  const { root, dir } = await folder(t), index = await checkpoint(dir), python = await fakePython(root), untrusted = { ...python.options, trusted: false };
  await assert.rejects(backend.inspect(index, untrusted), /requires a trusted VS Code workspace/);
  await assert.rejects(backend.inspect(path.join(dir, binShard(1)), untrusted), /requires a trusted VS Code workspace/);
  await fs.rm(index); await fs.rm(path.join(dir, binShard(2)));
  await assert.rejects(backend.inspect(path.join(dir, binShard(0)), untrusted), /requires a trusted VS Code workspace/);
  const lone = path.join(dir, 'solo-00001-of-00001.bin'); await fs.writeFile(lone, JSON.stringify({ w: [1] }));
  await assert.rejects(backend.inspect(lone, untrusted), /requires a trusted VS Code workspace/);
  assert.deepEqual(await python.requests(), []);
  assert.deepEqual((await backend.inspect(lone, python.options)).warnings, ['Counts describe stored tensors.']);
});

test('inconsistent PyTorch indexes fail explicitly when opened, while their shards still open alone', fake, async t => {
  const cases = [
    ['missing shard', async dir => { await fs.rm(path.join(dir, binShard(2))); }, /1 of 3 shards listed in pytorch_model\.bin\.index\.json are missing, including pytorch_model-00003-of-00003\.bin\. The download may be incomplete/],
    ['listed tensor absent', { map: m => ({ ...m, 'model.extra.weight': binShard(1) }) }, /lists model\.extra\.weight in pytorch_model-00002-of-00003.*does not contain/],
    ['wrong shard', { map: m => ({ ...m, 'lm_head.weight': binShard(0) }) }, /maps lm_head\.weight to pytorch_model-00001-of-00003.*stored in pytorch_model-00003-of-00003/],
    ['duplicate tensor', { shards: [SHARDS[0], { ...SHARDS[1], 'model.layers.0.mlp.weight': [9, 9] }, SHARDS[2]], map: m => ({ ...m, 'model.layers.0.mlp.weight': binShard(0) }) }, /stored in both/],
    ['parent path', { map: m => ({ ...m, 'lm_head.weight': '../outside.bin' }) }, /Invalid shard file name in the PyTorch index/],
    ['safetensors shard', { map: m => ({ ...m, 'lm_head.weight': 'model-00003-of-00003.safetensors' }) }, /PyTorch index refers to a file that is not \.bin/],
    ['unreadable shard', async dir => { await fs.writeFile(path.join(dir, binShard(2)), 'not a checkpoint'); }, /Cannot read shard pytorch_model-00003-of-00003\.bin: /],
    ['no weight map', { map: () => [] }, /no weight_map/]
  ];
  for (const [label, setup, expected] of cases) {
    const { root, dir } = await folder(t, label.replace(/ /g, '-')), index = await checkpoint(dir, typeof setup === 'function' ? {} : setup), python = await fakePython(root);
    if (typeof setup === 'function') await setup(dir);
    await assert.rejects(backend.inspect(index, python.options), expected, label);
    const alone = await backend.inspect(path.join(dir, binShard(0)), python.options);
    assert.equal(alone.format, 'pytorch', label); assert.equal(alone.source, undefined, label); assert.equal(alone.tensors.length, 2, label);
    assert.match(alone.warnings.join(' '), /Showing this shard only/, label);
    assert.deepEqual(await backend.sample(alone, 'model.embed_tokens.weight', python.options), [1, 2, 3, 4], label);
  }
});

test('changes to the PyTorch index or a shard are detected before the helper runs', fake, async t => {
  const { root, dir } = await folder(t), index = await checkpoint(dir), python = await fakePython(root);
  const model = await backend.inspect(path.join(dir, binShard(0)), python.options);
  await fs.appendFile(path.join(dir, binShard(2)), ' ');
  await assert.rejects(backend.sample(model, 'lm_head.weight', python.options), /Shard pytorch_model-00003-of-00003\.bin changed on disk/);
  assert.deepEqual(await backend.sample(model, 'model.layers.1.mlp.weight', python.options), [5, 7]);
  await fs.appendFile(index, ' ');
  await assert.rejects(backend.sample(model, 'model.norm.weight', python.options), /PyTorch index changed on disk/);
  assert.deepEqual((await python.requests()).filter(r => r.action === 'sample').map(r => r.name), ['model.layers.1.mlp.weight']);
});

test('standalone .bin files are not read as PyTorch, and GGUF files named .bin still open', async t => {
  const { dir } = await folder(t), raw = path.join(dir, 'weights.bin');
  await fs.writeFile(raw, F.f32([1, 2]));
  await assert.rejects(backend.inspect(raw, { trusted: true, pythonPath: path.join(dir, 'missing-python') }), /opens \.bin files as PyTorch only when they are Hugging Face shards/);
  const gguf = path.join(dir, 'ggml-model.bin');
  await F.gguf(gguf, [{ name: 'blk.0.attn_q.weight', shape: [2], values: [1, 2] }]);
  assert.equal((await backend.inspect(gguf)).format, 'gguf');
  assert.equal(sharded.shardFormat('pytorch_model-00001-of-00002.bin'), 'pytorch'); assert.equal(sharded.shardFormat('model-00001-of-00002.SAFETENSORS'), 'safetensors');
  assert.equal(sharded.shardFormat('pytorch_model.bin'), null); assert.ok(sharded.isIndex('/x/diffusion_pytorch_model.bin.index.json'));
});

test('the manifest opens PyTorch shard indexes and shards without claiming other .bin files', () => {
  const selectors = manifest.contributes.customEditors[0].selector.map(s => s.filenamePattern);
  assert.ok(selectors.includes('*.bin.index.json')); assert.ok(selectors.includes('*-[0-9]*-of-[0-9]*.bin')); assert.ok(!selectors.includes('*.bin'));
  const when = manifest.contributes.menus['explorer/context'][0].when;
  const patterns = [...when.matchAll(/resourceFilename =~ \/(.+?)\/(?: |$)/g)].map(m => new RegExp(m[1]));
  const menu = name => patterns.some(p => p.test(name));
  for (const name of ['pytorch_model.bin.index.json', 'model.safetensors.index.json', 'pytorch_model-00001-of-00002.bin']) assert.ok(menu(name), name);
  for (const name of ['pytorch_model.bin', 'weights.bin', 'config.json']) assert.ok(!menu(name), name);
});
