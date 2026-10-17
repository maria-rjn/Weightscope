'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const Module = require('node:module');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const backend = require('../src/backend');
const sharded = require('../src/sharded');
const { capabilities } = require('../src/checkpoint');
const F = require('./fixtures');

const SHARDS = [
  [{ name: 'model.embed_tokens.weight', shape: [2, 2], values: [1, 2, 3, 4] }, { name: 'model.layers.0.self_attn.q_proj.weight', values: [1, 1] }],
  [{ name: 'model.layers.1.self_attn.q_proj.weight', values: [5, 7] }],
  [{ name: 'model.norm.weight', values: [.5, .5] }, { name: 'lm_head.weight', values: [-2, 2] }]
];
const shardFile = (i, n = SHARDS.length) => `model-${String(i + 1).padStart(5, '0')}-of-${String(n).padStart(5, '0')}.safetensors`;
async function folder(t, name = 'tiny-llm') {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-sharded-')); t.after(() => fs.rm(root, { recursive: true, force: true }));
  const dir = path.join(root, name); await fs.mkdir(dir, { recursive: true }); return { root, dir };
}
// Writes Hugging Face style shards; `map` adjusts the index weight_map, `index: false` omits it.
async function checkpoint(dir, { shards = SHARDS, map = m => m, index = true } = {}) {
  const weightMap = {};
  for (const [i, tensors] of shards.entries()) {
    await F.safetensors(path.join(dir, shardFile(i, shards.length)), tensors, { format: 'pt' });
    for (const t of tensors) weightMap[t.name] = shardFile(i, shards.length);
  }
  const file = path.join(dir, 'model.safetensors.index.json');
  if (index) await fs.writeFile(file, JSON.stringify({ metadata: { total_size: 48 }, weight_map: map(weightMap) }));
  return file;
}

test('a sharded index opens as one checkpoint and samples each tensor from its own shard', async t => {
  const { dir } = await folder(t), index = await checkpoint(dir);
  const model = await backend.inspect(index);
  assert.equal(model.format, 'safetensors'); assert.equal(model.tensors.length, 5); assert.equal(model.summary.parameters, 12);
  assert.equal(model.summary.tensorBytes, 48); assert.equal(model.metadata.index.total_size, 48);
  assert.equal(model.source.kind, 'sharded'); assert.equal(model.source.name, 'tiny-llm/model.safetensors'); assert.equal(model.source.shards.length, 3);
  assert.deepEqual(model.source.shards.map(s => s.tensors), [2, 1, 2]);
  assert.equal(model.fileBytes, model.source.shards.reduce((n, s) => n + s.bytes, 0));
  assert.equal(model.byName.get('lm_head.weight').shard, shardFile(2));
  assert.deepEqual(await backend.sample(model, 'model.layers.1.self_attn.q_proj.weight'), [5, 7]);
  assert.equal((await backend.tensorStatistics(model, 'lm_head.weight')).mean, 0);
  const view = backend.publicModel(model);
  assert.equal(view.name, 'tiny-llm/model.safetensors'); assert.equal(view.tensors[0].storage, undefined); assert.equal(view.tensors[0].shard, shardFile(0));
  assert.deepEqual(model.protectedPaths.sort(), [index, ...SHARDS.map((_, i) => path.join(dir, shardFile(i)))].sort());
  assert.equal(capabilities(model).save, false); assert.match(capabilities(model).reason, /Sharded/);
});

test('opening any shard opens the whole checkpoint, which compares with single-file checkpoints', async t => {
  const { dir } = await folder(t); await checkpoint(dir);
  const model = await backend.inspect(path.join(dir, shardFile(1)));
  assert.equal(model.path, path.join(dir, shardFile(1))); assert.equal(model.tensors.length, 5); assert.equal(model.source.kind, 'sharded');
  const single = path.join(dir, '..', 'merged.safetensors');
  await F.safetensors(single, SHARDS.flat().map(t => t.name === 'model.norm.weight' ? { ...t, values: [.5, 1] } : t));
  const target = await backend.inspect(single), rows = backend.structureDiff(model, target);
  assert.equal(rows.length, 5); assert.ok(rows.every(r => r.status === 'unverified'));
  assert.equal((await backend.compareTensor(model, target, 'model.norm.weight')).maxAbs, .5);
  assert.equal((await backend.compareTensor(target, model, 'model.embed_tokens.weight')).status, 'equal');
});

test('Hugging Face cache snapshots keep the repository name and follow symlinked shard blobs', async t => {
  const { root } = await folder(t, 'unused'), repo = path.join(root, 'models--acme--Tiny-LLM'), snapshot = path.join(repo, 'snapshots', 'abc123'), blobs = path.join(repo, 'blobs');
  await fs.mkdir(snapshot, { recursive: true }); await fs.mkdir(blobs);
  await checkpoint(blobs);
  // The cache stores content-addressed blobs; only the snapshot symlinks carry Hugging Face file names.
  for (const [i, name] of [...SHARDS.map((_, i) => shardFile(i)), 'model.safetensors.index.json'].entries()) {
    await fs.rename(path.join(blobs, name), path.join(blobs, `blob${i}`));
    await fs.symlink(path.join('..', '..', 'blobs', `blob${i}`), path.join(snapshot, name));
  }
  const model = await backend.inspect(path.join(snapshot, shardFile(0)));
  assert.equal(model.source.name, 'acme/Tiny-LLM/model.safetensors'); assert.equal(model.tensors.length, 5);
  assert.deepEqual(await backend.sample(model, 'model.norm.weight'), [.5, .5]);
  assert.equal(sharded.label('/data/llama', 'diffusion_pytorch_model'), 'llama/diffusion_pytorch_model.safetensors');
});

test('shards without an index are grouped by file name; an incomplete set opens one shard with a note', async t => {
  const { dir } = await folder(t); await checkpoint(dir, { index: false });
  const model = await backend.inspect(path.join(dir, shardFile(2)));
  assert.equal(model.tensors.length, 5); assert.equal(model.source.indexPath, null); assert.equal(model.metadata.index, null);
  assert.match(model.warnings.join(' '), /grouped by file name/);
  await fs.rm(path.join(dir, shardFile(1)));
  const partial = await backend.inspect(path.join(dir, shardFile(0)));
  assert.equal(partial.source, undefined); assert.equal(partial.tensors.length, 2);
  assert.match(partial.warnings.join(' '), /Showing this shard only: This file is shard 1 of 3.*model-00002-of-00003/);
  const lone = path.join(dir, 'solo-00001-of-00001.safetensors'); await F.safetensors(lone, [{ name: 'w', values: [1] }]);
  assert.deepEqual((await backend.inspect(lone)).warnings, []);
});

test('inconsistent indexes fail explicitly when opened, while their shards still open alone', async t => {
  const cases = [
    ['missing shard', async dir => { await fs.rm(path.join(dir, shardFile(2))); }, /1 of 3 shards listed in model\.safetensors\.index\.json are missing, including model-00003-of-00003\.safetensors\. The download may be incomplete/],
    ['listed tensor absent', { map: m => ({ ...m, 'model.extra.weight': shardFile(1) }) }, /lists model\.extra\.weight in model-00002-of-00003.*does not contain/],
    ['wrong shard', { map: m => ({ ...m, 'lm_head.weight': shardFile(0) }) }, /maps lm_head\.weight to model-00001-of-00003.*stored in model-00003-of-00003/],
    ['duplicate tensor', { shards: [SHARDS[0], [...SHARDS[1], SHARDS[0][1]], SHARDS[2]], map: m => ({ ...m, [SHARDS[0][1].name]: shardFile(0) }) }, /stored in both/],
    ['parent path', { map: m => ({ ...m, 'lm_head.weight': '../outside.safetensors' }) }, /Invalid shard file name/],
    ['nested path', { map: m => ({ ...m, 'lm_head.weight': 'sub/model.safetensors' }) }, /Invalid shard file name/],
    ['absolute path', { map: m => ({ ...m, 'lm_head.weight': '/tmp/model.safetensors' }) }, /Invalid shard file name/],
    ['pickle shard', { map: m => ({ ...m, 'lm_head.weight': 'pytorch_model-00003-of-00003.bin' }) }, /not \.safetensors/],
    ['no weight map', { map: () => [] }, /no weight_map/],
    ['empty weight map', { map: () => ({}) }, /lists no tensors/]
  ];
  for (const [label, setup, expected] of cases) {
    const { dir } = await folder(t, label.replace(/ /g, '-')), index = await checkpoint(dir, typeof setup === 'function' ? {} : setup);
    if (typeof setup === 'function') await setup(dir);
    await assert.rejects(backend.inspect(index), expected, label);
    const alone = await backend.inspect(path.join(dir, shardFile(0)));
    assert.equal(alone.source, undefined, label); assert.match(alone.warnings.join(' '), /Showing this shard only/, label);
  }
  const { dir } = await folder(t, 'bad-json'), index = await checkpoint(dir); await fs.writeFile(index, '{"weight_map":');
  await assert.rejects(backend.inspect(index), /not valid JSON/);
  assert.match((await backend.inspect(path.join(dir, shardFile(1)))).warnings.join(' '), /not valid JSON/);
});

test('tensors absent from the index are kept with a note; a differently named index is found', async t => {
  const { dir } = await folder(t), index = await checkpoint(dir, { map: m => { const { 'lm_head.weight': _, ...rest } = m; return rest; } });
  const model = await backend.inspect(index);
  assert.equal(model.tensors.length, 5); assert.match(model.warnings.join(' '), /1 stored tensor\(s\) are missing from the index/);
  await fs.rename(index, path.join(dir, 'weights.safetensors.index.json'));
  const renamed = await backend.inspect(path.join(dir, shardFile(0)));
  assert.equal(renamed.source.indexPath, path.join(dir, 'weights.safetensors.index.json')); assert.equal(renamed.source.name, 'tiny-llm/weights.safetensors');
});

test('changes to the index or a shard are detected before analysis', async t => {
  const { dir } = await folder(t), index = await checkpoint(dir);
  let model = await backend.inspect(path.join(dir, shardFile(0)));
  await fs.appendFile(path.join(dir, shardFile(2)), Buffer.from([0]));
  await assert.rejects(backend.sample(model, 'lm_head.weight'), /Shard model-00003-of-00003\.safetensors changed on disk/);
  assert.deepEqual(await backend.sample(model, 'model.layers.1.self_attn.q_proj.weight'), [5, 7]);
  await fs.rm(path.join(dir, shardFile(2))); await checkpoint(dir);
  model = await backend.inspect(path.join(dir, shardFile(0)));
  await fs.appendFile(index, ' ');
  await assert.rejects(backend.sample(model, 'model.norm.weight'), /safetensors index changed on disk/);
});

test('cancellation while reading shards is reported instead of falling back to one shard', async t => {
  const { dir } = await folder(t); await checkpoint(dir);
  let checks = 0; const signal = { get aborted() { return ++checks > 4; } };
  await assert.rejects(backend.inspect(path.join(dir, shardFile(0)), { signal }), /cancelled/);
});

test('the inspector opens sharded checkpoints, names reports after them and protects every shard', async t => {
  const { dir } = await folder(t), index = await checkpoint(dir), shard = path.join(dir, shardFile(1));
  const posts = [], dialogs = [], uri = file => ({ scheme: 'file', fsPath: file, toString: () => `file:${file}` });
  let provider, receive, dispose, saveFile = path.join(dir, 'report.json');
  const vscode = {
    Uri: { file: uri, joinPath: (u, ...p) => uri(path.join(u.fsPath, ...p)) },
    workspace: { isTrusted: true, getConfiguration: () => ({ get: (_key, fallback) => fallback }), fs: { writeFile: (u, b) => fs.writeFile(u.fsPath, b) } },
    commands: { registerCommand: () => ({ dispose() {} }), executeCommand: async () => {} },
    window: {
      registerCustomEditorProvider: (_view, p) => { provider = p; return { dispose() {} }; },
      showOpenDialog: async options => { dialogs.push(options); return [uri(index)]; }, showSaveDialog: async options => { dialogs.push(options); return uri(saveFile); }, showInformationMessage() {}
    }
  };
  const originalLoad = Module._load; delete require.cache[require.resolve('../src/extension')];
  Module._load = function (name, ...args) { return name === 'vscode' ? vscode : originalLoad.call(this, name, ...args); };
  let extension; try { extension = require('../src/extension'); } finally { Module._load = originalLoad; }
  extension.activate({ extensionUri: uri(path.resolve(__dirname, '..')), subscriptions: [] });
  const panel = { webview: { cspSource: 'vscode-webview:', asWebviewUri: u => u.fsPath, postMessage: m => posts.push(m), onDidReceiveMessage: fn => { receive = fn; return { dispose() {} }; } }, onDidDispose: fn => { dispose = fn; } };
  try {
    provider.resolveCustomEditor(provider.openCustomDocument(uri(shard)), panel); await receive({ type: 'ready' });
    const model = posts.find(m => m.type === 'model').model;
    assert.equal(model.name, 'tiny-llm/model.safetensors'); assert.equal(model.summary.tensorCount, 5); assert.equal(model.editing.save, false);
    await receive({ type: 'compare' });
    assert.deepEqual(dialogs[0].filters['Sharded checkpoint index'], ['json']); assert.deepEqual(dialogs[0].filters['PyTorch shards'], ['bin']);
    assert.ok(posts.find(m => m.type === 'comparison').rows.every(r => r.status === 'unverified'));
    await receive({ type: 'export' }); assert.equal(dialogs.at(-1).defaultUri.fsPath, path.join(dir, 'model.safetensors.weightscope.json'));
    assert.equal(JSON.parse(await fs.readFile(saveFile, 'utf8')).model.source.shards.length, 3);
    for (const protectedFile of [index, path.join(dir, shardFile(0)), path.join(dir, shardFile(2))]) {
      saveFile = protectedFile; await receive({ type: 'export' }); assert.match(posts.at(-1).message, /cannot overwrite/);
    }
    await receive({ type: 'saveCheckpoint' }); assert.match(posts.findLast(m => m.type === 'error').message, /Sharded safetensors/);
  } finally { dispose?.(); }
});
