'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const Module = require('node:module');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const F = require('./fixtures');

test('extension activation, webview handshake, stats, comparison, export and checkpoint protection', async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-extension-'));
  const base = path.join(dir, 'base.safetensors'), target = path.join(dir, 'target.safetensors');
  await F.safetensors(base, [{ name: 'layer.weight', values: [1, 2] }]);
  await F.safetensors(target, [{ name: 'layer.weight', values: [1, 3] }]);
  const commands = new Map(), posts = [], uri = file => ({ scheme: 'file', fsPath: file, toString: () => `file:${file}` });
  let provider, receive, dispose, saveFile = path.join(dir, 'report.json');
  const vscode = {
    Uri: { file: uri, joinPath: (u, ...p) => uri(path.join(u.fsPath, ...p)) },
    workspace: { isTrusted: true, getConfiguration: () => ({ get: (_key, fallback) => fallback }), fs: { writeFile: (u, b) => fs.writeFile(u.fsPath, b) } },
    commands: { registerCommand: (name, fn) => { commands.set(name, fn); return { dispose() {} }; }, executeCommand: async () => {} },
    window: {
      registerCustomEditorProvider: (_view, p) => { provider = p; return { dispose() {} }; },
      showOpenDialog: async () => [uri(target)], showSaveDialog: async () => uri(saveFile), showInformationMessage() {}
    }
  };
  const originalLoad = Module._load;
  Module._load = function (name, ...args) { return name === 'vscode' ? vscode : originalLoad.call(this, name, ...args); };
  let extension;
  try { extension = require('../src/extension'); } finally { Module._load = originalLoad; }
  const context = { extensionUri: uri(path.resolve(__dirname, '..')), subscriptions: [] };
  extension.activate(context);
  const panel = { webview: { cspSource: 'vscode-webview:', asWebviewUri: u => u.fsPath,
    postMessage: m => posts.push(m), onDidReceiveMessage: fn => { receive = fn; return { dispose() {} }; } },
    onDidDispose: fn => { dispose = fn; } };
  try {
    const doc = provider.openCustomDocument(uri(base)); provider.resolveCustomEditor(doc, panel);
    assert.match(panel.webview.html, /Content-Security-Policy/); assert.match(panel.webview.html, /nonce-/);
    assert.ok(panel.webview.html.indexOf('model-view.js') < panel.webview.html.indexOf('app.js'));
    assert.ok(commands.has('weightscope.compare'));
    await receive({ type: 'ready' }); assert.equal(posts.find(m => m.type === 'model').model.summary.parameters, 2);
    await receive({ type: 'stats', name: 'layer.weight' }); assert.equal(posts.find(m => m.type === 'statistics').data.mean, 1.5);
    await receive({ type: 'compare' }); assert.equal(posts.find(m => m.type === 'comparison').rows[0].status, 'unverified');
    await receive({ type: 'analyzeDiff' }); assert.equal(posts.find(m => m.type === 'diffRow').row.metrics.maxAbs, 1);
    await receive({ type: 'export' }); const report = JSON.parse(await fs.readFile(saveFile, 'utf8')); assert.equal(report.comparison.rows[0].status, 'changed');
    saveFile = base; await receive({ type: 'export' }); assert.match(posts.at(-1).message, /cannot overwrite/);
    saveFile = path.join(dir, 'alias.json'); await fs.symlink(base, saveFile); await receive({ type: 'export' }); assert.match(posts.at(-1).message, /symlink/);
    await receive({ type: '__proto__' }); await receive({ type: 'stats', name: 'unknown' }); assert.match(posts.findLast(m => m.type === 'error').message, /Choose a tensor/);
    await receive({ type: 'refresh' }); assert.equal(posts.findLast(m => m.type === 'model').model.name, 'base.safetensors');
    assert.throws(() => provider.openCustomDocument({ scheme: 'vscode-vfs' }), /extension host/);
  } finally { dispose?.(); await fs.rm(dir, { recursive: true, force: true }); }
});

test('Ollama commands, folder fallback, restored editors, comparison and report protection', async t => {
  const { store } = require('./ollama-fixtures'), ollama = require('../src/ollama');
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-ollama-extension-')); t.after(() => fs.rm(dir, { recursive: true, force: true }));
  const s = await store(path.join(dir, 'models'));
  const a = await s.blob('model', [{ name: 'blk.0.weight', shape: [2], values: [1, 2] }]), b = await s.blob('model', [{ name: 'blk.0.weight', shape: [2], values: [1, 4] }]);
  const template = await s.blob('template', 'prompt'), config = await s.blob('config', {});
  const base = await s.manifest('tiny', 'base', [a, template], { config }), target = await s.manifest('tiny', 'target', [b]);
  await s.manifest('tiny', 'missing', [{ ...a, digest: `sha256:${'0'.repeat(64)}` }]);
  const commands = new Map(), executed = [], errors = [], pickers = [], posts = [], picks = [], uri = file => ({ scheme: 'file', fsPath: file, toString: () => `file:${file}` });
  let provider, receive, dispose, configured = path.join(dir, 'not-found'), saveFile = path.join(dir, 'report.json');
  const vscode = {
    Uri: { file: uri, joinPath: (u, ...p) => uri(path.join(u.fsPath, ...p)) },
    workspace: { isTrusted: false, getConfiguration: () => ({ get: (key, fallback) => key === 'ollamaModelsPath' ? configured : fallback }), fs: { writeFile: (u, data) => fs.writeFile(u.fsPath, data) } },
    commands: { registerCommand: (name, fn) => { commands.set(name, fn); return { dispose() {} }; }, executeCommand: async (...args) => { executed.push(args); } },
    window: {
      registerCustomEditorProvider: (_view, p) => { provider = p; return { dispose() {} }; },
      showQuickPick: async (items, options) => { const all = await items; pickers.push({ all, options }); const next = picks.shift(); if (!next) return; const item = all.find(i => next === 'browse' ? i.browse : i.label === next); assert.ok(item, `Missing picker choice: ${next}`); return item; },
      showOpenDialog: async options => { assert.equal(options.canSelectFolders, true); return [uri(s.root)]; },
      showSaveDialog: async options => { assert.ok(!options.defaultUri.fsPath.startsWith(s.root)); return uri(saveFile); },
      showInformationMessage() {}, showErrorMessage: async message => { errors.push(message); }
    }
  };
  const originalLoad = Module._load; delete require.cache[require.resolve('../src/extension')];
  Module._load = function (name, ...args) { return name === 'vscode' ? vscode : originalLoad.call(this, name, ...args); };
  let extension; try { extension = require('../src/extension'); } finally { Module._load = originalLoad; }
  extension.activate({ extensionUri: uri(path.resolve(__dirname, '..')), subscriptions: [] });
  picks.push('browse', 'tiny:missing', 'tiny:base'); await commands.get('weightscope.openOllama')();
  assert.match(pickers[0].all[0].detail, /No local catalog/); assert.match(errors[0], /missing/);
  assert.equal(executed.at(-1)[0], 'vscode.openWith'); assert.equal(executed.at(-1)[1].fsPath, base); assert.equal(executed.at(-1)[2], 'weightscope.inspector');
  assert.match(pickers.at(-1).all.find(i => i.label === 'tiny:base').description, /GiB.*GGUF/);
  const count = executed.length; await commands.get('weightscope.openOllama')(); assert.equal(executed.length, count);
  picks.push('tiny:base', 'tiny:target'); await commands.get('weightscope.compareOllama')();
  const panel = { webview: { cspSource: 'vscode-webview:', asWebviewUri: u => u.fsPath, postMessage: m => posts.push(m), onDidReceiveMessage: fn => { receive = fn; return { dispose() {} }; } }, onDidDispose: fn => { dispose = fn; } };
  try {
    provider.resolveCustomEditor(provider.openCustomDocument(uri(base)), panel); await receive({ type: 'ready' });
    assert.equal(posts.find(m => m.type === 'model').model.name, 'tiny:base'); assert.equal(posts.find(m => m.type === 'comparison').target.name, 'tiny:target');
    await receive({ type: 'analyzeDiff' }); assert.equal(posts.findLast(m => m.type === 'diffRow').row.metrics.maxAbs, 2);
    await receive({ type: 'export' }); assert.equal(JSON.parse(await fs.readFile(saveFile, 'utf8')).model.source.template.text, 'prompt');
    for (const protectedFile of [base, ollama.blobPath(s.root, template.digest), ollama.blobPath(s.root, config.digest), ollama.blobPath(s.root, b.digest)]) {
      saveFile = protectedFile; await receive({ type: 'export' }); assert.match(posts.at(-1).message, /cannot overwrite/);
    }
    saveFile = path.join(dir, 'report-alias.json'); await fs.symlink(ollama.blobPath(s.root, template.digest), saveFile);
    await receive({ type: 'export' }); assert.match(posts.at(-1).message, /symlink/);
    picks.push('tiny:target'); await receive({ type: 'compareOllama' }); assert.equal(posts.findLast(m => m.type === 'comparison').target.source.name, 'tiny:target');
    picks.push('tiny:target'); await receive({ type: 'openOllama' }); assert.equal(executed.at(-1)[1].fsPath, target);
    // Restoring the manifest URI without the picker retains the selected model/tag.
    await receive({ type: 'refresh' }); assert.equal(posts.findLast(m => m.type === 'model').model.source.name, 'tiny:base');
  } finally { dispose?.(); }
});
