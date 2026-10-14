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
