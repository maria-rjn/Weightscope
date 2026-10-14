'use strict';
const vscode = require('vscode');
const path = require('node:path');
const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const backend = require('./backend');
const VIEW = 'weightscope.inspector';
const filters = { 'Model checkpoints': ['safetensors', 'pt', 'pth', 'onnx', 'gguf'] };

function settings() {
  const c = vscode.workspace.getConfiguration('weightscope');
  return { trusted: vscode.workspace.isTrusted, pythonPath: c.get('pythonPath', ''), sampleSize: c.get('sampleSize', 32768),
    maxDiffTensors: c.get('maxDiffTensors', 500), absoluteTolerance: c.get('absoluteTolerance', 0) };
}
async function choose(title) { return (await vscode.window.showOpenDialog({ title, filters, canSelectMany: false, canSelectFolders: false }))?.[0]; }
function html(webview, root) {
  const media = name => webview.asWebviewUri(vscode.Uri.joinPath(root, 'media', name));
  const nonce = crypto.randomBytes(24).toString('base64');
  return `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}'; img-src ${webview.cspSource} data:;">
    <link rel="stylesheet" href="${media('styles.css')}"><title>WeightScope</title></head><body>
    <div id="app"></div><script nonce="${nonce}" src="${media('model-view.js')}"></script><script nonce="${nonce}" src="${media('app.js')}"></script></body></html>`;
}
function activate(context) {
  const panels = new Map(), pending = new Map();
  const open = uri => vscode.commands.executeCommand('vscode.openWith', uri, VIEW);
  const setup = () => vscode.commands.executeCommand('markdown.showPreview', vscode.Uri.joinPath(context.extensionUri, 'docs', 'PYTORCH.md'));
  const provider = {
    openCustomDocument(uri) {
      if (uri.scheme !== 'file') throw new Error('WeightScope needs a file on the extension host. Remote SSH works when installed on that host; virtual file systems are not supported.');
      return { uri, dispose() {} };
    },
    resolveCustomEditor(document, panel) {
      const key = document.uri.toString(), uri = document.uri;
      panel.webview.options = { enableScripts: true, localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')] };
      let model, target, rows = [], stats = new Map(), busy = false, operation, disposed = false, ready = false;
      const post = data => { if (!disposed) panel.webview.postMessage(data); };
      const payload = () => ({ type: 'model', model: backend.publicModel(model), settings: settings() });
      async function run(label, fn) {
        if (busy) { post({ type: 'error', message: 'An analysis is already running. Cancel it or wait for it to finish.' }); return; }
        busy = true; operation = new AbortController();
        post({ type: 'busy', label });
        try { await fn({ ...settings(), signal: operation.signal }); }
        catch (e) { if (!disposed) post({ type: 'error', message: e.message }); }
        finally { busy = false; operation = undefined; post({ type: 'idle' }); }
      }
      async function refresh() {
        await run('Reading model metadata…', async options => {
          model = await backend.inspect(uri.fsPath, options); target = undefined; rows = []; stats.clear();
          post(payload());
        });
      }
      async function compare(targetUri) {
        if (!model) { if (targetUri) pending.set(key, targetUri); return; }
        if (!targetUri) targetUri = await choose('Select target checkpoint (current model is the baseline)');
        if (!targetUri) return;
        await run('Reading target checkpoint…', async options => {
          const next = await backend.inspect(targetUri.fsPath, options);
          target = next; rows = backend.structureDiff(model, target);
          post({ type: 'comparison', target: backend.publicModel(target), rows });
        });
      }
      const actions = {
        ready: async () => {
          if (ready) { if (model) { post(payload()); for (const [name, data] of stats) post({ type: 'statistics', name, data }); if (target) post({ type: 'comparison', target: backend.publicModel(target), rows }); } return; }
          ready = true; await refresh();
          const next = pending.get(key); pending.delete(key); if (next) await compare(next);
        },
        refresh,
        compare: () => compare(),
        stats: message => run('Analyzing tensor values…', async options => {
          if (!model?.byName.has(message.name)) throw new Error('Choose a tensor first.');
          const data = await backend.tensorStatistics(model, message.name, options); stats.set(message.name, data);
          post({ type: 'statistics', name: message.name, data });
        }),
        analyzeDiff: message => run('Comparing tensor values…', async options => {
          if (!model || !target) throw new Error('Select a target checkpoint first.');
          const candidates = message.name ? rows.filter(r => r.name === message.name && ['unverified', 'unavailable', 'changed', 'equal', 'sample-match'].includes(r.status))
            : rows.filter(r => ['unverified', 'unavailable'].includes(r.status)).slice(0, options.maxDiffTensors);
          let completed = 0;
          for (const row of candidates) {
            if (options.signal.aborted) throw new Error('Operation cancelled.');
            try { row.metrics = await backend.compareTensor(model, target, row.name, options); row.status = row.metrics.status; delete row.error; }
            catch (e) { if (options.signal.aborted) throw e; row.status = 'unavailable'; row.error = e.message; }
            post({ type: 'diffRow', row, completed: ++completed, total: candidates.length });
          }
          post({ type: 'diffDone', remaining: rows.filter(r => ['unverified', 'unavailable'].includes(r.status)).length });
        }),
        cancel: () => operation?.abort(),
        setup,
        export: async () => {
          if (!model) return;
          const output = await vscode.window.showSaveDialog({ title: 'Export WeightScope report', defaultUri: vscode.Uri.file(path.join(path.dirname(uri.fsPath), `${path.basename(uri.fsPath)}.weightscope.json`)), filters: { JSON: ['json'] } });
          if (!output) return;
          const protectedPaths = [model.path, target?.path].filter(Boolean).map(p => path.resolve(p));
          if (output.scheme === 'file' && protectedPaths.includes(path.resolve(output.fsPath))) throw new Error('A report cannot overwrite an inspected checkpoint.');
          if (output.scheme === 'file') {
            const destination = await fs.realpath(output.fsPath).catch(e => { if (e.code === 'ENOENT') return path.resolve(output.fsPath); throw e; });
            for (const checkpoint of protectedPaths) if (destination === await fs.realpath(checkpoint)) throw new Error('A report cannot overwrite an inspected checkpoint through a symlink.');
          }
          const report = { version: 1, exportedAt: new Date().toISOString(), model: backend.publicModel(model), statistics: Object.fromEntries(stats),
            comparison: target ? { target: backend.publicModel(target), rows } : null,
            note: 'Counts refer to stored tensor elements. Sample matches do not prove whole-tensor equality. Numerical metrics use the stated tolerance and sampling coverage.' };
          await vscode.workspace.fs.writeFile(output, Buffer.from(JSON.stringify(report, null, 2)));
          vscode.window.showInformationMessage(`WeightScope report saved: ${output.fsPath}`);
        }
      };
      panels.set(key, { compare });
      const listener = panel.webview.onDidReceiveMessage(async message => {
        if (!message || typeof message.type !== 'string' || !Object.hasOwn(actions, message.type)) return;
        if (message.name !== undefined && (typeof message.name !== 'string' || message.name.length > 1048576)) return;
        try { await actions[message.type](message); } catch (e) { post({ type: 'error', message: e.message }); }
      });
      panel.onDidDispose(() => { disposed = true; operation?.abort(); listener.dispose(); panels.delete(key); });
      panel.webview.html = html(panel.webview, context.extensionUri);
    }
  };
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(VIEW, provider, { webviewOptions: { retainContextWhenHidden: true }, supportsMultipleEditorsPerDocument: false }),
    vscode.commands.registerCommand('weightscope.open', async uri => { const chosen = uri || await choose('Inspect a model checkpoint'); if (chosen) await open(chosen); }),
    vscode.commands.registerCommand('weightscope.compare', async () => {
      const base = await choose('Select baseline checkpoint'); if (!base) return;
      const target = await choose('Select target checkpoint'); if (!target) return;
      const panel = panels.get(base.toString());
      if (panel) { await open(base); await panel.compare(target); }
      else { pending.set(base.toString(), target); await open(base); }
    }),
    vscode.commands.registerCommand('weightscope.setup', setup)
  );
}
module.exports = { activate, deactivate() {} };
