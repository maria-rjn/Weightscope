'use strict';
const vscode = require('vscode');
const path = require('node:path');
const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const os = require('node:os');
const backend = require('./backend');
const ollama = require('./ollama');
const checkpoint = require('./checkpoint');
const metadata = require('./metadata');
const tensorEdits = require('./tensor-edits');
const { metadataDiff } = require('./math');
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
  let ollamaFolder;
  const open = uri => vscode.commands.executeCommand('vscode.openWith', uri, VIEW);
  const setup = () => vscode.commands.executeCommand('markdown.showPreview', vscode.Uri.joinPath(context.extensionUri, 'docs', 'PYTORCH.md'));
  async function chooseOllama(title) {
    while (true) {
      const configured = ollamaFolder || vscode.workspace.getConfiguration('weightscope').get('ollamaModelsPath', '');
      const { root, searched } = await ollama.findRoot(configured);
      const items = (async () => {
        const models = root ? await ollama.listModels(root) : [];
        return [...models.map(entry => ({ label: entry.name, description: entry.problem ? 'Unavailable' : `${(entry.size / 1024 ** 3).toFixed(2)} GiB · ${entry.format.toUpperCase()}`,
          detail: entry.problem || `${entry.layers.filter(l => ['model', 'tensor'].includes(l.kind)).length} weight blob(s) · ${root}`, entry })),
        { label: 'Choose Ollama models folder…', detail: root || `No local catalog found. Checked: ${searched.join(', ')}`, browse: true }];
      })();
      const choice = await vscode.window.showQuickPick(items, { title, placeHolder: 'Search downloaded models, or choose a folder containing manifests and blobs', matchOnDescription: true, matchOnDetail: true, ignoreFocusOut: true });
      if (!choice) return;
      if (choice.browse) {
        const folder = (await vscode.window.showOpenDialog({ title: 'Choose Ollama models folder (contains manifests and blobs)', canSelectFiles: false, canSelectFolders: true, canSelectMany: false }))?.[0];
        if (!folder) return;
        if (folder.scheme !== 'file') throw new Error('Choose an Ollama store on the extension host.');
        const found = await ollama.findRoot(folder.fsPath);
        if (!found.root) { await vscode.window.showErrorMessage('This folder has no manifests directory. Choose the Ollama models folder.'); continue; }
        ollamaFolder = found.root; continue;
      }
      if (choice.entry.problem) { await vscode.window.showErrorMessage(`${choice.entry.name}: ${choice.entry.problem}`); continue; }
      return vscode.Uri.file(choice.entry.manifestPath);
    }
  }
  async function compareFiles(base, target) {
    const panel = panels.get(base.toString());
    if (panel) { await open(base); await panel.compare(target); }
    else { pending.set(base.toString(), target); await open(base); }
  }
  const provider = {
    openCustomDocument(uri) {
      if (uri.scheme !== 'file') throw new Error('WeightScope needs a file on the extension host. Remote SSH works when installed on that host; virtual file systems are not supported.');
      return { uri, dispose() {} };
    },
    resolveCustomEditor(document, panel) {
      const key = document.uri.toString(), uri = document.uri;
      panel.webview.options = { enableScripts: true, localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')] };
      let model, target, rows = [], stats = new Map(), busy = false, operation, disposed = false, ready = false;
      let metadataEdits = new Map(), pendingTensors = new Map(), savedUri;
      const savedFiles = new Set();
      const editPayload = () => ({ type: 'edits', metadata: [...metadataEdits.values()], tensors: [...pendingTensors.values()].map(e => ({ name: e.name, operation: e.operation, factor: e.factor,
        ...(e.source ? { sourcePath: e.source.model.path, sourceName: e.source.name } : {}) })) });
      const post = data => { if (!disposed) panel.webview.postMessage(data); };
      const payload = () => {
        const data = backend.publicModel(model);
        return { type: 'model', model: { ...data, editing: checkpoint.capabilities(model), tensors: data.tensors.map(t => ({ ...t, editing: checkpoint.tensorCapabilities(model, model.byName.get(t.name)) })) }, settings: settings() };
      };
      const comparisonPayload = () => ({ type: 'comparison', target: backend.publicModel(target), rows, metadata: metadataDiff(model, target) });
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
          metadataEdits.clear(); pendingTensors.clear(); post(payload()); post(editPayload());
        });
      }
      async function compare(targetUri) {
        if (!model) { if (targetUri) pending.set(key, targetUri); return; }
        if (!targetUri) targetUri = await choose('Select target checkpoint (current model is the baseline)');
        if (!targetUri) return;
        await run('Reading target checkpoint…', async options => {
          await backend.assertCurrent(model);
          const next = await backend.inspect(targetUri.fsPath, options);
          target = next; rows = backend.structureDiff(model, target);
          post(comparisonPayload());
        });
      }
      const actions = {
        ready: async () => {
          if (ready) { if (model) { post(payload()); post(editPayload()); if (savedUri) post({ type: 'checkpointSaved', path: savedUri.fsPath }); for (const [name, data] of stats) post({ type: 'statistics', name, data }); if (target) post(comparisonPayload()); } return; }
          ready = true; await refresh();
          const next = pending.get(key); pending.delete(key); if (next) await compare(next);
        },
        refresh: async () => { if (metadataEdits.size || pendingTensors.size) throw new Error('Discard pending edits before refreshing the model.'); await refresh(); },
        stageMetadata: message => {
          if (busy) throw new Error('Wait for the current operation before editing.');
          if (!model || !checkpoint.capabilities(model).metadata) throw new Error('This checkpoint does not support metadata editing.');
          const change = { key: message.key, value: message.value, ...(message.valueType ? { type: message.valueType } : {}) };
          const next = new Map(metadataEdits); next.set(change.key, change);
          metadata.validateForModel(model, [...next.values()]); metadataEdits = next; post(editPayload());
        },
        stageTensor: message => run('Preparing tensor edit…', async options => {
          if (!model) throw new Error('Open a checkpoint first.');
          const edit = { name: message.name, operation: message.operation, factor: message.factor };
          const tensor = model.byName.get(edit.name);
          if (!tensor || !checkpoint.tensorCapabilities(model, tensor).operations.includes(edit.operation)) throw new Error('This tensor does not support the requested operation.');
          if (edit.operation === 'replace') {
            const chosen = await choose('Choose the replacement checkpoint'); if (!chosen) return;
            const donor = await backend.inspect(chosen.fsPath, options);
            const compatible = donor.tensors.filter(t => {
              try { tensorEdits.validate(model, [{ ...edit, source: { model: donor, name: t.name } }]); return true; } catch { return false; }
            }).sort((a, b) => Number(b.name === tensor.name) - Number(a.name === tensor.name));
            if (!compatible.length) throw new Error('The checkpoint has no tensors with a matching shape, dtype, and storage layout.');
            const selection = await vscode.window.showQuickPick(compatible.map(t => ({ label: t.name, description: `${t.dtype} · ${t.shape.join(' × ') || 'scalar'}` })), { title: 'Choose the replacement tensor', matchOnDescription: true });
            if (!selection) return;
            edit.source = { model: donor, name: selection.label };
          }
          const next = new Map(pendingTensors); next.set(edit.name, edit); tensorEdits.validate(model, [...next.values()]);
          pendingTensors = next; post(editPayload());
        }),
        discardEdits: () => { if (busy) throw new Error('Wait for the current operation before discarding edits.'); metadataEdits.clear(); pendingTensors.clear(); post(editPayload()); },
        compareSaved: () => { if (!savedUri) throw new Error('Save a checkpoint copy first.'); return compare(savedUri); },
        openSaved: () => { if (!savedUri) throw new Error('Save a checkpoint copy first.'); return open(savedUri); },
        compare: () => compare(),
        openOllama: async () => { const chosen = await chooseOllama('Open a downloaded Ollama model'); if (chosen) await open(chosen); },
        compareOllama: async () => { const chosen = await chooseOllama('Select an Ollama model to compare with the current model'); if (chosen) await compare(chosen); },
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
        saveCheckpoint: () => run('Saving checkpoint copy…', async options => {
          if (!model) throw new Error('Open a checkpoint first.');
          const support = checkpoint.capabilities(model);
          if (!support.save) throw new Error(support.reason);
          const name = model.source?.name.replace(/[^a-zA-Z0-9._-]/g, '_') || path.basename(model.path, path.extname(model.path));
          const directory = model.source ? os.homedir() : path.dirname(model.path);
          const output = await vscode.window.showSaveDialog({ title: 'Save checkpoint as a new file', defaultUri: vscode.Uri.file(path.join(directory, `${name}.copy${support.extension}`)), filters: { Checkpoint: [support.extension.slice(1)] } });
          if (!output) return;
          if (output.scheme !== 'file') throw new Error('Save the checkpoint on the extension host.');
          const result = await checkpoint.saveCheckpoint(model, output.fsPath, { ...options, metadata: [...metadataEdits.values()], tensors: [...pendingTensors.values()] });
          savedUri = vscode.Uri.file(result.path); savedFiles.add(result.path);
          post({ type: 'checkpointSaved', path: result.path });
          vscode.window.showInformationMessage(`Checkpoint saved: ${result.path}`);
        }),
        setup,
        export: async () => {
          if (!model) return;
          const defaultPath = model.source?.kind === 'ollama' ? path.join(os.homedir(), `${model.source.name.replace(/[^a-zA-Z0-9._-]/g, '_')}.weightscope.json`) : path.join(path.dirname(uri.fsPath), `${path.basename(uri.fsPath)}.weightscope.json`);
          const output = await vscode.window.showSaveDialog({ title: 'Export WeightScope report', defaultUri: vscode.Uri.file(defaultPath), filters: { JSON: ['json'] } });
          if (!output) return;
          const donors = [...pendingTensors.values()].flatMap(e => e.source ? [e.source.model.path, ...(e.source.model.protectedPaths || [])] : []);
          const protectedPaths = [model.path, target?.path, ...savedFiles, ...donors, ...(model.protectedPaths || []), ...(target?.protectedPaths || [])].filter(Boolean).map(p => path.resolve(p));
          if (output.scheme === 'file' && protectedPaths.includes(path.resolve(output.fsPath))) throw new Error('A report cannot overwrite an inspected checkpoint.');
          if (output.scheme === 'file') {
            const destination = await fs.realpath(output.fsPath).catch(e => { if (e.code === 'ENOENT') return path.resolve(output.fsPath); throw e; });
            for (const checkpoint of protectedPaths) if (destination === await fs.realpath(checkpoint).catch(e => { if (e.code === 'ENOENT') return checkpoint; throw e; })) throw new Error('A report cannot overwrite an inspected checkpoint through a symlink.');
            const existing = await fs.stat(output.fsPath).catch(e => { if (e.code === 'ENOENT') return null; throw e; });
            if (existing) for (const checkpoint of protectedPaths) {
              const source = await fs.stat(checkpoint).catch(e => { if (e.code === 'ENOENT') return null; throw e; });
              if (source && source.dev === existing.dev && source.ino === existing.ino) throw new Error('A report cannot overwrite a checkpoint through a hard link.');
            }
          }
          const report = { version: 1, exportedAt: new Date().toISOString(), model: backend.publicModel(model), statistics: Object.fromEntries(stats),
            comparison: target ? { target: backend.publicModel(target), rows, metadata: metadataDiff(model, target) } : null,
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
    vscode.commands.registerCommand('weightscope.openOllama', async () => {
      try { const chosen = await chooseOllama('Open a downloaded Ollama model'); if (chosen) await open(chosen); }
      catch (e) { await vscode.window.showErrorMessage(`WeightScope: ${e.message}`); }
    }),
    vscode.commands.registerCommand('weightscope.compareOllama', async () => {
      try {
        const base = await chooseOllama('Select baseline Ollama model'); if (!base) return;
        const target = await chooseOllama('Select target Ollama model'); if (target) await compareFiles(base, target);
      } catch (e) { await vscode.window.showErrorMessage(`WeightScope: ${e.message}`); }
    }),
    vscode.commands.registerCommand('weightscope.compare', async () => {
      const base = await choose('Select baseline checkpoint'); if (!base) return;
      const target = await choose('Select target checkpoint'); if (!target) return;
      await compareFiles(base, target);
    }),
    vscode.commands.registerCommand('weightscope.setup', setup)
  );
}
module.exports = { activate, deactivate() {} };
