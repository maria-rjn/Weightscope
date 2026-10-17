'use strict';
// Development preview: the production webview, connected to real generated checkpoints.
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const backend = require('../src/backend');
const checkpoint = require('../src/checkpoint');
const metadata = require('../src/metadata');
const tensorEdits = require('../src/tensor-edits');
const { metadataDiff } = require('../src/math');
const { makeDemo } = require('./make-demo');
const { store } = require('../test/ollama-fixtures');
async function main() {
  const demo = await makeDemo();
  const model = await backend.inspect(demo.base), target = await backend.inspect(demo.target);
  const catalog = await store(path.join(__dirname, '..', 'output', 'preview', 'models'));
  const template = await catalog.blob('template', '{{ .System }}\nUser: {{ .Prompt }}\nAssistant:');
  const params = await catalog.blob('params', { temperature: 0.7, num_ctx: 8192 });
  async function ollamaCheckpoint(file, tag) {
    const data = await fs.readFile(file), hash = crypto.createHash('sha256').update(data).digest('hex');
    await fs.writeFile(path.join(catalog.root, 'blobs', `sha256-${hash}`), data);
    return backend.inspect(await catalog.manifest('tiny-transformer', tag, [{ mediaType: 'application/vnd.ollama.image.tensor', digest: `sha256:${hash}`, size: data.length }, template, params]));
  }
  const ollamaModel = await ollamaCheckpoint(demo.base, 'base'), ollamaTarget = await ollamaCheckpoint(demo.target, 'finetuned');
  let rows = backend.structureDiff(model, target), selectedTarget = target, savedPath;
  const options = { sampleSize: 32768, maxDiffTensors: 500, absoluteTolerance: 0 };
  const edits = new Map(), tensorPlan = new Map();
  const publicModel = m => ({ ...backend.publicModel(m), editing: checkpoint.capabilities(m), tensors: backend.publicModel(m).tensors.map(t => ({ ...t, editing: checkpoint.tensorCapabilities(m, m.byName.get(t.name)) })) });
  const editPayload = () => ({ type: 'edits', metadata: [...edits.values()], tensors: [...tensorPlan.values()].map(e => ({ name: e.name, operation: e.operation, factor: e.factor, sourceName: e.source?.name, sourcePath: e.source?.model.path })) });
  const server = http.createServer(async (req, res) => {
    try {
      const route = new URL(req.url, 'http://127.0.0.1').pathname;
      if (req.method === 'GET' && route === '/') {
        const isOllama = new URL(req.url, 'http://127.0.0.1').searchParams.has('ollama');
        const mock = `let previewOllama=${isOllama};window.acquireVsCodeApi=()=>({getState:()=>({}),setState:()=>{},postMessage:async m=>{try{if(m.type==='openOllama')previewOllama=true;const r=await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...m,previewOllama})});for(const data of await r.json())window.dispatchEvent(new MessageEvent('message',{data}));}catch(e){window.dispatchEvent(new MessageEvent('message',{data:{type:'error',message:e.message}}));}}});`;
        const nonce = crypto.randomBytes(24).toString('base64');
        res.setHeader('Content-Security-Policy', `default-src 'none'; style-src 'self'; img-src 'self' data:; script-src 'nonce-${nonce}'; connect-src 'self'`);
        res.setHeader('Content-Type', 'text/html'); res.end(`<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WeightScope Preview</title><link rel="stylesheet" href="/styles.css"></head><body><div id="app"></div><script nonce="${nonce}">${mock}</script><script nonce="${nonce}" src="/model-view.js"></script><script nonce="${nonce}" src="/app.js"></script></body></html>`);
      } else if (req.method === 'GET' && route === '/favicon.ico') {
        res.writeHead(204); res.end();
      } else if (req.method === 'GET' && ['/styles.css', '/model-view.js', '/app.js'].includes(route)) {
        res.setHeader('Content-Type', route.endsWith('.css') ? 'text/css' : 'text/javascript'); res.end(await fs.readFile(path.join(__dirname, '..', 'media', route.slice(1))));
      } else if (req.method === 'POST' && route === '/action') {
        let body = ''; for await (const chunk of req) { body += chunk; if (body.length > 2 * 1024 * 1024) throw new Error('Request too large.'); }
        const m = JSON.parse(body), result = [], base = m.previewOllama ? ollamaModel : model;
        if (m.type === 'compare' || m.type === 'compareOllama' || m.type === 'compareSaved') {
          if (m.type === 'compareSaved' && !savedPath) throw new Error('Save a checkpoint copy first.');
          selectedTarget = m.type === 'compareSaved' ? await backend.inspect(savedPath) : m.previewOllama ? ollamaTarget : target;
          rows = backend.structureDiff(base, selectedTarget);
        }
        const comparison = selectedTarget;
        if (m.type === 'ready' || m.type === 'refresh' || m.type === 'openOllama') result.push({ type: 'model', model: publicModel(m.type === 'openOllama' ? ollamaModel : base), settings: options }, editPayload());
        else if (m.type === 'stageMetadata') {
          if (!checkpoint.capabilities(base).metadata) throw new Error('This checkpoint does not support metadata editing.');
          const change = { key: m.key, value: m.value, ...(m.valueType ? { type: m.valueType } : {}) };
          const next = new Map(edits); next.set(change.key, change); metadata.validateForModel(base, [...next.values()]);
          edits.set(change.key, change); result.push(editPayload());
        } else if (m.type === 'stageTensor') {
          const edit = { name: m.name, operation: m.operation, factor: m.factor, ...(m.operation === 'replace' ? { source: { model: target, name: m.name } } : {}) };
          const next = new Map(tensorPlan); next.set(edit.name, edit); tensorEdits.validate(base, [...next.values()]);
          tensorPlan.set(edit.name, edit); result.push(editPayload());
        } else if (m.type === 'discardEdits') { edits.clear(); tensorPlan.clear(); result.push(editPayload()); }
        else if (m.type === 'saveCheckpoint') {
          const output = path.join(__dirname, '..', 'output', 'preview', `saved-${crypto.randomUUID()}.safetensors`);
          const saved = await checkpoint.saveCheckpoint(base, output, { ...options, metadata: [...edits.values()], tensors: [...tensorPlan.values()] }); savedPath = saved.path; result.push({ type: 'checkpointSaved', path: saved.path });
        }
        else if (m.type === 'stats') result.push({ type: 'statistics', name: m.name, data: await backend.tensorStatistics(base, m.name, options) });
        else if (m.type === 'compare' || m.type === 'compareOllama' || m.type === 'compareSaved') result.push({ type: 'comparison', target: backend.publicModel(comparison), rows, metadata: metadataDiff(base, comparison) });
        else if (m.type === 'analyzeDiff') {
          const candidates = m.name ? rows.filter(r => r.name === m.name) : rows.filter(r => r.status === 'unverified');
          for (const row of candidates) { row.metrics = await backend.compareTensor(base, comparison, row.name, options); row.status = row.metrics.status; result.push({ type: 'diffRow', row, completed: result.length + 1, total: candidates.length }); }
          result.push({ type: 'diffDone' });
        } else if (m.type === 'export' || m.type === 'openSaved') result.push({ type: 'error', message: 'File dialogs, opening saved editors, and export are available in the VS Code extension. This browser preview uses generated local demo checkpoints.' });
        res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(result));
      } else { res.writeHead(404); res.end('Not found'); }
    } catch (e) { res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify([{ type: 'error', message: e.message }])); }
  });
  server.on('error', e => { console.error(`Cannot start local preview: ${e.message}`); process.exitCode = 1; });
  const port = Number(process.env.WEIGHTSCOPE_PREVIEW_PORT || 4173);
  server.listen(port, '127.0.0.1', () => console.log(`WeightScope preview: http://127.0.0.1:${port}`));
}
main().catch(e => { console.error(e); process.exitCode = 1; });
