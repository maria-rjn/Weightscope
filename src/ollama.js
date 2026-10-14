'use strict';
// Read-only discovery of models pulled by Ollama.
// Ollama stores each model as content-addressed blobs plus a small JSON manifest per tag:
//   <models>/manifests/<registry>/<namespace>/<model>/<tag>   (OCI-style manifest)
//   <models>/blobs/sha256-<hex>                               (GGUF weights, template, params, license…)
// Nothing here contacts the Ollama server or modifies the model store.
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');

const DIGEST = /^sha256[:-]([0-9a-f]{64})$/;
const PREFIX = 'application/vnd.ollama.image.';
const MANIFEST_LIMIT = 1024 * 1024, TEXT_LIMIT = 64 * 1024, MAX_MODELS = 5000;
const DEFAULT_REGISTRY = 'registry.ollama.ai';

function expandHome(p, home = os.homedir()) { return p === '~' ? home : p.startsWith('~/') || p.startsWith('~\\') ? path.join(home, p.slice(2)) : p; }
// Same lookup order Ollama uses: explicit setting, OLLAMA_MODELS, the per-user store, then Linux service installs.
function candidateRoots(configured = '', env = process.env, platform = process.platform, home = os.homedir()) {
  const list = [configured, env.OLLAMA_MODELS, path.join(home, '.ollama', 'models')];
  if (platform === 'linux') list.push('/usr/share/ollama/.ollama/models', '/var/lib/ollama/.ollama/models');
  return [...new Set(list.filter(p => typeof p === 'string' && p.trim()).map(p => path.resolve(expandHome(p.trim(), home))))];
}
async function isDirectory(p) { try { return (await fs.stat(p)).isDirectory(); } catch { return false; } }
async function findRoot(configured, env, platform, home) {
  const searched = candidateRoots(configured, env, platform, home);
  for (const root of searched) if (await isDirectory(path.join(root, 'manifests'))) return { root, searched };
  return { root: null, searched };
}
function displayName(host, namespace, model, tag) {
  const repo = host === DEFAULT_REGISTRY ? (namespace === 'library' ? model : `${namespace}/${model}`) : `${host}/${namespace}/${model}`;
  return `${repo}:${tag}`;
}
function blobPath(root, digest) {
  const match = DIGEST.exec(String(digest || ''));
  if (!match) throw new Error(`Invalid blob digest: ${String(digest).slice(0, 80)}`);
  return path.join(root, 'blobs', `sha256-${match[1]}`);
}
async function readManifest(file) {
  const handle = await fs.open(file, 'r');
  try {
    const { size } = await handle.stat();
    if (size > MANIFEST_LIMIT) throw new Error('Manifest is larger than 1 MiB.');
    return JSON.parse((await handle.readFile()).toString('utf8'));
  } finally { await handle.close(); }
}
async function walk(dir, depth, prefix = []) {
  const out = [];
  let entries; try { entries = await fs.readdir(dir, { withFileTypes: true }); } catch { return out; }
  for (const e of entries) {
    if (e.name.startsWith('.')) continue;
    const parts = [...prefix, e.name];
    if (e.isDirectory() && depth > 1) out.push(...await walk(path.join(dir, e.name), depth - 1, parts));
    else if (e.isFile() && depth === 1) out.push(parts);
  }
  return out;
}
async function describeManifest(root, parts) {
  const [host, namespace, model, tag] = parts, manifestPath = path.join(root, 'manifests', ...parts);
  const entry = { name: displayName(host, namespace, model, tag), registry: host, namespace, model, tag, root, manifestPath, layers: [], size: 0, modelPath: null, problem: null };
  try {
    const manifest = await readManifest(manifestPath), stat = await fs.stat(manifestPath);
    entry.modified = stat.mtimeMs;
    if (!Array.isArray(manifest.layers)) throw new Error('Manifest has no layer list.');
    for (const layer of manifest.layers.slice(0, 64)) {
      const kind = typeof layer?.mediaType === 'string' && layer.mediaType.startsWith(PREFIX) ? layer.mediaType.slice(PREFIX.length) : String(layer?.mediaType || 'unknown');
      const size = Number.isSafeInteger(layer?.size) && layer.size >= 0 ? layer.size : 0;
      let file = null; try { file = blobPath(root, layer?.digest); } catch { /* reported below */ }
      entry.layers.push({ kind, digest: String(layer?.digest || ''), size, path: file });
      entry.size += size;
    }
    const weights = entry.layers.find(l => l.kind === 'model');
    if (!weights) entry.problem = 'No GGUF model layer in this manifest.';
    else if (!weights.path) entry.problem = 'The model layer has an invalid digest.';
    else if (!(await fs.stat(weights.path).then(s => s.isFile(), () => false))) entry.problem = 'The model blob is missing. Run `ollama pull` again.';
    else entry.modelPath = weights.path;
  } catch (e) { entry.problem = `Unreadable manifest: ${e.message}`; }
  return entry;
}
async function listModels(root) {
  const manifests = (await walk(path.join(root, 'manifests'), 4)).slice(0, MAX_MODELS);
  const models = await Promise.all(manifests.map(parts => describeManifest(root, parts)));
  return models.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
}
async function readText(file, limit = TEXT_LIMIT) {
  const handle = await fs.open(file, 'r');
  try {
    const { size } = await handle.stat(), buffer = Buffer.alloc(Math.min(size, limit));
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    return { text: buffer.subarray(0, bytesRead).toString('utf8'), truncated: size > limit, size };
  } finally { await handle.close(); }
}
// Small text layers that explain how Ollama runs the weights: prompt template, system prompt, default parameters, license.
async function details(entry) {
  const out = { kind: 'ollama', name: entry.name, registry: entry.registry, tag: entry.tag, manifestPath: entry.manifestPath, blobPath: entry.modelPath,
    totalSize: entry.size, layers: entry.layers.map(({ kind, digest, size }) => ({ kind, digest, size })), extraWeights: [] };
  for (const layer of entry.layers) {
    if (!layer.path) continue;
    try {
      if (layer.kind === 'params') { const { text } = await readText(layer.path); out.params = JSON.parse(text); }
      else if (['template', 'system', 'license', 'messages'].includes(layer.kind)) {
        const r = await readText(layer.path); const item = { text: r.text, truncated: r.truncated, size: r.size };
        if (layer.kind === 'license') (out.licenses ||= []).push(item); else out[layer.kind] = item;
      } else if (['adapter', 'projector'].includes(layer.kind)) out.extraWeights.push({ kind: layer.kind, size: layer.size, path: layer.path });
    } catch (e) { (out.warnings ||= []).push(`${layer.kind}: ${e.message}`); }
  }
  return out;
}
module.exports = { candidateRoots, findRoot, listModels, details, blobPath, displayName, expandHome };
