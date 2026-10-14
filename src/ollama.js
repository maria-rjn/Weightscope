'use strict';
// Read-only access to Ollama's local manifests and content-addressed blobs.
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { cancelled } = require('./binary');
const DIGEST = /^sha256[:-]([0-9a-f]{64})$/i;
const PREFIX = 'application/vnd.ollama.image.';
const MANIFEST_LIMIT = 4 * 1024 * 1024, TEXT_LIMIT = 64 * 1024, MAX_MODELS = 5000, MAX_LAYERS = 200000;

function expandHome(p, home = os.homedir()) { return p === '~' ? home : p.startsWith('~/') || p.startsWith('~\\') ? path.join(home, p.slice(2)) : p; }
function candidateRoots(configured = '', env = process.env, platform = process.platform, home = os.homedir()) {
  const explicit = configured.trim() || env.OLLAMA_MODELS?.trim();
  const list = explicit ? [explicit] : [path.join(home, '.ollama', 'models')];
  if (!explicit && platform === 'linux') list.push('/usr/share/ollama/.ollama/models', '/var/lib/ollama/.ollama/models');
  return [...new Set(list.map(p => path.resolve(expandHome(p, home))))];
}
async function findRoot(configured, env, platform, home) {
  const searched = candidateRoots(configured, env, platform, home);
  for (const root of searched) {
    try { if ((await fs.stat(path.join(root, 'manifests'))).isDirectory()) return { root, searched }; }
    catch (e) { if (!['ENOENT', 'ENOTDIR'].includes(e.code)) throw new Error(`Cannot read Ollama models at ${root}: ${e.message}`); }
  }
  return { root: null, searched };
}
function displayName(host, namespace, model, tag) {
  const repo = host === 'registry.ollama.ai' ? (namespace === 'library' ? model : `${namespace}/${model}`) : `${host}/${namespace}/${model}`;
  return `${repo}:${tag}`;
}
function blobPath(root, digest) {
  const match = DIGEST.exec(String(digest || ''));
  if (!match) throw new Error(`Invalid blob digest: ${String(digest).slice(0, 80)}`);
  return path.join(root, 'blobs', `sha256-${match[1].toLowerCase()}`);
}
// The manifest URI lets VS Code restore a selected model/tag after a restart.
function manifestLocation(file) {
  const parts = path.resolve(file).split(path.sep), index = parts.length - 5;
  if (parts[index] !== 'manifests' || parts.slice(index + 1).some(p => !p || p.startsWith('.'))) return null;
  return { root: parts.slice(0, index).join(path.sep) || path.parse(file).root, parts: parts.slice(index + 1) };
}
async function readJSON(file) {
  const handle = await fs.open(file, 'r');
  try {
    const { size } = await handle.stat();
    if (size > MANIFEST_LIMIT) throw new Error('Manifest exceeds the 4 MiB limit.');
    return JSON.parse((await handle.readFile()).toString('utf8'));
  } finally { await handle.close(); }
}
async function walk(dir, depth, prefix = [], out = []) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const e of entries) {
    if (out.length >= MAX_MODELS) break;
    if (e.name.startsWith('.')) continue;
    const parts = [...prefix, e.name];
    if (e.isDirectory() && depth > 1) await walk(path.join(dir, e.name), depth - 1, parts, out);
    else if (e.isFile() && depth === 1) out.push(parts);
  }
  return out;
}
async function resolveManifest(root, manifest, selectedPaths = [], depth = 0) {
  if (depth > 4) throw new Error('Manifest references exceed the nesting limit.');
  if (Array.isArray(manifest.manifests)) {
    // Prefer a downloaded GGUF variant, then a downloaded tensor variant.
    const candidates = [...manifest.manifests].sort((a, b) => Number(b.format === 'gguf') - Number(a.format === 'gguf'));
    for (const child of candidates.slice(0, 32)) {
      try { return await resolveManifest(root, child, selectedPaths, depth + 1); } catch { /* Try another local variant. */ }
    }
    throw new Error('No supported local variant in this manifest list.');
  }
  if (!Array.isArray(manifest.layers) && manifest.digest) {
    const file = blobPath(root, manifest.digest);
    return resolveManifest(root, await readJSON(file), [...selectedPaths, file], depth + 1);
  }
  if (!Array.isArray(manifest.layers)) throw new Error('No local weight layers (this may be a cloud-only model).');
  if (manifest.layers.length > MAX_LAYERS) throw new Error('Manifest contains too many layers.');
  const weights = manifest.layers.filter(l => l?.mediaType === PREFIX + 'model'), tensors = manifest.layers.filter(l => l?.mediaType === PREFIX + 'tensor');
  if (!weights.length && !tensors.length) throw new Error('No GGUF or safetensors weight layers (this may be a cloud-only model).');
  if (weights.length > 1) throw new Error('Multiple GGUF weight layers are not supported in one manifest.');
  for (const layer of weights.length ? weights : tensors) {
    let file; try { file = blobPath(root, layer.digest); } catch { throw new Error('A weight layer has an invalid digest.'); }
    const stat = await fs.stat(file).catch(e => { throw new Error(e.code === 'ENOENT' ? 'A weight blob is missing. Download the model again with ollama pull.' : e.message); });
    if (!stat.isFile()) throw new Error('A weight blob is not a file.');
  }
  if (weights.length) {
    const handle = await fs.open(blobPath(root, weights[0].digest), 'r');
    try {
      const header = Buffer.alloc(4); await handle.read(header, 0, 4, 0);
      if (header.toString('ascii') !== 'GGUF') throw new Error('The model blob is not a supported little-endian GGUF file.');
    } finally { await handle.close(); }
  }
  return { manifest, selectedPaths };
}
async function describeManifest(root, parts) {
  const [host, namespace, model, tag] = parts, manifestPath = path.join(root, 'manifests', ...parts);
  const entry = { name: displayName(host, namespace, model, tag), registry: host, namespace, model, tag, root, manifestPath, layers: [], size: 0, modelPath: null, problem: null };
  try {
    const { manifest, selectedPaths } = await resolveManifest(root, await readJSON(manifestPath));
    entry.selectedPaths = selectedPaths; entry.runner = manifest.runner;
    if (manifest.config?.digest) { try { entry.configPath = blobPath(root, manifest.config.digest); } catch { /* Optional config. */ } }
    for (const layer of manifest.layers) {
      const kind = typeof layer?.mediaType === 'string' && layer.mediaType.startsWith(PREFIX) ? layer.mediaType.slice(PREFIX.length) : String(layer?.mediaType || 'unknown');
      const size = Number.isSafeInteger(layer?.size) && layer.size >= 0 ? layer.size : 0;
      let file = null; try { file = blobPath(root, layer?.digest); } catch { /* Report invalid weight paths below. */ }
      entry.layers.push({ kind, name: layer?.name, digest: String(layer?.digest || ''), size, path: file });
    }
    const unique = new Set();
    for (const layer of [...entry.layers, manifest.config].filter(Boolean)) {
      if (!unique.has(layer.digest)) { unique.add(layer.digest); entry.size += Number.isSafeInteger(layer.size) && layer.size > 0 ? layer.size : 0; }
    }
    if (!Number.isSafeInteger(entry.size)) throw new Error('Model size exceeds exact numeric precision.');
    const weights = entry.layers.find(l => l.kind === 'model');
    entry.format = weights ? 'gguf' : 'safetensors'; entry.modelPath = weights?.path || null;
  } catch (e) { entry.problem = e.message; }
  return entry;
}
async function fromManifest(file) {
  const location = manifestLocation(file);
  if (!location) throw new Error('Choose an Ollama manifest under models/manifests/<registry>/<namespace>/<model>/<tag>.');
  return describeManifest(location.root, location.parts);
}
async function listModels(root) {
  const manifests = await walk(path.join(root, 'manifests'), 4), models = [];
  // Bound open files even in large local catalogs.
  for (let i = 0; i < manifests.length; i += 8) models.push(...await Promise.all(manifests.slice(i, i + 8).map(parts => describeManifest(root, parts))));
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
async function details(entry, options = {}) {
  const out = { kind: 'ollama', name: entry.name, registry: entry.registry, tag: entry.tag, runner: entry.runner,
    manifestPath: entry.manifestPath, blobPath: entry.modelPath, totalSize: entry.size,
    layers: entry.layers.map(({ kind, name, digest, size }) => ({ kind, name, digest, size })), extraWeights: [] };
  for (const layer of entry.layers) {
    cancelled(options.signal);
    if (!layer.path) continue;
    try {
      if (layer.kind === 'params') { const { text, truncated } = await readText(layer.path); if (truncated) throw new Error('Parameters exceed the 64 KiB preview limit.'); out.params = JSON.parse(text); }
      else if (layer.kind === 'json' && typeof layer.name === 'string' && (layer.name === 'model_index.json' || layer.name.endsWith('/config.json'))) {
        const { text, truncated } = await readText(layer.path); if (truncated) throw new Error('Configuration exceeds the 64 KiB preview limit.');
        const config = JSON.parse(text); (out.configs ||= Object.create(null))[layer.name] = config;
        if (layer.name === 'model_index.json' && typeof config?.quantization === 'string') out.quantization = config.quantization;
      }
      else if (['template', 'system', 'license', 'messages'].includes(layer.kind)) {
        const item = await readText(layer.path);
        if (layer.kind === 'license') (out.licenses ||= []).push(item); else out[layer.kind] = item;
      } else if (['adapter', 'projector'].includes(layer.kind)) out.extraWeights.push({ kind: layer.kind, size: layer.size, path: layer.path });
    } catch (e) { (out.warnings ||= []).push(`${layer.kind}: ${e.message}`); }
  }
  return out;
}
function protectedPaths(entry) {
  return [entry.manifestPath, entry.configPath, ...(entry.selectedPaths || []), ...entry.layers.map(l => l.path)].filter(Boolean);
}
module.exports = { candidateRoots, findRoot, listModels, details, blobPath, displayName, expandHome, manifestLocation, fromManifest, protectedPaths };
