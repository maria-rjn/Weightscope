'use strict';
// Hugging Face sharded checkpoints: <prefix>-00001-of-0000N.<ext> files plus <prefix>.<ext>.index.json.
// Safetensors shards are read natively; PyTorch .bin shards go through the Python helper.
const fs = require('node:fs/promises');
const path = require('node:path');
const FORMATS = { safetensors: { extension: '.safetensors', title: 'safetensors' }, pytorch: { extension: '.bin', title: 'PyTorch' } };
const SHARD = /^(.+)-(\d+)-of-(\d+)(\.safetensors|\.bin)$/i;
const INDEX_LIMIT = 64 * 1024 * 1024, MAX_SHARDS = 10000, MAX_INDEX_CANDIDATES = 16;

const indexSuffix = format => `${FORMATS[format].extension}.index.json`;
function indexFormat(file) { const base = path.basename(file).toLowerCase(); return Object.keys(FORMATS).find(f => base.endsWith(indexSuffix(f))) || null; }
function isIndex(file) { return indexFormat(file) !== null; }
// The format of a file named like a Hugging Face shard, or null.
function shardFormat(file) { const match = SHARD.exec(path.basename(file)); return match && Object.keys(FORMATS).find(f => FORMATS[f].extension === match[4].toLowerCase()); }
// Hugging Face cache snapshots live under models--<owner>--<name>/snapshots/<revision>; elsewhere the folder names the model.
function label(directory, prefix, extension = FORMATS.safetensors.extension) {
  const parts = path.resolve(directory).split(path.sep).filter(Boolean);
  const cache = parts.at(-2) === 'snapshots' && /^(?:models|datasets|spaces)--(.+)$/.exec(parts.at(-3) || '');
  const owner = cache ? cache[1].replace('--', '/') : parts.at(-1);
  return `${owner ? `${owner}/` : ''}${prefix}${extension}`;
}
// Shards must be plain file names beside the index, so an index cannot reach other directories.
function shardName(name, format) {
  const { extension, title } = FORMATS[format];
  if (typeof name !== 'string' || !name || /[\\/\0]/.test(name) || ['.', '..'].includes(name)) throw new Error(`Invalid shard file name in the ${title} index: ${String(name).slice(0, 120)}`);
  if (!name.toLowerCase().endsWith(extension)) throw new Error(`The ${title} index refers to a file that is not ${extension}: ${name.slice(0, 120)}`);
  return name;
}
async function readIndex(file) {
  const base = path.basename(file), format = indexFormat(file), handle = await fs.open(file, 'r');
  let index;
  try {
    if ((await handle.stat()).size > INDEX_LIMIT) throw new Error(`${base} exceeds the 64 MiB index limit.`);
    try { index = JSON.parse((await handle.readFile()).toString('utf8')); } catch (e) { throw new Error(`${base} is not valid JSON: ${e.message}`); }
  } finally { await handle.close(); }
  const map = index?.weight_map;
  if (!map || typeof map !== 'object' || Array.isArray(map)) throw new Error(`${base} has no weight_map object.`);
  const weightMap = new Map(Object.entries(map).map(([name, shard]) => [name, shardName(shard, format)]));
  if (!weightMap.size) throw new Error(`${base} lists no tensors.`);
  const files = [...new Set(weightMap.values())].sort();
  if (files.length > MAX_SHARDS) throw new Error(`${base} lists more than ${MAX_SHARDS} shards.`);
  const directory = path.dirname(file), prefix = base.slice(0, -indexSuffix(format).length);
  const metadata = index.metadata && typeof index.metadata === 'object' && !Array.isArray(index.metadata) ? index.metadata : {};
  return { format, name: label(directory, prefix, FORMATS[format].extension), indexPath: file, directory, files, weightMap, metadata, warnings: [] };
}
async function isFile(file) {
  try { return (await fs.stat(file)).isFile(); } catch (e) { if (['ENOENT', 'ENOTDIR'].includes(e.code)) return false; throw e; }
}
async function missingFiles(directory, files) {
  const missing = [];
  for (const name of files) if (!await isFile(path.join(directory, name))) missing.push(name);
  return missing;
}
// Returns null for ordinary files and a shard set otherwise. Throws when an index is invalid or a set is incomplete.
async function locate(file) {
  if (isIndex(file)) return readIndex(file);
  const base = path.basename(file), match = SHARD.exec(base);
  if (!match) return null;
  // Hugging Face repositories often ship safetensors and .bin shards side by side; each shard only uses an index of its own format.
  const [, prefix, number, total, extension] = match, format = shardFormat(file), suffix = indexSuffix(format);
  const directory = path.dirname(file), preferred = `${prefix}${suffix}`;
  const candidates = [preferred];
  try { candidates.push(...(await fs.readdir(directory)).filter(n => n !== preferred && n.toLowerCase().endsWith(suffix)).sort().slice(0, MAX_INDEX_CANDIDATES)); }
  catch { /* The preferred index is still tried directly. */ }
  const problems = [];
  for (const candidate of candidates) {
    try { const index = await readIndex(path.join(directory, candidate)); if (index.files.includes(base)) return index; }
    catch (e) { if (e.code !== 'ENOENT') problems.push(e.message); }
  }
  if (problems.length) throw new Error(problems.join(' '));
  const count = Number(total), position = Number(number);
  if (count < 2 || count > MAX_SHARDS || position < 1 || position > count) return null;
  // Without an index, shards that follow the Hugging Face naming pattern are grouped by name.
  const files = Array.from({ length: count }, (_, i) => `${prefix}-${String(i + 1).padStart(number.length, '0')}-of-${total}${extension}`);
  const missing = await missingFiles(directory, files);
  if (missing.length) throw new Error(`This file is shard ${position} of ${count}, but ${preferred} was not found and ${missing.length} shard file(s) are missing, including ${missing[0]}.`);
  return { format, name: label(directory, prefix, FORMATS[format].extension), indexPath: null, directory, files, weightMap: null, metadata: {},
    warnings: [`${preferred} was not found, so shards were grouped by file name. Loaders that require the index may not open this checkpoint.`] };
}
module.exports = { FORMATS, locate, readIndex, isIndex, shardFormat, label, missingFiles };
