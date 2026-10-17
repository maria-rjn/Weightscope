'use strict';
// Hugging Face sharded safetensors: <prefix>-00001-of-0000N.safetensors files plus <prefix>.safetensors.index.json.
const fs = require('node:fs/promises');
const path = require('node:path');
const INDEX = '.safetensors.index.json', EXTENSION = '.safetensors';
const SHARD = /^(.+)-(\d+)-of-(\d+)\.safetensors$/i;
const INDEX_LIMIT = 64 * 1024 * 1024, MAX_SHARDS = 10000, MAX_INDEX_CANDIDATES = 16;

function isIndex(file) { return path.basename(file).toLowerCase().endsWith(INDEX); }
// Hugging Face cache snapshots live under models--<owner>--<name>/snapshots/<revision>; elsewhere the folder names the model.
function label(directory, prefix) {
  const parts = path.resolve(directory).split(path.sep).filter(Boolean);
  const cache = parts.at(-2) === 'snapshots' && /^(?:models|datasets|spaces)--(.+)$/.exec(parts.at(-3) || '');
  const owner = cache ? cache[1].replace('--', '/') : parts.at(-1);
  return `${owner ? `${owner}/` : ''}${prefix}${EXTENSION}`;
}
// Shards must be plain file names beside the index, so an index cannot reach other directories.
function shardName(name) {
  if (typeof name !== 'string' || !name || /[\\/\0]/.test(name) || ['.', '..'].includes(name)) throw new Error(`Invalid shard file name in the safetensors index: ${String(name).slice(0, 120)}`);
  if (!name.toLowerCase().endsWith(EXTENSION)) throw new Error(`The safetensors index refers to a file that is not .safetensors: ${name.slice(0, 120)}`);
  return name;
}
async function readIndex(file) {
  const base = path.basename(file), handle = await fs.open(file, 'r');
  let index;
  try {
    if ((await handle.stat()).size > INDEX_LIMIT) throw new Error(`${base} exceeds the 64 MiB index limit.`);
    try { index = JSON.parse((await handle.readFile()).toString('utf8')); } catch (e) { throw new Error(`${base} is not valid JSON: ${e.message}`); }
  } finally { await handle.close(); }
  const map = index?.weight_map;
  if (!map || typeof map !== 'object' || Array.isArray(map)) throw new Error(`${base} has no weight_map object.`);
  const weightMap = new Map(Object.entries(map).map(([name, shard]) => [name, shardName(shard)]));
  if (!weightMap.size) throw new Error(`${base} lists no tensors.`);
  const files = [...new Set(weightMap.values())].sort();
  if (files.length > MAX_SHARDS) throw new Error(`${base} lists more than ${MAX_SHARDS} shards.`);
  const directory = path.dirname(file), prefix = base.slice(0, -INDEX.length);
  const metadata = index.metadata && typeof index.metadata === 'object' && !Array.isArray(index.metadata) ? index.metadata : {};
  return { name: label(directory, prefix), indexPath: file, directory, files, weightMap, metadata, warnings: [] };
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
  const [, prefix, number, total] = match, directory = path.dirname(file), preferred = `${prefix}${INDEX}`;
  const candidates = [preferred];
  try { candidates.push(...(await fs.readdir(directory)).filter(n => n !== preferred && n.toLowerCase().endsWith(INDEX)).sort().slice(0, MAX_INDEX_CANDIDATES)); }
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
  const extension = base.slice(-EXTENSION.length);
  const files = Array.from({ length: count }, (_, i) => `${prefix}-${String(i + 1).padStart(number.length, '0')}-of-${total}${extension}`);
  const missing = await missingFiles(directory, files);
  if (missing.length) throw new Error(`This file is shard ${position} of ${count}, but ${preferred} was not found and ${missing.length} shard file(s) are missing, including ${missing[0]}.`);
  return { name: label(directory, prefix), indexPath: null, directory, files, weightMap: null, metadata: {},
    warnings: [`${preferred} was not found, so shards were grouped by file name. Loaders that require the index may not open this checkpoint.`] };
}
module.exports = { locate, readIndex, isIndex, label, missingFiles };
