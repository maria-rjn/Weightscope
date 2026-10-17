'use strict';
const { safeInteger } = require('./binary');
const { fields } = require('./parsers/onnx');

const TYPES = { string: 8, bool: 7, u8: 0, i8: 1, u16: 2, i16: 3, u32: 4, i32: 5, f32: 6, u64: 10, i64: 11, f64: 12 };
function u32(n) { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; }
function u64(n) { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; }
function string(value) { const b = Buffer.from(value); if (b.length > 1024 * 1024) throw new Error('Metadata strings are limited to 1 MiB.'); return Buffer.concat([u64(b.length), b]); }
function validateChanges(changes) {
  if (!Array.isArray(changes) || changes.length > 10000) throw new Error('Metadata edits must be a list with at most 10000 entries.');
  const keys = new Set();
  for (const change of changes) {
    if (!change || typeof change.key !== 'string' || !change.key.length || change.key.length > 1024 || !Object.hasOwn(change, 'value')) throw new Error('Each metadata edit needs a key and a value (null removes a key).');
    if (keys.has(change.key)) throw new Error(`Duplicate metadata edit: ${change.key}`); keys.add(change.key);
    if (change.type !== undefined && !Object.hasOwn(TYPES, change.type)) throw new Error('Choose a supported scalar metadata type.');
    if (change.value !== null && !['string', 'number', 'boolean'].includes(typeof change.value)) throw new Error('Metadata values must be strings, numbers, booleans, or null.');
    if (typeof change.value === 'number' && !Number.isFinite(change.value)) throw new Error('Metadata numbers must be finite.');
  }
  if (Buffer.byteLength(JSON.stringify(changes)) > 2 * 1024 * 1024) throw new Error('Metadata edits exceed the 2 MiB limit.');
  return changes;
}
function scalar(type, value) {
  if (type === 8) { if (typeof value !== 'string') throw new Error('String metadata requires a string.'); return string(value); }
  if (type === 7) { if (typeof value !== 'boolean') throw new Error('Boolean metadata requires true or false.'); return Buffer.from([Number(value)]); }
  if (type === 10 || type === 11) {
    if (!(typeof value === 'string' && /^-?\d+$/.test(value)) && !(typeof value === 'number' && Number.isSafeInteger(value))) throw new Error('64-bit metadata integers require an exact decimal string or a safe integer.');
    const b = Buffer.alloc(8); b[type === 10 ? 'writeBigUInt64LE' : 'writeBigInt64LE'](BigInt(value)); return b;
  }
  if (type === 6 || type === 12) {
    if (typeof value !== 'number' || !Number.isFinite(value) || (type === 6 && !Number.isFinite(Math.fround(value)))) throw new Error('Floating-point metadata must fit its finite numeric type.');
    const b = Buffer.alloc(type === 6 ? 4 : 8); b[type === 6 ? 'writeFloatLE' : 'writeDoubleLE'](value); return b;
  }
  const spec = { 0: [1, 'writeUInt8'], 1: [1, 'writeInt8'], 2: [2, 'writeUInt16LE'], 3: [2, 'writeInt16LE'], 4: [4, 'writeUInt32LE'], 5: [4, 'writeInt32LE'] }[type];
  if (!spec || typeof value !== 'number' || !Number.isSafeInteger(value)) throw new Error('Only scalar metadata can be edited; integer metadata requires an integer.');
  const b = Buffer.alloc(spec[0]); b[spec[1]](value); return b;
}
function validateForModel(model, changes) {
  validateChanges(changes);
  if (!['safetensors', 'gguf', 'onnx'].includes(model.format)) throw new Error('Metadata editing is supported for safetensors, GGUF, and ONNX checkpoints.');
  for (const change of changes) {
    if (change.key === 'general.alignment' && model.format === 'gguf') throw new Error('general.alignment cannot be edited because it controls tensor storage layout.');
    if (change.value === null) continue;
    if (model.format !== 'gguf') { if (typeof change.value !== 'string') throw new Error('Metadata property values must be strings.'); continue; }
    const existing = model.layout.metadataEntries.find(e => e.key === change.key);
    const type = existing?.type ?? (change.type === undefined ? 8 : TYPES[change.type]);
    if (existing && change.type !== undefined && TYPES[change.type] !== existing.type) throw new Error('Existing GGUF metadata keys retain their original type.');
    scalar(type, change.value);
  }
  return changes;
}
async function writeSafetensors(reader, output, changes, io) {
  const size = safeInteger((await reader.read(0, 8)).readBigUInt64LE()), header = JSON.parse((await reader.read(8, size)).toString('utf8'));
  const metadata = { ...header.__metadata__ };
  for (const { key, value } of changes) { if (value === null) delete metadata[key]; else Object.defineProperty(metadata, key, { value, enumerable: true, configurable: true }); }
  if (Object.values(metadata).some(v => typeof v !== 'string')) throw new Error('Safetensors metadata values must all be strings.');
  header.__metadata__ = metadata;
  const json = Buffer.from(JSON.stringify(header)), padding = Buffer.alloc((8 - json.length % 8) % 8, 32);
  if (json.length + padding.length > 64 * 1024 * 1024) throw new Error('Edited safetensors header exceeds 64 MiB.');
  await io.writeAll(output, Buffer.concat([u64(json.length + padding.length), json, padding]));
  await io.copyRange(reader, output, 8 + size, reader.size - 8 - size);
}
async function writeGGUF(model, reader, output, changes, io) {
  const patches = new Map(changes.map(c => [c.key, c])), entries = [];
  if (patches.has('general.alignment')) throw new Error('general.alignment cannot be edited because it controls tensor storage layout.');
  for (const entry of model.layout.metadataEntries) {
    const change = patches.get(entry.key); patches.delete(entry.key);
    if (!change) entries.push({ start: entry.start, length: entry.end - entry.start });
    else if (change.value !== null) entries.push(Buffer.concat([string(entry.key), u32(entry.type), scalar(entry.type, change.value)]));
  }
  for (const change of patches.values()) if (change.value !== null) {
    const type = change.type === undefined ? 8 : TYPES[change.type];
    entries.push(Buffer.concat([string(change.key), u32(type), scalar(type, change.value)]));
  }
  const prefix = await reader.read(0, 24); prefix.writeBigUInt64LE(BigInt(entries.length), 16); await io.writeAll(output, prefix);
  let size = 24;
  for (const entry of entries) {
    if (Buffer.isBuffer(entry)) { await io.writeAll(output, entry); size += entry.length; }
    else { await io.copyRange(reader, output, entry.start, entry.length); size += entry.length; }
  }
  await io.copyRange(reader, output, model.layout.tensorInfoStart, model.layout.tensorInfoEnd - model.layout.tensorInfoStart);
  size += model.layout.tensorInfoEnd - model.layout.tensorInfoStart;
  await io.writeAll(output, Buffer.alloc((model.layout.alignment - size % model.layout.alignment) % model.layout.alignment));
  if (reader.size > model.layout.dataStart) await io.copyRange(reader, output, model.layout.dataStart, reader.size - model.layout.dataStart);
}
function varint(n) { let value = BigInt(n); const bytes = []; do { let b = Number(value & 127n); value >>= 7n; if (value) b |= 128; bytes.push(b); } while (value); return Buffer.from(bytes); }
function field(no, data) { const b = typeof data === 'string' ? Buffer.from(data) : data; return Buffer.concat([varint(no * 8 + 2), varint(b.length), b]); }
async function writeONNX(reader, output, changes, io) {
  const patches = new Map(changes.map(c => [c.key, c]));
  for (const change of changes) if (change.value !== null && typeof change.value !== 'string') throw new Error('ONNX metadata property values must be strings.');
  for await (const f of fields(reader, 0, reader.size)) {
    let changed = false;
    if (f.no === 14 && f.wire === 2) {
      for await (const item of fields(reader, f.start, f.length)) if (item.no === 1 && item.wire === 2) {
        const key = (await reader.read(item.start, item.length)).toString('utf8'); changed = patches.has(key); break;
      }
    }
    if (!changed) await io.copyRange(reader, output, f.fieldStart, f.end - f.fieldStart);
  }
  for (const { key, value } of changes) if (value !== null) await io.writeAll(output, field(14, Buffer.concat([field(1, key), field(2, value)])));
}
async function writeMetadata(model, reader, output, changes, io) {
  validateForModel(model, changes);
  if (model.format === 'safetensors') return writeSafetensors(reader, output, changes, io);
  if (model.format === 'gguf') return writeGGUF(model, reader, output, changes, io);
  if (model.format === 'onnx') return writeONNX(reader, output, changes, io);
  throw new Error('Metadata editing is supported for safetensors, GGUF, and ONNX checkpoints.');
}
module.exports = { writeMetadata, validateChanges, validateForModel, TYPES };
