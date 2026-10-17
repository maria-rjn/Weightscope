'use strict';
const { safeInteger, elements, half } = require('../binary');

// [name, elements per block, bytes per block], from ggml's public type layouts.
const TYPES = {
  0: ['F32', 1, 4], 1: ['F16', 1, 2], 2: ['Q4_0', 32, 18], 3: ['Q4_1', 32, 20],
  6: ['Q5_0', 32, 22], 7: ['Q5_1', 32, 24], 8: ['Q8_0', 32, 34], 9: ['Q8_1', 32, null],
  10: ['Q2_K', 256, 84], 11: ['Q3_K', 256, 110], 12: ['Q4_K', 256, 144], 13: ['Q5_K', 256, 176], 14: ['Q6_K', 256, 210], 15: ['Q8_K', 256, 292],
  16: ['IQ2_XXS', 256, 66], 17: ['IQ2_XS', 256, 74], 18: ['IQ3_XXS', 256, 98], 19: ['IQ1_S', 256, 50], 20: ['IQ4_NL', 32, 18],
  21: ['IQ3_S', 256, 110], 22: ['IQ2_S', 256, 82], 23: ['IQ4_XS', 256, 136],
  24: ['I8', 1, 1], 25: ['I16', 1, 2], 26: ['I32', 1, 4], 27: ['I64', 1, 8], 28: ['F64', 1, 8], 29: ['IQ1_M', 256, 56], 30: ['BF16', 1, 2],
  34: ['TQ1_0', 256, 54], 35: ['TQ2_0', 256, 66], 39: ['MXFP4', 32, 17], 40: ['NVFP4', 64, 36], 41: ['Q1_0', 128, 18], 42: ['Q2_0', 64, 18]
};
const DECODED = new Set(['Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_K']);
async function parseGGUF(reader) {
  let pos = 0;
  async function take(n) { const b = await reader.small(pos, n); if (b.length !== n) throw new Error('Truncated GGUF header.'); pos += n; return b; }
  const magic = (await take(4)).toString('ascii');
  if (magic !== 'GGUF') throw new Error('Invalid GGUF magic. Big-endian GGUF is not supported.');
  const version = (await take(4)).readUInt32LE();
  if (version !== 2 && version !== 3) throw new Error(`Unsupported GGUF version ${version}; expected little-endian v2 or v3.`);
  const u64 = async () => safeInteger((await take(8)).readBigUInt64LE(), 'GGUF length');
  const u32 = async () => (await take(4)).readUInt32LE();
  async function string(keep = true) {
    const length = await u64();
    if (length > 16 * 1024 * 1024 || pos + length > reader.size) throw new Error('Invalid GGUF string length.');
    if (!keep) { pos += length; return null; }
    const b = await reader.read(pos, length); pos += length; return b.toString('utf8');
  }
  async function value(type, keep = true, depth = 0) {
    if (depth > 8) throw new Error('GGUF metadata nesting exceeds the limit.');
    if (type === 8) return string(keep);
    if (type === 9) {
      const itemType = await u32(), count = await u64();
      if (count > 2000000) throw new Error('GGUF metadata array exceeds the 2M entry limit.');
      const preview = [];
      for (let i = 0; i < count; i++) { const v = await value(itemType, keep && i < 32, depth + 1); if (keep && i < 32) preview.push(v); }
      return keep ? { count, preview, truncated: count > 32 } : null;
    }
    const widths = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8];
    if (!widths[type]) throw new Error(`Unknown GGUF metadata type ${type}.`);
    const b = await take(widths[type]);
    switch (type) {
      case 0: return b.readUInt8(); case 1: return b.readInt8(); case 2: return b.readUInt16LE(); case 3: return b.readInt16LE();
      case 4: return b.readUInt32LE(); case 5: return b.readInt32LE(); case 6: return b.readFloatLE(); case 7: return Boolean(b[0]);
      case 10: return b.readBigUInt64LE().toString(); case 11: return b.readBigInt64LE().toString(); case 12: return b.readDoubleLE();
    }
  }
  const tensorCount = await u64(), kvCount = await u64();
  if (tensorCount > 200000 || kvCount > 200000) throw new Error('GGUF header entry count exceeds the limit.');
  const metadata = Object.create(null), tensors = [], warnings = [], metadataEntries = [];
  for (let i = 0; i < kvCount; i++) {
    const start = pos, key = await string(), type = await u32();
    if (Object.hasOwn(metadata, key)) throw new Error(`Duplicate GGUF metadata key: ${key}`);
    metadata[key] = await value(type); metadataEntries.push({ key, type, start, end: pos });
  }
  const tensorInfoStart = pos;
  const names = new Set();
  for (let i = 0; i < tensorCount; i++) {
    const name = await string(), dims = await u32();
    if (names.has(name)) throw new Error(`Duplicate GGUF tensor name: ${name}`); names.add(name);
    if (dims > 4 || dims < 1) throw new Error('Invalid GGUF dimension count.');
    const shape = []; for (let d = 0; d < dims; d++) shape.push(await u64());
    const type = await u32(), offset = await u64(), spec = TYPES[type], count = elements(shape);
    if (spec && (count % spec[1] || shape[0] % spec[1])) throw new Error(`GGUF tensor does not align to quantization blocks: ${name}`);
    const bytes = spec?.[2] ? safeInteger(count / spec[1] * spec[2], 'Tensor byte size') : null;
    tensors.push({ name, shape: shape.reverse(), dtype: spec?.[0] || `GGML_TYPE_${type}`, elements: count, bytes,
      quantization: spec && spec[1] > 1 ? { scheme: spec[0], blockSize: spec[1], bytesPerBlock: spec[2], bitsPerWeight: spec[2] ? spec[2] * 8 / spec[1] : null, decoded: DECODED.has(spec[0]) } : null,
      storage: { kind: spec && spec[1] > 1 ? 'gguf-blocks' : 'raw', offset, length: bytes, blockSize: spec?.[1], blockBytes: spec?.[2] } });
  }
  const alignment = safeInteger(Number(metadata['general.alignment'] ?? 32), 'GGUF alignment');
  if (!alignment || alignment > 65536 || (alignment & (alignment - 1))) throw new Error('Invalid GGUF alignment.');
  const dataStart = Math.ceil(pos / alignment) * alignment;
  if (dataStart > reader.size && tensors.length) throw new Error('Missing GGUF data section.');
  for (const t of tensors) {
    if (t.storage.offset % alignment) throw new Error(`Unaligned GGUF tensor offset: ${t.name}`);
    t.storage.offset += dataStart;
    if (t.storage.offset > reader.size || (t.bytes !== null && t.storage.offset + t.bytes > reader.size)) throw new Error(`Truncated GGUF tensor: ${t.name}`);
  }
  const ranges = tensors.filter(t => t.bytes).sort((a, b) => a.storage.offset - b.storage.offset);
  for (let i = 1; i < ranges.length; i++) if (ranges[i].storage.offset < ranges[i - 1].storage.offset + ranges[i - 1].bytes) throw new Error('Overlapping GGUF tensor data.');
  if (tensors.some(t => t.bytes === null)) warnings.push('Unknown or version-dependent GGML block layouts have unavailable byte sizes and statistics (including Q8_1).');
  if (tensors.some(t => t.quantization && !t.quantization.decoded)) warnings.push('IQ and newer quantization types expose metadata; numerical decoding is available for Q4_0/1, Q5_0/1, Q8_0, and Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K.');
  return { format: 'gguf', tensors, metadata: { version, ...metadata }, warnings,
    layout: { metadataEntries, tensorInfoStart, tensorInfoEnd: pos, dataStart, alignment } };
}
function decodeBlock(b, dtype, index) {
  if (!DECODED.has(dtype)) throw new Error(`Numerical decoding for ${dtype} is not implemented. Quantization metadata remains available.`);
  if (dtype === 'Q8_K') return b.readFloatLE(0) * b.readInt8(4 + index);
  if (dtype === 'Q2_K') {
    const scale = b[Math.floor(index / 16)];
    const quant = (b[16 + Math.floor(index / 128) * 32 + index % 32] >> (Math.floor(index % 128 / 32) * 2)) & 3;
    return half(b.readUInt16LE(80)) * (scale & 15) * quant - half(b.readUInt16LE(82)) * (scale >> 4);
  }
  if (dtype === 'Q3_K') {
    const group = Math.floor(index / 16);
    const lowScale = (b[96 + group % 8] >> (group >= 8 ? 4 : 0)) & 15;
    const highScale = (b[104 + group % 4] >> (Math.floor(group / 4) * 2)) & 3;
    const scale = (lowScale | highScale << 4) - 32;
    const low = (b[32 + Math.floor(index / 128) * 32 + index % 32] >> (Math.floor(index % 128 / 32) * 2)) & 3;
    const high = (b[index % 32] >> Math.floor(index / 32)) & 1;
    return half(b.readUInt16LE(108)) * scale * (low - (high ? 0 : 4));
  }
  if (dtype === 'Q4_K' || dtype === 'Q5_K') {
    const group = Math.floor(index / 32), j = group % 4;
    const scale = group < 4 ? b[4 + j] & 63 : (b[12 + j] & 15) | ((b[4 + j] >> 2) & 48);
    const min = group < 4 ? b[8 + j] & 63 : (b[12 + j] >> 4) | ((b[8 + j] >> 2) & 48);
    const base = dtype === 'Q4_K' ? 16 : 48;
    let q = (b[base + Math.floor(group / 2) * 32 + index % 32] >> ((group % 2) * 4)) & 15;
    if (dtype === 'Q5_K') q |= ((b[16 + index % 32] >> group) & 1) << 4;
    return half(b.readUInt16LE(0)) * scale * q - half(b.readUInt16LE(2)) * min;
  }
  if (dtype === 'Q6_K') {
    const halfBlock = Math.floor(index / 128), local = index % 128;
    const low = (b[halfBlock * 64 + local % 64] >> (local >= 64 ? 4 : 0)) & 15;
    const high = (b[128 + halfBlock * 32 + index % 32] >> (Math.floor(local / 32) * 2)) & 3;
    return half(b.readUInt16LE(208)) * b.readInt8(192 + Math.floor(index / 16)) * ((low | high << 4) - 32);
  }
  const scale = half(b.readUInt16LE(0));
  if (dtype === 'Q8_0') return scale * b.readInt8(2 + index);
  const affine = dtype.endsWith('_1'), five = dtype.startsWith('Q5');
  const base = affine ? 4 : 2, quants = base + (five ? 4 : 0);
  let q = index < 16 ? b[quants + index] & 15 : b[quants + index - 16] >> 4;
  if (five) q |= ((b.readUInt32LE(base) >>> index) & 1) << 4;
  return scale * (q - (affine ? 0 : five ? 16 : 8)) + (affine ? half(b.readUInt16LE(2)) : 0);
}
module.exports = { parseGGUF, decodeBlock, TYPES };
