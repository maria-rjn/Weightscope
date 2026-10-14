'use strict';
const fs = require('node:fs/promises');

function safeInteger(value, label = 'Integer') {
  const n = Number(value);
  if (!Number.isSafeInteger(n) || n < 0) throw new Error(`${label} is invalid or exceeds the supported safe integer range.`);
  return n;
}
function elements(shape) {
  if (!Array.isArray(shape) || shape.length > 64) throw new Error('Invalid tensor shape.');
  shape.forEach(n => safeInteger(n, 'Dimension'));
  return shape.reduce((n, d) => safeInteger(n * d, 'Parameter count'), 1);
}
function cancelled(signal) {
  if (signal?.aborted) throw new Error('Operation cancelled.');
}
class Reader {
  static async open(file, signal) {
    const handle = await fs.open(file, 'r');
    try { return new Reader(handle, (await handle.stat()).size, signal); }
    catch (e) { await handle.close(); throw e; }
  }
  constructor(handle, size, signal) { this.handle = handle; this.size = size; this.signal = signal; this.cache = Buffer.alloc(0); this.cacheStart = 0; }
  async read(start, length) {
    cancelled(this.signal);
    safeInteger(start, 'Offset'); safeInteger(length, 'Length');
    if (start + length > this.size) throw new Error('Truncated file or tensor data outside the file.');
    if (length > 64 * 1024 * 1024) throw new Error('Metadata field exceeds the 64 MiB limit.');
    const buffer = Buffer.alloc(length);
    let n = 0;
    while (n < length) {
      const { bytesRead } = await this.handle.read(buffer, n, length - n, start + n);
      if (!bytesRead) throw new Error('Unexpected end of file.');
      n += bytesRead;
    }
    return buffer;
  }
  async small(start, length) {
    if (start < this.cacheStart || start + length > this.cacheStart + this.cache.length) {
      this.cacheStart = start;
      this.cache = await this.read(start, Math.min(this.size - start, Math.max(65536, length)));
    }
    cancelled(this.signal);
    return this.cache.subarray(start - this.cacheStart, start - this.cacheStart + length);
  }
  close() { return this.handle.close(); }
}
function half(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >> 10) & 31, mant = bits & 1023;
  return exp === 31 ? (mant ? NaN : sign * Infinity) : sign * (exp ? 2 ** (exp - 15) * (1 + mant / 1024) : mant * 2 ** -24);
}
const scratch = Buffer.alloc(4);
function bfloat(bits) { scratch.writeUInt32LE(bits * 65536); return scratch.readFloatLE(); }
const WIDTHS = { F64: 8, F32: 4, F16: 2, BF16: 2, I64: 8, U64: 8, I32: 4, U32: 4, I16: 2, U16: 2, I8: 1, U8: 1, BOOL: 1, F8_E4M3: 1, F8_E5M2: 1, F8_E4M3FN: 1, F8_E4M3FNUZ: 1, F8_E5M2FNUZ: 1 };
function float8(bits, dtype) {
  const e5 = dtype.includes('E5M2'), uz = dtype.endsWith('FNUZ');
  const mantBits = e5 ? 2 : 3, maxExp = e5 ? 31 : 15;
  const exp = (bits & 127) >> mantBits, mant = bits & ((1 << mantBits) - 1);
  const bias = (e5 ? 15 : 7) + (uz ? 1 : 0), sign = bits & 128 ? -1 : 1;
  if (uz && bits === 128) return NaN;
  if (!uz && exp === maxExp && (e5 || mant === 7)) return e5 && !mant ? sign * Infinity : NaN;
  return sign * (exp ? 2 ** (exp - bias) * (1 + mant / (1 << mantBits)) : 2 ** (1 - bias) * mant / (1 << mantBits));
}
function scalar(buffer, offset, dtype) {
  switch (dtype) {
    case 'F32': return buffer.readFloatLE(offset);
    case 'F64': return buffer.readDoubleLE(offset);
    case 'F16': return half(buffer.readUInt16LE(offset));
    case 'BF16': return bfloat(buffer.readUInt16LE(offset));
    case 'I64': return Number(buffer.readBigInt64LE(offset));
    case 'U64': return Number(buffer.readBigUInt64LE(offset));
    case 'BOOL': return buffer[offset] ? 1 : 0;
    default:
      if (dtype.startsWith('F8_')) return float8(buffer[offset], dtype);
      if (/^[IU](8|16|32)$/.test(dtype)) return buffer[`read${dtype[0] === 'I' ? 'Int' : 'UInt'}${dtype.slice(1)}${dtype.slice(1) === '8' ? '' : 'LE'}`](offset);
      throw new Error(`Statistics for ${dtype} are not supported.`);
  }
}

// Same deterministic windows across formats: comparison always pairs identical logical indices.
function sampleWindows(count, limit) {
  safeInteger(count); limit = Math.max(1, Math.min(262144, safeInteger(limit)));
  if (count <= limit) return count ? [{ start: 0, count }] : [];
  const windows = Math.min(32, limit), width = Math.floor(limit / windows);
  return Array.from({ length: windows }, (_, i) => ({ start: Math.floor(i * (count - width) / (windows - 1 || 1)), count: width }));
}
module.exports = { Reader, safeInteger, elements, cancelled, WIDTHS, half, bfloat, scalar, sampleWindows };
