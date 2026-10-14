'use strict';
const fs = require('node:fs/promises');
function u32(n) { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; }
function u64(n) { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; }
function f32(values) { const b = Buffer.alloc(values.length * 4); values.forEach((v, i) => b.writeFloatLE(v, i * 4)); return b; }
function str(s) { const b = Buffer.from(s); return Buffer.concat([u64(b.length), b]); }
async function safetensors(file, tensors, metadata = {}) {
  const header = { __metadata__: metadata }, data = []; let pos = 0;
  for (const t of tensors) {
    const b = t.data || f32(t.values);
    header[t.name] = { dtype: t.dtype || 'F32', shape: t.shape || [t.values.length], data_offsets: [pos, pos + b.length] };
    pos += b.length; data.push(b);
  }
  let json = JSON.stringify(header); json += ' '.repeat((8 - Buffer.byteLength(json) % 8) % 8);
  await fs.writeFile(file, Buffer.concat([u64(Buffer.byteLength(json)), Buffer.from(json), ...data]));
}
async function gguf(file, tensors, metadata = [['general.name', 8, str('Tiny test model')]]) {
  let pos = 0; const info = [], data = [];
  for (const t of tensors) {
    const dims = [...t.shape].reverse();
    info.push(str(t.name), u32(dims.length), ...dims.map(u64), u32(t.type || 0), u64(pos));
    const raw = t.data || f32(t.values), padding = Buffer.alloc((32 - raw.length % 32) % 32);
    data.push(raw, padding); pos += raw.length + padding.length;
  }
  const header = Buffer.concat([Buffer.from('GGUF'), u32(3), u64(tensors.length), u64(metadata.length), ...metadata.flatMap(([key, type, value]) => [str(key), u32(type), value]), ...info]);
  await fs.writeFile(file, Buffer.concat([header, Buffer.alloc((32 - header.length % 32) % 32), ...data]));
}
function vint(n) { let value = BigInt(n); if (value < 0) value = BigInt.asUintN(64, value); const out = []; do { let b = Number(value & 127n); value >>= 7n; if (value) b |= 128; out.push(b); } while (value); return Buffer.from(out); }
function vfield(no, value) { return Buffer.concat([vint(no * 8), vint(value)]); }
function field(no, data) { const b = typeof data === 'string' ? Buffer.from(data) : data; return Buffer.concat([vint(no * 8 + 2), vint(b.length), b]); }
function onnxTensor(t) {
  const parts = [field(1, Buffer.concat((t.shape || [t.values.length]).map(vint))), vfield(2, t.dtype || 1), field(8, t.name)];
  if (t.external) for (const [k, v] of Object.entries(t.external)) parts.push(field(13, Buffer.concat([field(1, k), field(2, String(v))])));
  else parts.push(field(t.typed || 9, t.data || f32(t.values)));
  return Buffer.concat(parts);
}
async function onnx(file, tensors, nodes = []) {
  const graph = Buffer.concat([field(2, 'Test graph'), ...tensors.map(t => field(5, onnxTensor(t))), ...nodes.map(n => field(1, Buffer.concat([field(3, n.name || 'node'), field(4, n.type || 'MatMul'), ...(n.inputs || []).map(s => field(1, s)), ...(n.outputs || ['output']).map(s => field(2, s)), ...(n.attrs || [])]))) ]);
  await fs.writeFile(file, Buffer.concat([vfield(1, 9), field(2, 'WeightScope tests'), field(8, vfield(2, 18)), field(7, graph)]));
}
module.exports = { safetensors, gguf, onnx, f32, str, u32, u64, vint, field, vfield, onnxTensor };
