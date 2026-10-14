'use strict';
const { elements, safeInteger, WIDTHS } = require('../binary');

async function parseSafetensors(reader) {
  const length = safeInteger((await reader.read(0, 8)).readBigUInt64LE(), 'Header length');
  if (length < 2 || length > 64 * 1024 * 1024) throw new Error('Invalid safetensors header length (maximum 64 MiB).');
  const raw = await reader.read(8, length);
  if (raw[0] !== 123) throw new Error('The safetensors header must start with a JSON object.');
  const header = JSON.parse(raw.toString('utf8'));
  const tensors = [];
  for (const [name, t] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    if (!t || typeof t.dtype !== 'string' || !Array.isArray(t.data_offsets) || t.data_offsets.length !== 2) throw new Error(`Invalid tensor header: ${name}`);
    const count = elements(t.shape), [start, end] = t.data_offsets.map(n => safeInteger(n, 'Tensor offset'));
    if (end < start || 8 + length + end > reader.size) throw new Error(`Invalid data range: ${name}`);
    const width = WIDTHS[t.dtype];
    if (width && count * width !== end - start) throw new Error(`Tensor byte length does not match shape and dtype: ${name}`);
    tensors.push({ name, dtype: t.dtype, shape: t.shape, elements: count, bytes: end - start,
      storage: { kind: 'raw', offset: 8 + length + start, length: end - start } });
  }
  const ranges = tensors.filter(t => t.bytes).sort((a, b) => a.storage.offset - b.storage.offset);
  let end = 8 + length;
  for (const t of ranges) {
    if (t.storage.offset !== end) throw new Error('Safetensors data contains an overlap or uncovered range.');
    end += t.bytes;
  }
  if (end !== reader.size) throw new Error('Safetensors data contains trailing bytes or an incomplete header.');
  return { format: 'safetensors', tensors, metadata: header.__metadata__ || {}, warnings: [] };
}
module.exports = { parseSafetensors };
