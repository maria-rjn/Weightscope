'use strict';
const backend = require('./backend');
const { Reader, WIDTHS, scalar, cancelled } = require('./binary');

const FLOATS = new Set(['F64', 'F32', 'F16', 'BF16']);
// GGML block scale offsets. Scaling these fields keeps the encoded quantization codes intact.
const SCALES = { Q4_0: [0], Q4_1: [0, 2], Q5_0: [0], Q5_1: [0, 2], Q8_0: [0], Q2_K: [80, 82], Q3_K: [108], Q4_K: [0, 2], Q5_K: [0, 2], Q6_K: [208], Q8_K: [0] };
function ranges(tensor) {
  const s = tensor.storage;
  if (['raw', 'gguf-blocks'].includes(s.kind) && Number.isSafeInteger(s.length)) return [{ offset: s.offset, length: s.length }];
  if (s.kind === 'onnx-typed' && ['F32', 'F64'].includes(tensor.dtype) && s.fields.every(f => f.no === (tensor.dtype === 'F32' ? 4 : 10) && [1, 2, 5].includes(f.wire) && f.length % WIDTHS[tensor.dtype] === 0) &&
      s.fields.reduce((n, f) => n + f.length, 0) === tensor.bytes) return s.fields.map(f => ({ offset: f.start, length: f.length }));
  return null;
}
function capabilities(model, tensor) {
  if (model.source?.kind === 'ollama' && model.format === 'safetensors') return { operations: [], reason: 'Open a single-file checkpoint to edit tensors.' };
  const spans = ranges(tensor);
  if (model.format === 'pytorch') return { operations: tensor.storage.editable === false ? [] : FLOATS.has(tensor.dtype) ? ['replace', 'scale', 'zero'] : ['replace'], reason: 'PyTorch edits require dense real, non-quantized tensors and load and reserialize the checkpoint on CPU in a trusted workspace.' };
  if (!spans || tensor.bytes === null) return { operations: [], reason: 'Editing requires inline fixed-width storage. External ONNX weights, variable-length typed data, and unknown block layouts are inspect-only.' };
  const arithmetic = tensor.storage.kind === 'gguf-blocks' ? Object.hasOwn(SCALES, tensor.dtype) : FLOATS.has(tensor.dtype) && !tensor.quantization;
  return { operations: arithmetic ? ['replace', 'scale', 'zero'] : ['replace'], reason: arithmetic ? '' : 'This dtype supports compatible byte-for-byte replacement. Scaling and zeroing require supported floating-point or GGUF block storage.' };
}
function validate(model, edits) {
  if (!Array.isArray(edits) || edits.length > 10000) throw new Error('Tensor edits must be a list with at most 10000 entries.');
  const names = new Set();
  for (const edit of edits) {
    const tensor = model.byName.get(edit?.name);
    if (!tensor) throw new Error('Choose an existing tensor to edit.');
    if (names.has(edit.name)) throw new Error('Only one operation can be staged per tensor.'); names.add(edit.name);
    if (!capabilities(model, tensor).operations.includes(edit.operation)) throw new Error(`Operation ${edit.operation} is unavailable for ${tensor.name}.`);
    if (edit.operation === 'scale' && (typeof edit.factor !== 'number' || !Number.isFinite(edit.factor))) throw new Error('The scale factor must be a finite number.');
    if (edit.operation === 'replace') {
      const donor = edit.source?.model, other = donor?.byName.get(edit.source.name);
      if (!other || tensor.dtype !== other.dtype || JSON.stringify(tensor.shape) !== JSON.stringify(other.shape) || tensor.bytes !== other.bytes) throw new Error('Replacement tensors must have the same shape, dtype, and byte length.');
      if (model.format === 'pytorch' ? donor.format !== 'pytorch' : !ranges(other)) throw new Error('The replacement storage is incompatible. PyTorch replacement requires another PyTorch checkpoint.');
      if ((tensor.storage.kind === 'gguf-blocks') !== (other.storage.kind === 'gguf-blocks')) throw new Error('Replacement tensors must use the same encoded storage layout.');
      if (other.storage.file || other.analysisUnavailable) throw new Error('Choose a standalone replacement checkpoint.');
    }
  }
  return edits;
}
function roundEven(value) { const lo = Math.floor(value), fraction = value - lo; return fraction > .5 || (fraction === .5 && lo % 2) ? lo + 1 : lo; }
function float16(value, mantissa, bias) {
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0, abs = Math.abs(value), minimumExponent = 1 - bias;
  if (!abs) return sign;
  if (abs < 2 ** minimumExponent) return sign | roundEven(abs / 2 ** (minimumExponent - mantissa));
  let exponent = Math.floor(Math.log2(abs));
  let significand = roundEven(abs / 2 ** (exponent - mantissa));
  if (significand === 2 ** (mantissa + 1)) { exponent++; significand /= 2; }
  if (exponent + bias >= (1 << (15 - mantissa)) - 1) throw new Error('Scaling overflows the tensor dtype.');
  return sign | ((exponent + bias) << mantissa) | (significand - 2 ** mantissa);
}
function writeScalar(buffer, offset, dtype, value) {
  if (!Number.isFinite(value)) throw new Error('Scaling requires finite weights and must not overflow. Use zeroing to clear non-finite values.');
  if (dtype === 'F64') buffer.writeDoubleLE(value, offset);
  else if (dtype === 'F32') { if (!Number.isFinite(Math.fround(value))) throw new Error('Scaling overflows the tensor dtype.'); buffer.writeFloatLE(value, offset); }
  else buffer.writeUInt16LE(float16(value, dtype === 'F16' ? 10 : 7, dtype === 'F16' ? 15 : 127), offset);
}
function transform(buffer, tensor, edit) {
  if (tensor.storage.kind === 'gguf-blocks') {
    const dtype = tensor.dtype === 'Q8_K' ? 'F32' : 'F16';
    for (let block = 0; block < buffer.length; block += tensor.storage.blockBytes) for (const pos of SCALES[tensor.dtype]) {
      const offset = block + pos;
      writeScalar(buffer, offset, dtype, edit.operation === 'zero' ? 0 : scalar(buffer, offset, dtype) * edit.factor);
    }
  } else if (edit.operation === 'zero') buffer.fill(0);
  else for (let pos = 0; pos < buffer.length; pos += WIDTHS[tensor.dtype]) writeScalar(buffer, pos, tensor.dtype, scalar(buffer, pos, tensor.dtype) * edit.factor);
}
async function writeAt(output, buffer, position) {
  let offset = 0;
  while (offset < buffer.length) { const { bytesWritten } = await output.write(buffer, offset, buffer.length - offset, position + offset); if (!bytesWritten) throw new Error('Could not write tensor bytes.'); offset += bytesWritten; }
}
async function applyNative(model, saved, reader, output, edits, options) {
  validate(model, edits);
  for (const edit of edits) {
    cancelled(options.signal);
    const tensor = model.byName.get(edit.name), target = saved.byName.get(edit.name);
    const donor = edit.operation === 'replace' ? edit.source.model : model, source = edit.operation === 'replace' ? donor.byName.get(edit.source.name) : tensor;
    await backend.assertCurrent(donor);
    const sourceReader = donor === model ? reader : await Reader.open(donor.path, options.signal);
    try {
      if (backend.identity(await sourceReader.handle.stat()) !== donor.identity) throw new Error('Replacement checkpoint changed on disk. Inspect it again.');
      const from = ranges(source), to = ranges(target), unit = tensor.storage.kind === 'gguf-blocks' ? tensor.storage.blockBytes : WIDTHS[tensor.dtype] || 1;
      const chunk = Math.max(unit, Math.floor(1024 * 1024 / unit) * unit);
      let si = 0, sp = 0;
      for (const span of to) {
        let tp = 0;
        while (tp < span.length) {
          while (si < from.length && sp === from[si].length) { si++; sp = 0; }
          if (si >= from.length) throw new Error('Replacement storage is shorter than the target tensor.');
          const count = Math.min(chunk, span.length - tp, from[si].length - sp), buffer = await sourceReader.read(from[si].offset + sp, count);
          if (edit.operation !== 'replace') transform(buffer, tensor, edit);
          await writeAt(output, buffer, span.offset + tp); tp += count; sp += count;
        }
      }
      await backend.assertCurrent(donor);
    } finally { if (sourceReader !== reader) await sourceReader.close(); }
  }
}
module.exports = { capabilities, validate, applyNative, writeScalar };
