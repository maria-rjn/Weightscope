'use strict';
const { elements, safeInteger, scalar } = require('../binary');

// Ollama/MLX packs four E4M3FN values into each little-endian U32 word,
// with one unsigned E8M0 power-of-two scale per 32 logical values.
// https://github.com/ollama/ollama/blob/main/mlx/quant/quant.go
// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
function layout(tensor, scale, groupSize = 32, hasBias = false) {
  if (tensor.dtype !== 'U32' || !tensor.shape.length) throw new Error('MXFP8 weights must use a non-scalar U32 packed tensor.');
  if (Number(groupSize) !== 32) throw new Error('MXFP8 requires a group size of 32.');
  if (hasBias) throw new Error('MXFP8 does not use a quantization bias tensor.');
  const logicalShape = [...tensor.shape];
  const last = logicalShape.length - 1;
  logicalShape[last] = safeInteger(logicalShape[last] * 4, 'MXFP8 logical dimension');
  if (!logicalShape[last] || logicalShape[last] % 32) throw new Error('MXFP8 rows must contain a multiple of 32 logical values.');
  const logicalElements = elements(logicalShape);
  if (!scale) throw new Error('MXFP8 is missing its companion scale tensor.');
  const expectedShape = [...logicalShape]; expectedShape[last] /= 32;
  if (scale.dtype !== 'U8' || JSON.stringify(scale.shape) !== JSON.stringify(expectedShape) || scale.storage.kind !== 'raw') {
    throw new Error(`MXFP8 requires a U8 scale tensor with shape [${expectedShape.join(', ')}].`);
  }
  if (tensor.storage.length !== logicalElements || scale.storage.length !== logicalElements / 32) throw new Error('MXFP8 storage length does not match its logical layout.');
  return { logicalShape, logicalElements,
    quantization: { scheme: 'mxfp8', decoded: true, bitsPerWeight: 8, blockSize: 32, bytesPerBlock: 33,
      effectiveBitsPerWeight: 8.25, valueDtype: 'F8_E4M3FN', scaleDtype: 'F8_E8M0', decodedDtype: 'F32', scaleTensor: scale.name } };
}

async function sample(reader, scaleReader, tensor, windows) {
  const values = [], scale = tensor.storage.scale;
  for (const w of windows) {
    const firstBlock = Math.floor(w.start / 32), lastBlock = Math.ceil((w.start + w.count) / 32);
    const codes = await reader.read(tensor.storage.offset + w.start, w.count);
    const scales = await scaleReader.read(scale.offset + firstBlock, lastBlock - firstBlock);
    for (let i = 0; i < w.count; i++) {
      const exponent = scales[Math.floor((w.start + i) / 32) - firstBlock];
      // E8M0 has no zero encoding: byte 0 is 2^-127; byte 255 is NaN.
      const factor = exponent === 255 ? NaN : 2 ** (exponent - 127);
      values.push(Math.fround(scalar(codes, i, 'F8_E4M3FN') * factor));
    }
  }
  return values;
}
module.exports = { layout, sample };
