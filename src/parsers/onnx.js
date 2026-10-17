'use strict';
const { safeInteger, elements, WIDTHS } = require('../binary');

const DTYPES = { 1: 'F32', 2: 'U8', 3: 'I8', 4: 'U16', 5: 'I16', 6: 'I32', 7: 'I64', 8: 'STRING', 9: 'BOOL', 10: 'F16', 11: 'F64', 12: 'U32', 13: 'U64', 14: 'COMPLEX64', 15: 'COMPLEX128', 16: 'BF16', 17: 'F8_E4M3FN', 18: 'F8_E4M3FNUZ', 19: 'F8_E5M2', 20: 'F8_E5M2FNUZ', 21: 'U4', 22: 'I4', 23: 'F4_E2M1', 24: 'F8_E8M0', 25: 'U2', 26: 'I2', 27: 'F6_E2M3', 28: 'F6_E3M2' };
async function varint(reader, pos, end) {
  let value = 0n;
  for (let i = 0; i < 10; i++) {
    if (pos >= end) throw new Error('Truncated ONNX protobuf varint.');
    const b = (await reader.small(pos++, 1))[0];
    if (i === 9 && b > 1) throw new Error('Invalid protobuf varint.');
    value |= BigInt(b & 127) << BigInt(i * 7);
    if (!(b & 128)) return { value, pos };
  }
  throw new Error('Invalid protobuf varint.');
}
async function* fields(reader, start, length) {
  const end = start + length;
  if (end > reader.size) throw new Error('ONNX protobuf field extends outside the file.');
  let pos = start, count = 0;
  while (pos < end) {
    const fieldStart = pos;
    if (++count > 2000000) throw new Error('ONNX message has too many fields.');
    const key = await varint(reader, pos, end); pos = key.pos;
    const tag = safeInteger(key.value), no = Math.floor(tag / 8), wire = tag & 7;
    if (!no || no > 536870911) throw new Error('Invalid ONNX protobuf field number.');
    if (wire === 0) {
      const result = await varint(reader, pos, end); pos = result.pos;
      yield { no, wire, value: result.value, fieldStart, end: pos }; continue;
    }
    let size;
    if (wire === 1) size = 8;
    else if (wire === 5) size = 4;
    else if (wire === 2) { const result = await varint(reader, pos, end); pos = result.pos; size = safeInteger(result.value, 'ONNX field length'); }
    else throw new Error(`Unsupported protobuf wire type ${wire}.`);
    if (pos + size > end) throw new Error('Truncated ONNX protobuf field.');
    yield { no, wire, start: pos, length: size, fieldStart, end: pos + size };
    pos += size;
  }
}
async function text(reader, f) {
  if (f.wire !== 2 || f.length > 1024 * 1024) throw new Error('Invalid or oversized ONNX text field.');
  return (await reader.read(f.start, f.length)).toString('utf8');
}
async function entry(reader, field) {
  let key = '', value = '';
  for await (const f of fields(reader, field.start, field.length)) {
    if (f.no === 1) key = await text(reader, f);
    if (f.no === 2) value = await text(reader, f);
  }
  return [key, value];
}
async function parseONNX(reader) {
  const tensors = [], nodes = [], warnings = [], metadata = Object.create(null), names = new Set();
  function add(t) {
    if (names.has(t.name)) throw new Error(`Duplicate ONNX tensor name: ${t.name}`);
    if (tensors.length >= 200000) throw new Error('ONNX tensor count exceeds the limit.');
    names.add(t.name); tensors.push(t);
  }
  async function tensor(field, prefix, fallback) {
    let name = '', dtype = 'UNDEFINED', shape = [], raw = null, typed = [], external = Object.create(null), segmented = false;
    for await (const f of fields(reader, field.start, field.length)) {
      if (f.no === 1) {
        if (f.wire === 0) shape.push(safeInteger(f.value, 'ONNX dimension'));
        else if (f.wire === 2) {
          let p = f.start;
          while (p < f.start + f.length) { const v = await varint(reader, p, f.start + f.length); shape.push(safeInteger(v.value, 'ONNX dimension')); p = v.pos; if (shape.length > 64) throw new Error('Too many ONNX dimensions.'); }
        }
      } else if (f.no === 2) dtype = DTYPES[Number(f.value)] || `ONNX_TYPE_${f.value}`;
      else if (f.no === 3) segmented = true;
      else if (f.no === 8) name = await text(reader, f);
      else if (f.no === 9) raw = { kind: 'raw', offset: f.start, length: f.length };
      else if ([4, 5, 7, 10, 11].includes(f.no)) typed.push(f);
      else if (f.no === 13) { const [k, v] = await entry(reader, f); external[k] = v; }
    }
    const count = elements(shape), width = WIDTHS[dtype];
    let storage = raw || { kind: 'onnx-typed', fields: typed };
    if (external.location) storage = { kind: 'external', location: external.location, offset: external.offset, length: external.length };
    if (segmented) storage = { kind: 'unsupported', reason: 'Segmented ONNX tensor data is not supported.' };
    if (raw && width && raw.length !== count * width) throw new Error(`ONNX tensor byte length does not match its shape: ${name || fallback}`);
    return { name: prefix + (name || fallback), dtype, shape, elements: count, bytes: width ? safeInteger(count * width) : null, storage };
  }
  async function graph(field, prefix = '', depth = 0) {
    if (depth > 8) throw new Error('ONNX graph nesting exceeds the limit.');
    const annotations = [];
    for await (const f of fields(reader, field.start, field.length)) {
      if (f.no === 5) add(await tensor(f, prefix, `initializer_${tensors.length}`));
      else if (f.no === 2) { const name = await text(reader, f); if (!prefix) metadata.graph = name; }
      else if (f.no === 15) warnings.push('Sparse ONNX initializers are excluded from tensor totals.');
      else if (f.no === 14) annotations.push(f);
      else if (f.no === 1) {
        if (nodes.length >= 100000) throw new Error('ONNX operator count exceeds the limit.');
        const node = { name: '', opType: '', inputs: [], outputs: [] }, attrs = [];
        for await (const n of fields(reader, f.start, f.length)) {
          if (n.no === 1) node.inputs.push(await text(reader, n));
          else if (n.no === 2) node.outputs.push(await text(reader, n));
          else if (n.no === 3) node.name = await text(reader, n);
          else if (n.no === 4) node.opType = await text(reader, n);
          else if (n.no === 5) attrs.push(n);
        }
        node.name = prefix + (node.name || `${node.opType}_${nodes.length}`);
        node.inputs = node.inputs.map(n => prefix + n); node.outputs = node.outputs.map(n => prefix + n);
        nodes.push(node);
        for (let i = 0; i < attrs.length; i++) {
          let attrName = `attr_${i}`;
          for await (const a of fields(reader, attrs[i].start, attrs[i].length)) {
            if (a.no === 1) attrName = await text(reader, a);
            else if (a.no === 5 && node.opType === 'Constant') add(await tensor(a, prefix, (node.outputs[0] || node.name).slice(prefix.length)));
            else if (a.no === 6 || a.no === 11) await graph(a, `${node.name}/${attrName}/`, depth + 1);
          }
        }
      }
    }
    for (const field of annotations) {
      let name = '', params = Object.create(null);
      for await (const a of fields(reader, field.start, field.length)) {
        if (a.no === 1) name = prefix + await text(reader, a);
        if (a.no === 2) { const [k, v] = await entry(reader, a); params[k] = prefix + v; }
      }
      const t = tensors.find(t => t.name === name);
      if (t) { t.quantization = { scheme: 'ONNX annotated quantization', parameters: params, decoded: false }; t.valueBasis = 'Stored numeric codes (not dequantized weights)'; }
    }
  }
  let graphFound = false;
  for await (const f of fields(reader, 0, reader.size)) {
    if (f.no === 7) { await graph(f); graphFound = true; }
    else if (f.no === 1) metadata.irVersion = safeInteger(f.value);
    else if (f.no === 2) metadata.producer = await text(reader, f);
    else if (f.no === 3) metadata.producerVersion = await text(reader, f);
    else if (f.no === 14) { const [k, v] = await entry(reader, f); metadata[k] = v; }
    else if (f.no === 8) {
      let domain = 'ai.onnx', version = 0;
      for await (const o of fields(reader, f.start, f.length)) { if (o.no === 1) domain = await text(reader, o) || 'ai.onnx'; if (o.no === 2) version = safeInteger(o.value); }
      metadata[`opset.${domain}`] = version;
    }
  }
  if (!graphFound) throw new Error('This file does not contain an ONNX model graph.');
  const byName = new Map(tensors.map(t => [t.name, t]));
  for (const n of nodes) {
    if (n.opType === 'DequantizeLinear' || n.opType === 'QuantizeLinear') {
      const t = byName.get(n.inputs[0]);
      if (t) { t.quantization = { scheme: n.opType, scaleTensor: n.inputs[1], zeroPointTensor: n.inputs[2], decoded: false }; t.valueBasis = 'Stored numeric codes (not dequantized weights)'; }
    }
  }
  if (tensors.some(t => t.storage.kind === 'external')) warnings.push('External ONNX weight files are read only when statistics are requested; paths must remain inside the model directory.');
  return { format: 'onnx', tensors, nodes, metadata, warnings: [...new Set(warnings)] };
}
module.exports = { parseONNX, fields, varint };
