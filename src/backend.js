'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { Reader, WIDTHS, scalar, half, bfloat, safeInteger, sampleWindows, cancelled } = require('./binary');
const { parseSafetensors } = require('./parsers/safetensors');
const { parseGGUF, decodeBlock } = require('./parsers/gguf');
const { parseONNX, varint } = require('./parsers/onnx');
const { statistics, difference, structureDiff } = require('./math');

function identity(s) { return `${s.size}:${s.mtimeMs}:${s.ctimeMs}:${s.ino}`; }
async function assertCurrent(model) {
  if (identity(await fs.stat(model.path)) !== model.identity) throw new Error('The checkpoint changed on disk. Refresh the inspector before analyzing it.');
}
function summarize(model) {
  const dtypes = Object.create(null);
  let parameters = 0, tensorBytes = 0, unknownByteSizes = 0;
  for (const t of model.tensors) {
    parameters = safeInteger(parameters + t.elements, 'Total parameter count');
    if (t.bytes === null) unknownByteSizes++; else tensorBytes = safeInteger(tensorBytes + t.bytes, 'Total tensor bytes');
    const d = dtypes[t.dtype] ||= { tensors: 0, parameters: 0, bytes: 0 };
    d.tensors++; d.parameters += t.elements; d.bytes += t.bytes || 0;
  }
  return { parameters, tensorBytes, unknownByteSizes, tensorCount: model.tensors.length, fileBytes: model.fileBytes, dtypes,
    quantizedTensors: model.tensors.filter(t => t.quantization).length };
}
function publicModel(model) {
  return { name: path.basename(model.path), path: model.path, format: model.format, metadata: model.metadata, warnings: model.warnings,
    summary: model.summary, nodes: model.nodes || [], tensors: model.tensors.map(({ storage, ...t }) => t) };
}
function pythonRequest(request, options = {}) {
  if (!options.trusted) return Promise.reject(new Error('PyTorch inspection requires a trusted VS Code workspace.'));
  const exe = options.pythonPath || (process.platform === 'win32' ? 'py' : 'python3');
  const args = [...(!options.pythonPath && process.platform === 'win32' ? ['-3'] : []), '-I', path.join(__dirname, '..', 'python', 'inspect_pt.py')];
  return new Promise((resolve, reject) => {
    cancelled(options.signal);
    const child = spawn(exe, args, { shell: false, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    let out = '', err = '', settled = false;
    const finish = (error, value) => { if (settled) return; settled = true; clearTimeout(timer); options.signal?.removeEventListener('abort', abort); error ? reject(error) : resolve(value); };
    const abort = () => { child.kill(); finish(new Error('Operation cancelled.')); };
    const timer = setTimeout(() => { child.kill(); finish(new Error('PyTorch inspection timed out after 120 seconds.')); }, 120000);
    options.signal?.addEventListener('abort', abort, { once: true });
    child.on('error', e => finish(new Error(`Cannot start Python (${exe}): ${e.message}. Configure weightscope.pythonPath.`)));
    child.stdout.on('data', chunk => { out += chunk; if (out.length > 64 * 1024 * 1024) { child.kill(); finish(new Error('Python response exceeds the size limit.')); } });
    child.stderr.on('data', chunk => { err = (err + chunk).slice(-8000); });
    child.stdin.on('error', () => {});
    child.on('close', code => {
      if (settled) return;
      try {
        const result = JSON.parse(out);
        if (!result.ok) throw new Error(result.error || err || `Python exited with code ${code}.`);
        finish(null, result.data);
      } catch (e) { finish(new Error(out ? e.message : err || 'No response from the PyTorch helper.')); }
    });
    child.stdin.end(JSON.stringify(request));
  });
}
async function inspect(file, options = {}) {
  const stat = await fs.stat(file);
  if (!stat.isFile()) throw new Error('Choose a model file, not a directory.');
  const ext = path.extname(file).toLowerCase();
  let parsed;
  if (['.pt', '.pth'].includes(ext)) parsed = await pythonRequest({ action: 'inspect', path: file }, options);
  else {
    const reader = await Reader.open(file, options.signal);
    try {
      if (ext === '.safetensors') parsed = await parseSafetensors(reader);
      else if (ext === '.gguf') parsed = await parseGGUF(reader);
      else if (ext === '.onnx') parsed = await parseONNX(reader);
      else throw new Error('Supported model files: .safetensors, .gguf, .onnx, .pt, .pth.');
    } finally { await reader.close(); }
  }
  const model = { ...parsed, path: file, identity: identity(stat), fileBytes: stat.size };
  await assertCurrent(model);
  model.summary = summarize(model);
  model.byName = new Map(model.tensors.map(t => [t.name, t]));
  return model;
}
async function rawValues(reader, tensor, windows) {
  const s = tensor.storage, values = [];
  if (s.kind === 'gguf-blocks') {
    if (!tensor.quantization?.decoded) throw new Error(`Numerical decoding for ${tensor.dtype} is not available; inspect its quantization metadata instead.`);
    for (const w of windows) {
      const first = Math.floor(w.start / s.blockSize), last = Math.ceil((w.start + w.count) / s.blockSize);
      const b = await reader.read(s.offset + first * s.blockBytes, (last - first) * s.blockBytes);
      for (let i = w.start; i < w.start + w.count; i++) {
        const blockOffset = (Math.floor(i / s.blockSize) - first) * s.blockBytes;
        values.push(decodeBlock(b.subarray(blockOffset, blockOffset + s.blockBytes), tensor.dtype, i % s.blockSize));
      }
    }
    return values;
  }
  const width = WIDTHS[tensor.dtype];
  if (!width) throw new Error(`Statistics for ${tensor.dtype} are not supported.`);
  if (s.length < tensor.elements * width) throw new Error('Tensor storage is shorter than its declared shape.');
  for (const w of windows) {
    const b = await reader.read(s.offset + w.start * width, w.count * width);
    for (let i = 0; i < w.count; i++) {
      const v = scalar(b, i * width, tensor.dtype);
      if (['I64', 'U64'].includes(tensor.dtype) && !Number.isSafeInteger(v)) throw new Error('64-bit integer weights exceed exact JavaScript precision; numerical analysis is unavailable.');
      values.push(v);
    }
  }
  return values;
}
async function typedValues(reader, tensor, windows) {
  if (!WIDTHS[tensor.dtype]) throw new Error(`Statistics for ${tensor.dtype} are not supported.`);
  const values = []; let index = 0, wi = 0;
  function accept(v) {
    if (index >= tensor.elements) throw new Error('ONNX typed storage exceeds its tensor shape.');
    while (wi < windows.length && index >= windows[wi].start + windows[wi].count) wi++;
    if (wi < windows.length && index >= windows[wi].start) values.push(v);
    index++;
  }
  function integer(bits) {
    if (tensor.dtype === 'F16') return half(Number(bits & 65535n));
    if (tensor.dtype === 'BF16') return bfloat(Number(bits & 65535n));
    if (tensor.dtype.startsWith('F8_')) return scalar(Buffer.from([Number(bits & 255n)]), 0, tensor.dtype);
    const value = tensor.dtype[0] === 'I' ? BigInt.asIntN(Number(tensor.dtype.slice(1)), bits) : bits;
    const n = Number(value);
    if (!Number.isSafeInteger(n)) throw new Error('Integer weight exceeds exact JavaScript precision.');
    return n;
  }
  for (const f of tensor.storage.fields) {
    if (f.wire === 0) { accept(integer(f.value)); continue; }
    if ([4, 10].includes(f.no)) {
      const dtype = f.no === 4 ? 'F32' : 'F64', width = WIDTHS[dtype];
      if (f.length % width) throw new Error('Invalid ONNX floating-point storage.');
      for (let p = f.start; p < f.start + f.length; p += width) accept(scalar(await reader.small(p, width), 0, dtype));
    } else {
      if (f.wire !== 2) throw new Error('Invalid ONNX integer storage wire type.');
      let p = f.start;
      while (p < f.start + f.length) { const v = await varint(reader, p, f.start + f.length); p = v.pos; accept(integer(v.value)); }
    }
  }
  if (index !== tensor.elements) throw new Error('ONNX typed storage does not match its tensor shape.');
  return values;
}
async function externalStorage(model, tensor) {
  const directory = await fs.realpath(path.dirname(model.path)), s = tensor.storage;
  if (path.isAbsolute(s.location)) throw new Error('Absolute ONNX external-data paths are not allowed.');
  const candidate = path.resolve(directory, s.location);
  const within = file => { const rel = path.relative(directory, file); return rel && !rel.startsWith(`..${path.sep}`) && rel !== '..' && !path.isAbsolute(rel); };
  if (!within(candidate)) throw new Error('ONNX external weights must stay inside the model directory.');
  const file = await fs.realpath(candidate);
  if (!within(file)) throw new Error('ONNX external weight symlink points outside the model directory.');
  const stat = await fs.stat(file), width = WIDTHS[tensor.dtype];
  const offset = safeInteger(Number(s.offset || 0), 'External offset');
  const length = s.length ? safeInteger(Number(s.length), 'External length') : (width ? tensor.elements * width : stat.size - offset);
  return { file, identity: identity(stat), tensor: { ...tensor, storage: { kind: 'raw', offset, length } } };
}
async function sample(model, name, options = {}) {
  cancelled(options.signal); await assertCurrent(model);
  const tensor = model.byName.get(name);
  if (!tensor) throw new Error('Unknown tensor.');
  const windows = sampleWindows(tensor.elements, options.sampleSize || 32768);
  let values;
  if (model.format === 'pytorch') {
    const result = await pythonRequest({ action: 'sample', path: model.path, name, windows }, options);
    values = result.map(v => v === 'NaN' ? NaN : v === 'Infinity' ? Infinity : v === '-Infinity' ? -Infinity : v);
  } else {
    const external = tensor.storage.kind === 'external' ? await externalStorage(model, tensor) : null;
    const current = external?.tensor || tensor;
    const reader = await Reader.open(external?.file || model.path, options.signal);
    try {
      if (current.storage.kind === 'unsupported') throw new Error(current.storage.reason);
      values = current.storage.kind === 'onnx-typed' ? await typedValues(reader, current, windows) : await rawValues(reader, current, windows);
      if (external && identity(await fs.stat(external.file)) !== external.identity) throw new Error('External ONNX weights changed during analysis. Retry the operation.');
    } finally { await reader.close(); }
  }
  await assertCurrent(model); cancelled(options.signal);
  return values;
}
async function tensorStatistics(model, name, options) {
  const tensor = model.byName.get(name);
  return statistics(await sample(model, name, options), tensor.elements, tensor.valueBasis || 'Weight values');
}
async function compareTensor(base, target, name, options = {}) {
  const row = structureDiff({ tensors: [base.byName.get(name)].filter(Boolean) }, { tensors: [target.byName.get(name)].filter(Boolean) })[0];
  if (!row || row.status !== 'unverified') throw new Error('This tensor has incompatible shapes or numerical units, or is missing from a checkpoint.');
  const left = await sample(base, name, options), right = await sample(target, name, options);
  return difference(left, right, base.byName.get(name).elements, options.absoluteTolerance || 0);
}
module.exports = { inspect, publicModel, sample, tensorStatistics, compareTensor, structureDiff, pythonRequest };
