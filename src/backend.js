'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { Reader, WIDTHS, scalar, half, bfloat, safeInteger, sampleWindows, cancelled } = require('./binary');
const { parseSafetensors } = require('./parsers/safetensors');
const { parseGGUF, decodeBlock } = require('./parsers/gguf');
const { parseONNX, varint } = require('./parsers/onnx');
const { statistics, difference, structureRow, structureDiff } = require('./math');
const { pairTensors, sourceIndex } = require('./tensor-names');
const ollama = require('./ollama');
const sharded = require('./sharded');
const mxfp8 = require('./quantization/mxfp8');

function identity(s) { return `${s.size}:${s.mtimeMs}:${s.ctimeMs}:${s.ino}`; }
async function assertCurrent(model) {
  if (identity(await fs.stat(model.path)) !== model.identity) throw new Error('The checkpoint changed on disk. Refresh the inspector before analyzing it.');
  for (const item of model.dependencies || []) if (identity(await fs.stat(item.path)) !== item.identity) throw new Error(`${model.source?.kind === 'sharded' ? 'The safetensors index' : 'An Ollama manifest'} changed on disk. Refresh the inspector before analyzing it.`);
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
  return { name: model.source?.name || path.basename(model.path), path: model.path, format: model.format, metadata: model.metadata, warnings: model.warnings, source: model.source,
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
  cancelled(options.signal);
  if (ollama.manifestLocation(file)) return inspectOllama(file, options);
  // Opening any shard opens the whole checkpoint. A shard whose set cannot be assembled still opens on its own.
  try { const set = await sharded.locate(file); if (set) return await inspectSharded(file, set, options); }
  catch (e) {
    if (sharded.isIndex(file) || options.signal?.aborted) throw e;
    const model = await inspectFile(file, options);
    model.warnings.push(`Showing this shard only: ${e.message} Totals and comparisons cover this file alone.`);
    return model;
  }
  return inspectFile(file, options);
}
async function inspectFile(file, options = {}) {
  const stat = await fs.stat(file);
  if (!stat.isFile()) throw new Error('Choose a model file, not a directory.');
  const ext = path.extname(file).toLowerCase();
  let parsed;
  if (['.pt', '.pth'].includes(ext)) parsed = await pythonRequest({ action: 'inspect', path: file }, options);
  else {
    const reader = await Reader.open(file, options.signal);
    try {
      if (ext === '.safetensors' || options.format === 'safetensors') parsed = await parseSafetensors(reader);
      else if (ext === '.gguf') parsed = await parseGGUF(reader);
      else if (ext === '.onnx') parsed = await parseONNX(reader);
      else if (reader.size >= 4 && (await reader.read(0, 4)).toString('ascii') === 'GGUF') parsed = await parseGGUF(reader);
      else throw new Error('Supported model files: .safetensors (or a sharded .safetensors.index.json), .gguf, .onnx, .pt, .pth, or a GGUF blob. Use Open Ollama Model for downloaded models.');
    } finally { await reader.close(); }
  }
  const model = { ...parsed, path: file, identity: identity(stat), fileBytes: stat.size };
  await assertCurrent(model);
  model.summary = summarize(model);
  model.byName = new Map(model.tensors.map(t => [t.name, t]));
  return model;
}
async function inspectOllama(file, options) {
  const manifestStat = await fs.stat(file), entry = await ollama.fromManifest(file);
  if (entry.problem) throw new Error(`${entry.name}: ${entry.problem}`);
  const dependencies = [];
  for (const dependency of [file, ...(entry.selectedPaths || [])]) dependencies.push({ path: dependency, identity: identity(await fs.stat(dependency)) });
  if (identity(manifestStat) !== dependencies[0].identity) throw new Error('Ollama manifest changed during inspection. Refresh the inspector.');
  const source = await ollama.details(entry, options);
  let model;
  if (entry.modelPath) model = await inspect(entry.modelPath, options);
  else {
    const tensors = [], names = new Map(), shards = new Map(), schemes = new Map(), metadata = { blobs: [] }, warnings = new Set();
    const layerNames = new Set(entry.layers.map(l => l.name));
    let fileBytes = 0;
    for (const layer of entry.layers.filter(l => l.kind === 'tensor')) {
      cancelled(options.signal);
      let shard = shards.get(layer.path);
      if (!shard) { shard = await inspectFile(layer.path, { ...options, format: 'safetensors' }); shards.set(layer.path, shard); fileBytes = safeInteger(fileBytes + shard.fileBytes, 'Ollama weight bytes'); }
      metadata.blobs.push({ name: layer.name, digest: layer.digest, metadata: shard.metadata });
      for (const tensor of shard.tensors) {
        const name = tensor.name === 'data' && typeof layer.name === 'string' && layer.name ? layer.name : tensor.name;
        const storageKey = `${layer.path}:${tensor.storage.offset}`;
        if (names.has(name)) { if (names.get(name) === storageKey) continue; throw new Error(`Duplicate Ollama tensor name: ${name}`); } names.set(name, storageKey);
        const t = { ...tensor, name, storage: { ...tensor.storage, file: shard.path, identity: shard.identity } };
        // Blob-wide quantization metadata describes the packed weights, not
        // their scale bytes or ordinary floating-point tensors in the blob.
        const perTensorScheme = shard.metadata[`${tensor.name}.quant_type`] ?? shard.metadata[`${name}.quant_type`];
        const explicitScheme = perTensorScheme ?? shard.metadata.quant_type;
        const packed = tensor.dtype === 'U32' && ['.scale', '_scale', '.bias', '_qbias'].some(suffix => layerNames.has(name + suffix) || shard.byName.has(tensor.name + suffix));
        if (typeof perTensorScheme === 'string' || (tensor.dtype === 'U32' && (typeof explicitScheme === 'string' || packed))) {
          schemes.set(name, { scheme: typeof explicitScheme === 'string' ? explicitScheme : source.quantization || 'packed integer',
            groupSize: shard.metadata[`${tensor.name}.group_size`] ?? shard.metadata[`${name}.group_size`] ?? shard.metadata.group_size ?? 32 });
        }
        tensors.push(t);
      }
    }
    const byName = new Map(tensors.map(t => [t.name, t]));
    for (const [name, { scheme, groupSize }] of schemes) {
      const t = byName.get(name);
      t.quantization = { scheme, decoded: false };
      let reason = `Numerical decoding for Ollama ${scheme} is unavailable. Shapes and counts describe stored codes and scales, not dequantized weights.`;
      if (scheme.trim().toLowerCase() === 'mxfp8') {
        const scale = byName.get(`${name}.scale`) || byName.get(`${name}_scale`);
        try {
          Object.assign(t, mxfp8.layout(t, scale, groupSize, byName.has(`${name}.bias`) || byName.has(`${name}_qbias`)));
          t.storage.kind = 'ollama-mxfp8'; t.storage.scale = { ...scale.storage };
          scale.valueBasis = 'Stored MXFP8 E8M0 scale codes (not weight values)';
          continue;
        } catch (e) { reason = `Numerical decoding for Ollama MXFP8 is unavailable: ${e.message} Shapes and counts describe stored codes, not dequantized weights.`; }
      }
      t.valueBasis = `Packed Ollama ${scheme} storage; numerical decoding unavailable`;
      t.storage.kind = 'unsupported'; t.storage.reason = reason;
      t.analysisUnavailable = reason; warnings.add(reason);
    }
    model = { format: 'safetensors', tensors, metadata, warnings: [...warnings], path: file, identity: dependencies[0].identity, fileBytes };
    model.summary = summarize(model); model.byName = new Map(tensors.map(t => [t.name, t]));
  }
  model.source = source;
  model.protectedPaths = ollama.protectedPaths(entry); model.dependencies = dependencies;
  model.warnings.push(...(model.source.warnings || []));
  if (model.source.extraWeights.length) model.warnings.push('Adapter/projector blobs are listed in Model details; totals and analysis cover the primary weights only.');
  await assertCurrent(model);
  return model;
}
async function inspectSharded(file, set, options) {
  const missing = await sharded.missingFiles(set.directory, set.files);
  if (missing.length) throw new Error(`${missing.length} of ${set.files.length} shards listed in ${set.indexPath ? path.basename(set.indexPath) : 'this set'} are missing, including ${missing[0]}. The download may be incomplete.`);
  const tensors = [], shards = [], stored = new Map(), warnings = [...set.warnings];
  const dependencies = set.indexPath ? [{ path: set.indexPath, identity: identity(await fs.stat(set.indexPath)) }] : [];
  let fileBytes = 0, unindexed = 0;
  for (const name of set.files) {
    cancelled(options.signal);
    const shard = await inspectFile(path.join(set.directory, name), { ...options, format: 'safetensors' });
    fileBytes = safeInteger(fileBytes + shard.fileBytes, 'Total shard bytes');
    shards.push({ file: name, path: shard.path, bytes: shard.fileBytes, tensors: shard.tensors.length, metadata: shard.metadata });
    for (const tensor of shard.tensors) {
      if (stored.has(tensor.name)) throw new Error(`Tensor ${tensor.name} is stored in both ${stored.get(tensor.name)} and ${name}.`);
      stored.set(tensor.name, name);
      const listed = set.weightMap?.get(tensor.name);
      if (set.weightMap && listed === undefined) unindexed++;
      else if (listed !== undefined && listed !== name) throw new Error(`The index maps ${tensor.name} to ${listed}, but it is stored in ${name}.`);
      // Each tensor records its own shard, so analysis reads and change checks touch only that file.
      tensors.push({ ...tensor, shard: name, storage: { ...tensor.storage, file: shard.path, identity: shard.identity } });
    }
  }
  for (const [name, listed] of set.weightMap || []) if (!stored.has(name)) throw new Error(`The index lists ${name} in ${listed}, but that shard does not contain it.`);
  if (unindexed) warnings.push(`${unindexed} stored tensor(s) are missing from the index weight_map. They are shown here, but loaders that follow the index may ignore them.`);
  const model = { format: 'safetensors', tensors, warnings, path: file, identity: identity(await fs.stat(file)), fileBytes, dependencies,
    metadata: { index: set.indexPath ? set.metadata : null, shards: shards.map(({ file, metadata }) => ({ file, metadata })) },
    source: { kind: 'sharded', name: set.name, indexPath: set.indexPath, shards: shards.map(({ metadata, ...shard }) => shard) },
    protectedPaths: [...dependencies.map(d => d.path), ...shards.map(s => s.path)] };
  model.summary = summarize(model); model.byName = new Map(tensors.map(t => [t.name, t]));
  await assertCurrent(model);
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
  const tensor = model.byName.get(name);
  if (!tensor) throw new Error('Unknown tensor.');
  return sampleAt(model, name, sampleWindows(tensor.logicalElements ?? tensor.elements, options.sampleSize || 32768), options);
}
// Reads the GGUF values a translated pair's reference windows point at, in reference order.
// Reordered positions are sorted and coalesced so storage is still read in contiguous runs.
async function sampleMapped(model, windows, layout, options) {
  let values;
  if (!layout.heads && !layout.offset) values = await sampleAt(model, layout.tensor, windows, options);
  else {
    const positions = [];
    for (const w of windows) for (let i = w.start; i < w.start + w.count; i++) positions.push(sourceIndex(layout, i));
    const order = positions.map((_, i) => i).sort((a, b) => positions[a] - positions[b]), runs = [];
    for (const i of order) { const run = runs.at(-1); if (run && run.start + run.count === positions[i]) run.count++; else runs.push({ start: positions[i], count: 1 }); }
    const stored = await sampleAt(model, layout.tensor, runs, options);
    values = new Array(positions.length); order.forEach((i, k) => { values[i] = stored[k]; });
  }
  return layout.shift ? values.map(v => v - layout.shift) : values;
}
async function sampleAt(model, name, windows, options = {}) {
  cancelled(options.signal); await assertCurrent(model);
  const tensor = model.byName.get(name);
  if (!tensor) throw new Error('Unknown tensor.');
  let values;
  if (model.format === 'pytorch') {
    const result = await pythonRequest({ action: 'sample', path: model.path, name, windows }, options);
    values = result.map(v => v === 'NaN' ? NaN : v === 'Infinity' ? Infinity : v === '-Infinity' ? -Infinity : v);
  } else {
    const external = tensor.storage.kind === 'external' ? await externalStorage(model, tensor) : null;
    const current = external?.tensor || tensor;
    const storageFile = tensor.storage.file;
    const blob = model.source?.kind === 'sharded' ? `Shard ${tensor.shard}` : 'An Ollama weight blob';
    if (storageFile && identity(await fs.stat(storageFile)) !== tensor.storage.identity) throw new Error(`${blob} changed on disk. Refresh the inspector before analyzing it.`);
    const scale = tensor.storage.scale;
    if (scale && identity(await fs.stat(scale.file)) !== scale.identity) throw new Error('An Ollama scale blob changed on disk. Refresh the inspector before analyzing it.');
    const reader = await Reader.open(external?.file || storageFile || model.path, options.signal);
    let scaleReader;
    try {
      if (current.storage.kind === 'unsupported') throw new Error(current.storage.reason);
      if (current.storage.kind === 'ollama-mxfp8') {
        scaleReader = scale.file === storageFile ? reader : await Reader.open(scale.file, options.signal);
        values = await mxfp8.sample(reader, scaleReader, current, windows);
      } else values = current.storage.kind === 'onnx-typed' ? await typedValues(reader, current, windows) : await rawValues(reader, current, windows);
      if (external && identity(await fs.stat(external.file)) !== external.identity) throw new Error('External ONNX weights changed during analysis. Retry the operation.');
      if (storageFile && identity(await fs.stat(storageFile)) !== tensor.storage.identity) throw new Error(`${blob} changed during analysis. Refresh the inspector.`);
      if (scale && identity(await fs.stat(scale.file)) !== scale.identity) throw new Error('An Ollama scale blob changed during analysis. Refresh the inspector.');
    } finally { if (scaleReader && scaleReader !== reader) await scaleReader.close(); await reader.close(); }
  }
  await assertCurrent(model); cancelled(options.signal);
  return values;
}
async function tensorStatistics(model, name, options) {
  const tensor = model.byName.get(name);
  return statistics(await sample(model, name, options), tensor.logicalElements ?? tensor.elements, tensor.valueBasis || 'Weight values');
}
async function compareTensor(base, target, name, options = {}) {
  const pair = pairTensors(base, target).byName.get(name);
  if (!pair || structureRow(pair).status !== 'unverified') throw new Error('This tensor has incompatible shapes or numerical units, or is missing from a checkpoint.');
  const { layout } = pair;
  if (layout?.error) throw new Error(layout.error);
  // The non-GGUF tensor defines sample positions; the GGUF side is gathered to match them.
  const reference = layout?.side === 'left' ? pair.right : pair.left, total = reference.logicalElements ?? reference.elements;
  const windows = sampleWindows(total, options.sampleSize || 32768);
  const read = (model, tensor, side) => layout?.side === side ? sampleMapped(model, windows, layout, options) : sampleAt(model, tensor.name, windows, options);
  const left = await read(base, pair.left, 'left'), right = await read(target, pair.right, 'right');
  return difference(left, right, total, options.absoluteTolerance || 0);
}
function tensorMatching(base, target) { return pairTensors(base, target).matching; }
module.exports = { inspect, publicModel, sample, tensorStatistics, compareTensor, structureDiff, tensorMatching, pythonRequest, assertCurrent, identity };
