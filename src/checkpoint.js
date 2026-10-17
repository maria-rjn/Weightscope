'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const backend = require('./backend');
const { FORMATS } = require('./sharded');
const { Reader, cancelled } = require('./binary');

const EXTENSIONS = { safetensors: ['.safetensors'], gguf: ['.gguf'], onnx: ['.onnx'], pytorch: ['.pt', '.pth'] };
const CHUNK = 1024 * 1024;
function capabilities(model) {
  if (model.source?.kind === 'sharded') return { save: false, reason: `Sharded ${FORMATS[model.format].title} checkpoints cannot yet be saved as a copy. Copy the model folder to keep every shard and its index together.` };
  if (model.source?.kind === 'ollama' && model.format === 'safetensors') return { save: false, reason: 'Multi-blob Ollama models cannot yet be exported as one checkpoint. Open an individual safetensors file to edit it.' };
  return { save: Boolean(EXTENSIONS[model.format]), extension: EXTENSIONS[model.format]?.[0] };
}
async function copyRange(reader, output, start, length, transform) {
  for (let pos = 0; pos < length; pos += CHUNK) {
    const buffer = await reader.read(start + pos, Math.min(CHUNK, length - pos));
    if (transform) transform(buffer, pos);
    await writeAll(output, buffer);
  }
}
async function writeAll(output, buffer) {
  let pos = 0;
  while (pos < buffer.length) {
    const { bytesWritten } = await output.write(buffer, pos, buffer.length - pos);
    if (!bytesWritten) throw new Error('Could not write checkpoint data.');
    pos += bytesWritten;
  }
}
async function transaction(model, destination, options, write) {
  const support = capabilities(model);
  if (!support.save) throw new Error(support.reason || 'This checkpoint format cannot be saved.');
  if (!EXTENSIONS[model.format].includes(path.extname(destination).toLowerCase())) throw new Error(`Save this checkpoint with a ${support.extension} extension.`);
  cancelled(options.signal); await backend.assertCurrent(model);
  // ONNX references are relative to the model; a raw copy cannot relocate its dependencies.
  if (model.format === 'onnx' && model.tensors.some(t => t.storage.kind === 'external') &&
      await fs.realpath(path.dirname(destination)) !== await fs.realpath(path.dirname(model.path))) {
    throw new Error('Save ONNX models with external weights in the original model directory so their relative references remain valid.');
  }
  const temp = path.join(path.dirname(destination), `.weightscope-${crypto.randomUUID()}${path.extname(destination)}`);
  let reader, output;
  try {
    reader = await Reader.open(model.path, options.signal);
    if (backend.identity(await reader.handle.stat()) !== model.identity) throw new Error('The checkpoint changed on disk. Refresh before saving.');
    output = await fs.open(temp, 'wx', 0o600);
    await write(reader, output);
    cancelled(options.signal);
    await output.sync(); await output.close(); output = undefined;
    const verified = await backend.inspect(temp, options);
    if (verified.tensors.length !== model.tensors.length || model.tensors.some((t, i) => {
      const other = verified.tensors[i];
      return t.name !== other.name || t.dtype !== other.dtype || JSON.stringify(t.shape) !== JSON.stringify(other.shape);
    })) throw new Error('Saved checkpoint validation failed: tensor names, shapes, or dtypes changed.');
    await backend.assertCurrent(model); cancelled(options.signal);
    // Linking is atomic and exclusive: existing files, symlinks, and hard links are never overwritten.
    try { await fs.link(temp, destination); }
    catch (e) { if (e.code === 'EEXIST') throw new Error('Choose a new file name. Save As never overwrites an existing file.'); throw e; }
    return { path: destination };
  } finally {
    if (output) await output.close();
    if (reader) await reader.close();
    await fs.unlink(temp).catch(e => { if (e.code !== 'ENOENT') throw e; });
  }
}
async function saveCheckpoint(model, destination, options = {}) {
  return transaction(model, destination, options, (reader, output) => copyRange(reader, output, 0, reader.size));
}
module.exports = { saveCheckpoint, capabilities, transaction, copyRange, writeAll };
