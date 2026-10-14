'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const F = require('./fixtures');
async function store(root) {
  await fs.mkdir(path.join(root, 'blobs'), { recursive: true });
  await fs.mkdir(path.join(root, 'manifests'), { recursive: true });
  async function blob(kind, contents, options = {}) {
    const temporary = path.join(root, 'temporary');
    if (kind === 'model') await F.gguf(temporary, contents);
    else if (kind === 'tensor') await F.safetensors(temporary, contents, options.metadata || {});
    else await fs.writeFile(temporary, typeof contents === 'string' ? contents : JSON.stringify(contents));
    const data = await fs.readFile(temporary), hash = crypto.createHash('sha256').update(data).digest('hex');
    const file = path.join(root, 'blobs', `sha256-${hash}`); await fs.rename(temporary, file);
    return { mediaType: `application/vnd.ollama.image.${kind}`, digest: `sha256:${hash}`, size: data.length, ...(options.name ? { name: options.name } : {}) };
  }
  async function manifest(model, tag, layers, fields = {}, namespace = 'library') {
    const dir = path.join(root, 'manifests', 'registry.ollama.ai', namespace, model); await fs.mkdir(dir, { recursive: true });
    const file = path.join(dir, tag); await fs.writeFile(file, JSON.stringify({ schemaVersion: 2, layers, ...fields })); return file;
  }
  return { root, blob, manifest };
}
module.exports = { store };
