'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
function files(dir) { return fs.readdirSync(dir, { withFileTypes: true }).flatMap(e => e.isDirectory() ? files(path.join(dir, e.name)) : [path.join(dir, e.name)]); }
for (const file of ['src', 'media', 'scripts', 'test'].flatMap(files).filter(f => f.endsWith('.js'))) {
  const result = spawnSync(process.execPath, ['--check', file], { encoding: 'utf8' });
  if (result.status) { process.stderr.write(result.stderr); process.exit(1); }
}
const manifest = JSON.parse(fs.readFileSync('package.json', 'utf8'));
for (const required of ['name', 'version', 'publisher', 'main', 'engines', 'contributes']) if (!manifest[required]) throw new Error(`Missing manifest field ${required}`);
for (const file of [manifest.main, 'python/inspect_pt.py', 'media/app.js', 'media/model-view.js', 'media/styles.css', 'README.md', 'LICENSE']) if (!fs.existsSync(file)) throw new Error(`Missing packaged file: ${file}`);
console.log('JavaScript syntax, extension manifest and required assets passed.');
