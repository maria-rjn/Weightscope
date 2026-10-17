'use strict';
// Captures the README screenshots from the browser preview with headless Chrome, driven over the
// DevTools protocol (Node.js 22+ WebSocket, no dependencies). Use real checkpoints, for example:
//   WEIGHTSCOPE_SHOT_MODEL=<checkpoint for Explore> \
//   WEIGHTSCOPE_PREVIEW_BASE=<baseline> WEIGHTSCOPE_PREVIEW_TARGET=<target> npm run screenshots
// WEIGHTSCOPE_SHOT_TENSOR names the tensor for the weight image (default: the largest matrix).
// CHROME_PATH overrides the browser. Without checkpoints, the generated demo models are used.
const { spawn } = require('node:child_process');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const backend = require('../src/backend');

const OUT = path.join(__dirname, '..', 'docs', 'images'), WIDTH = 1280, HEIGHT = 900, SCALE = 2;
const CHROME = process.env.CHROME_PATH || { darwin: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', win32: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe' }[process.platform] || 'google-chrome';

// Starts a process and resolves once a line of its output matches.
function launch(command, args, env, pattern) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { env: { ...process.env, ...env }, stdio: ['ignore', 'pipe', 'pipe'] });
    let output = '';
    const timer = setTimeout(() => { child.kill(); reject(new Error(`${path.basename(command)} did not start:\n${output.slice(-2000)}`)); }, 120000);
    const read = chunk => { output += chunk; const match = output.match(pattern); if (match) { clearTimeout(timer); resolve({ child, match }); } };
    child.stdout.on('data', read); child.stderr.on('data', read);
    child.on('error', e => { clearTimeout(timer); reject(e); });
    child.on('exit', code => { clearTimeout(timer); reject(new Error(`${path.basename(command)} exited with code ${code}:\n${output.slice(-2000)}`)); });
  });
}
function devtools(url) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url), pending = new Map(); let id = 0;
    socket.onopen = () => resolve({
      send: (method, params = {}) => new Promise((ok, fail) => { pending.set(++id, { ok, fail }); socket.send(JSON.stringify({ id, method, params })); }),
      close: () => socket.close()
    });
    socket.onerror = () => reject(new Error('Cannot connect to Chrome DevTools.'));
    socket.onmessage = event => { const m = JSON.parse(event.data), p = pending.get(m.id); if (!p) return; pending.delete(m.id); if (m.error) p.fail(new Error(m.error.message)); else p.ok(m.result); };
  });
}
// Runs an async snippet in the page with small waiting helpers and returns its value.
const PRELUDE = `const sleep = ms => new Promise(r => setTimeout(r, ms));
  const until = async (test, label, ms = 600000) => { const t = Date.now(); while (!test()) { if (Date.now() - t > ms) throw new Error('Timed out waiting for ' + label); await sleep(100); } };
  const choose = (selector, value) => { const n = document.querySelector(selector); n.value = value; n.dispatchEvent(new Event('change')); };
  const box = (n, pad = 0) => { const r = n.getBoundingClientRect(); return { x: 0, y: Math.max(0, r.top + scrollY - pad), width: ${WIDTH}, height: r.height + pad * 2 }; };`;
async function run(page, code) {
  const r = await page.send('Runtime.evaluate', { expression: `(async () => { ${PRELUDE}\n${code} })()`, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
}
async function shot(page, name, clip) {
  const { data } = await page.send('Page.captureScreenshot', { format: 'jpeg', quality: 86, captureBeyondViewport: Boolean(clip), ...(clip && { clip: { ...clip, scale: 1 } }) });
  await fs.writeFile(path.join(OUT, name), Buffer.from(data, 'base64')); console.log(`Saved docs/images/${name}`);
}
async function preview(env, port, steps) {
  const server = await launch(process.execPath, [path.join(__dirname, 'preview.js')], { ...env, WEIGHTSCOPE_PREVIEW_PORT: String(port) }, /WeightScope preview: (\S+)/);
  const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-shots-'));
  let chrome, page;
  try {
    chrome = await launch(CHROME, ['--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check', '--hide-scrollbars', '--force-color-profile=srgb', 'about:blank'], {}, /DevTools listening on ws:\/\/[^:]+:(\d+)\//);
    const targets = await (await fetch(`http://127.0.0.1:${chrome.match[1]}/json/list`)).json();
    page = await devtools(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
    await page.send('Page.enable'); await page.send('Runtime.enable');
    await page.send('Emulation.setDeviceMetricsOverride', { width: WIDTH, height: HEIGHT, deviceScaleFactor: SCALE, mobile: false });
    await page.send('Page.navigate', { url: server.match[1] });
    await run(page, `await until(() => document.querySelector('.hero, .loading h1') && document.readyState === 'complete', 'the inspector');`);
    await steps(page);
  } finally { page?.close(); chrome?.child.kill(); server.child.kill(); await fs.rm(profile, { recursive: true, force: true }).catch(() => {}); }
}
async function main() {
  const demo = await require('./make-demo').makeDemo(), base = process.env.WEIGHTSCOPE_PREVIEW_BASE || demo.base, target = process.env.WEIGHTSCOPE_PREVIEW_TARGET || demo.target;
  const model = process.env.WEIGHTSCOPE_SHOT_MODEL || base, port = Number(process.env.WEIGHTSCOPE_SHOT_PORT || 4180);
  const tensor = process.env.WEIGHTSCOPE_SHOT_TENSOR || (await backend.inspect(model)).tensors.filter(t => (t.logicalShape || t.shape).length >= 2 && !t.analysisUnavailable).sort((a, b) => (b.logicalElements ?? b.elements) - (a.logicalElements ?? a.elements))[0]?.name;
  await fs.mkdir(OUT, { recursive: true });
  await preview({ WEIGHTSCOPE_PREVIEW_BASE: model, WEIGHTSCOPE_PREVIEW_TARGET: target }, port, async page => {
    await shot(page, 'model-map.jpg', await run(page, `await until(() => document.querySelector('#model-map canvas'), 'the model map'); choose('[data-matrix-key=model-color]', 'precision'); await sleep(300);
      return box(document.querySelector('#model-map')); `).then(b => ({ ...b, height: b.y + b.height + 24, y: 0 })));
    await shot(page, 'model-map-rms.jpg', await run(page, `document.querySelector('[data-action=scan]').click(); await until(() => !document.querySelector('[data-action=scan]') || document.querySelector('[data-action=scan]').dataset.inactive === 'true', 'the weight scan');
      choose('[data-matrix-key=model-color]', 'rms'); await sleep(300); return box(document.querySelector('#model-map'), 16);`));
    if (!tensor) return;
    await run(page, `const search = document.querySelector('input[type=search]'); search.value = ${JSON.stringify(tensor)}; search.dispatchEvent(new Event('input')); await sleep(200);
      [...document.querySelectorAll('.tensor-link')].find(b => b.title === ${JSON.stringify(tensor)}).click(); await until(() => document.querySelector('[data-action=image]'), 'the tensor inspector');
      document.querySelector('[data-action=image]').click(); await until(() => document.querySelector('.weight-image canvas'), 'the weight image'); choose('[aria-label="Image statistic"]', 'rms'); await sleep(300); scrollTo(0, 0);`);
    await shot(page, 'weight-image.jpg');
  });
  await preview({ WEIGHTSCOPE_PREVIEW_BASE: base, WEIGHTSCOPE_PREVIEW_TARGET: target }, port + 1, async page => {
    await shot(page, 'drift-fingerprint.jpg', await run(page, `document.querySelector('[data-action=compare]').click(); await until(() => document.getElementById('analyze-comparison'), 'the comparison');
      const analyze = document.getElementById('analyze-comparison'); if (!analyze.disabled) analyze.click();
      await until(() => /complete|Retry|No comparable/.test(document.getElementById('analyze-comparison')?.textContent), 'the comparison analysis'); await sleep(500);
      choose('[data-matrix-key=map-view]', 'fingerprint'); choose('[data-matrix-key=map-scale]', 'log'); await sleep(300);
      const pane = document.querySelector('.compare-pane'), legend = document.querySelector('#change-map .heatmap-legend'), top = pane.getBoundingClientRect().top + scrollY;
      return { x: 0, y: top, width: ${WIDTH}, height: legend.getBoundingClientRect().bottom + scrollY - top + 48 };`));
  });
}
main().catch(e => { console.error(e.message); process.exitCode = 1; });
