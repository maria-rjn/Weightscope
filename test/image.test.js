'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const backend = require('../src/backend');
const image = require('../src/image');
const F = require('./fixtures');
let dir;
before(async () => { dir = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-image-')); });
after(async () => { await fs.rm(dir, { recursive: true, force: true }); });

test('tensors are drawn as rows of leading dimensions and columns of trailing dimensions', () => {
  const shape = s => { const { rows, cols, split } = image.matrixShape(s); return [rows, cols, split]; };
  assert.deepEqual(shape([]), [1, 1, 0]); assert.deepEqual(shape([5]), [1, 5, 0]); assert.deepEqual(shape([4, 8]), [4, 8, 1]);
  assert.deepEqual(shape([16, 3, 3, 3]), [16, 27, 1]); assert.deepEqual(shape([8192, 1, 4]), [8192, 4, 1]); assert.deepEqual(shape([2, 768, 2048]), [1536, 2048, 2]);
});

test('small tensors are read in full; large ones read whole rows in evenly spaced runs that cover every tile row', () => {
  const exact = image.plan([64, 64], 10000);
  assert.equal(exact.exact, true); assert.deepEqual(exact.windows, [{ start: 0, count: 4096 }]); assert.deepEqual([exact.tileRows, exact.tileCols], [64, 64]);
  const sampled = image.plan([1000, 100], 10000), read = sampled.windows.reduce((n, w) => n + w.count, 0);
  assert.equal(sampled.exact, false); assert.equal(sampled.columnsSampled, false); assert.ok(read <= 10000); assert.equal(read, sampled.rowsRead * 100);
  for (const [i, w] of sampled.windows.entries()) {
    assert.equal(w.start % 100, 0); assert.equal(w.count % 100, 0); assert.ok(w.start + w.count <= 100000);
    if (i) assert.ok(w.start > sampled.windows[i - 1].start + sampled.windows[i - 1].count, 'windows are sorted, separate and coalesced');
  }
  const tiles = image.tiles(new Array(read).fill(1), sampled);
  assert.equal(tiles.unread, 0); assert.ok(tiles.tileRows <= sampled.windows.length);
  const wide = image.plan([2, 1000000], 1000);
  assert.equal(wide.columnsSampled, true); assert.equal(wide.tileRows, 1); assert.equal(wide.rowsRead, 1); assert.ok(wide.windows.reduce((n, w) => n + w.count, 0) <= 1000);
  assert.ok(image.tiles(new Array(wide.windows.reduce((n, w) => n + w.count, 0)).fill(0), wide).unread > 0, 'unread column gaps stay blank');
  assert.deepEqual(image.plan([0, 8], 100).windows, []);
  // Tall and wide matrices keep tiles roughly square along their long side.
  const tall = image.plan([262144, 2560], 4194304);
  assert.equal(tall.tileRows, image.MAX_TILES); assert.equal(tall.tileCols, 24);
});

test('each tile keeps its signed peak and RMS, and lists non-finite values', () => {
  const layout = { ...image.plan([4, 4], 100), tileRows: 2, tileCols: 2 };
  const values = [1, -3, 0, 0,
                  2, 0, 0, 5,
                  0, 0, NaN, 1,
                  0, 0, 2, 0];
  const t = image.tiles(values, layout, 'Weight values');
  assert.deepEqual(t.peak, [-3, 5, 0, 2]); assert.equal(t.rms[0], Number(Math.sqrt(14 / 4).toPrecision(5)));
  assert.deepEqual(t.nonFinite, [3]); assert.equal(t.rms[3], Number(Math.sqrt(5 / 3).toPrecision(5)));
  assert.deepEqual(t.largest, { value: 5, tile: 1 }); assert.equal(t.exact, true); assert.equal(t.valuesRead, 16);
  assert.throws(() => image.tiles([1, 2], layout), /read plan/);
  assert.deepEqual(image.delta([1, NaN, Infinity, 2, 0], [2, NaN, Infinity, -Infinity, 3]), [1, 0, 0, NaN, 3]);
});

test('weight and change images read real checkpoint values at matching positions', async () => {
  const values = Array.from({ length: 32 * 48 }, (_, i) => Math.fround(Math.sin(i) * (i % 48 === 7 ? 9 : 1)));
  const base = path.join(dir, 'base.safetensors'), target = path.join(dir, 'target.safetensors');
  await F.safetensors(base, [{ name: 'layer.weight', shape: [32, 48], values }]);
  await F.safetensors(target, [{ name: 'layer.weight', shape: [32, 48], values: values.map((v, i) => i === 5 * 48 + 20 ? v + .5 : v) }]);
  const a = await backend.inspect(base), b = await backend.inspect(target);
  const picture = await backend.tensorImage(a, 'layer.weight');
  assert.equal(picture.exact, true); assert.deepEqual([picture.tileRows, picture.tileCols], [32, 48]);
  assert.equal(picture.peak[3 * 48 + 11], Number(values[3 * 48 + 11].toPrecision(5)));
  assert.equal(picture.largest.tile % 48, 7, 'the outlier column holds the largest value');
  const change = await backend.differenceImage(a, b, 'layer.weight');
  assert.equal(change.basis, 'Change (target − baseline)'); assert.equal(change.largest.tile, 5 * 48 + 20);
  assert.equal(change.peak.filter(v => v !== 0).length, 1);
});
