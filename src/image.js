'use strict';
// Downsampled 2D images of tensors. Values are read in logical row-major order, so the trailing
// dimensions are contiguous and become image columns; the leading dimensions become rows.
const { sampleWindows } = require('./binary');

const MAX_TILES = 192, MIN_TILES = 24, MIN_RUN = 1024;

// Trailing dimensions join the columns until a row has at least 32 values, so convolution
// kernels [out, in, kh, kw] draw as out × (in·kh·kw) rather than as thin strips.
function matrixShape(shape) {
  if (!shape.length) return { rows: 1, cols: 1, split: 0 };
  if (shape.length === 1) return { rows: 1, cols: shape[0], split: 0 };
  let split = shape.length - 1, cols = shape[split];
  while (split > 1 && cols < 32) cols *= shape[--split];
  return { rows: shape.slice(0, split).reduce((n, d) => n * d, 1), cols, split };
}
function tileCounts(rows, cols) {
  // Tiles stay roughly square, so a wide or tall matrix spends its tiles along its long side.
  if (rows >= cols) return [Math.min(rows, MAX_TILES), Math.min(cols, Math.max(MIN_TILES, Math.round(MAX_TILES * cols / rows)))];
  return [Math.min(rows, Math.max(MIN_TILES, Math.round(MAX_TILES * rows / cols))), Math.min(cols, MAX_TILES)];
}
// Chooses which values to read. Small tensors are read in full. Larger ones read evenly spaced
// runs of whole rows, so every column is covered; rows wider than the budget use sampled windows.
function plan(shape, budget) {
  const { rows, cols, split } = matrixShape(shape), total = rows * cols;
  if (!total) return { shape, rows, cols, split, total, windows: [], tileRows: 0, tileCols: 0, rowsRead: 0, exact: true, columnsSampled: false };
  const perRow = cols <= budget ? [{ start: 0, count: cols }] : sampleWindows(cols, budget), kept = perRow.reduce((n, w) => n + w.count, 0);
  const capacity = Math.max(1, Math.floor(budget / kept)), exact = capacity >= rows && kept === cols;
  let [tileRows, tileCols] = tileCounts(rows, cols), windows, rowsRead;
  if (exact) { windows = [{ start: 0, count: total }]; rowsRead = rows; }
  else {
    // A run of consecutive rows per stratum keeps reads at least MIN_RUN values long.
    const run = Math.min(capacity, Math.max(1, Math.ceil(MIN_RUN / kept))), strata = Math.max(1, Math.min(rows, Math.floor(capacity / run)));
    tileRows = Math.min(tileRows, strata); windows = []; rowsRead = 0;
    for (let i = 0; i < strata; i++) {
      const first = Math.max(0, Math.min(rows - run, Math.floor((i + .5) * rows / strata - run / 2)));
      const previous = windows.at(-1);
      for (let r = Math.max(first, previous ? Math.floor((previous.start + previous.count - 1) / cols) + 1 : 0); r < first + run; r++) {
        rowsRead++;
        for (const w of perRow) { const start = r * cols + w.start, last = windows.at(-1); if (last && last.start + last.count === start) last.count += w.count; else windows.push({ start, count: w.count }); }
      }
    }
  }
  return { shape, rows, cols, split, total, windows, tileRows, tileCols, rowsRead, exact, columnsSampled: kept < cols };
}
const round = v => Number(v.toPrecision(5));
// Reduces values read at plan.windows to one signed peak (the value with the largest magnitude)
// and one RMS per tile. Tiles with no values read are null; tiles containing NaN or infinity are listed.
function tiles(values, layout, basis = 'Weight values') {
  const { rows, cols, tileRows, tileCols } = layout, count = tileRows * tileCols;
  const peak = new Float64Array(count), squares = new Float64Array(count), finite = new Uint32Array(count), read = new Uint8Array(count), invalid = new Uint8Array(count);
  let k = 0, largest = null;
  for (const w of layout.windows) for (let index = w.start; index < w.start + w.count; index++, k++) {
    const v = values[k], r = Math.floor(index / cols), c = index - r * cols, t = Math.floor(r * tileRows / rows) * tileCols + Math.floor(c * tileCols / cols);
    read[t] = 1;
    if (!Number.isFinite(v)) { invalid[t] = 1; continue; }
    finite[t]++; squares[t] += v * v;
    if (Math.abs(v) > Math.abs(peak[t]) || finite[t] === 1) peak[t] = v;
    if (largest === null || Math.abs(v) > Math.abs(largest.value)) largest = { value: v, tile: t };
  }
  if (k !== values.length) throw new Error('Image sample does not match its read plan.');
  const tile = i => read[i] && finite[i] ? i : null;
  return { shape: layout.shape, rows, cols, split: layout.split, tileRows, tileCols, rowsRead: layout.rowsRead, valuesRead: values.length, total: layout.total,
    exact: layout.exact, columnsSampled: layout.columnsSampled, basis,
    peak: Array.from(peak, (v, i) => tile(i) === null ? null : round(v)), rms: Array.from(squares, (v, i) => tile(i) === null ? null : round(Math.sqrt(v / finite[i]))),
    nonFinite: [...invalid.keys()].filter(i => invalid[i]), unread: [...read.keys()].filter(i => !read[i]).length,
    largest: largest && { value: round(largest.value), tile: largest.tile } };
}
// Target minus baseline. Matching non-finite pairs agree, as in numerical comparison; mismatches stay non-finite.
function delta(left, right) {
  return right.map((y, i) => { const x = left[i]; return Number.isFinite(x) && Number.isFinite(y) ? y - x : x === y || (Number.isNaN(x) && Number.isNaN(y)) ? 0 : NaN; });
}
module.exports = { matrixShape, plan, tiles, delta, MAX_TILES };
