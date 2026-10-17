'use strict';
const { pairTensors } = require('./tensor-names');
function histogramDomain(...series) {
  let min = Infinity, max = -Infinity;
  for (const values of series) for (const value of values) if (Number.isFinite(value)) { min = Math.min(min, value); max = Math.max(max, value); }
  if (min === Infinity) return { min: null, max: null };
  if (min === max) {
    const padding = Math.abs(min) * .05 || .5;
    if (Number.isFinite(min - padding)) min -= padding;
    if (Number.isFinite(max + padding)) max += padding;
  }
  return { min, max };
}
function histogram(values, domain, count = 24) {
  const bins = Array(count).fill(0);
  if (domain.min == null || domain.max == null) return bins;
  const range = domain.max - domain.min;
  for (const value of values) if (Number.isFinite(value)) {
    const position = Number.isFinite(range) ? (value - domain.min) / range : (value / 2 - domain.min / 2) / (domain.max / 2 - domain.min / 2);
    bins[Math.max(0, Math.min(count - 1, Math.floor(position * count)))]++;
  }
  return bins;
}
function statistics(values, total, basis = 'Weight values') {
  let finite = 0, nan = 0, inf = 0, zero = 0, min = Infinity, max = -Infinity, mean = 0, m2 = 0, squareSum = 0;
  for (const v of values) {
    if (Number.isNaN(v)) { nan++; continue; }
    if (!Number.isFinite(v)) { inf++; continue; }
    finite++; if (v === 0) zero++;
    min = Math.min(min, v); max = Math.max(max, v);
    const delta = v - mean; mean += delta / finite; m2 += delta * (v - mean); squareSum += v * v;
  }
  const histogramRange = histogramDomain(values), bins = histogram(values, histogramRange);
  return { total, sampled: values.length, exact: values.length === total, finite, nan, inf, zero,
    min: finite ? min : null, max: finite ? max : null, mean: finite ? mean : null,
    std: finite ? Math.sqrt(m2 / finite) : null, rms: finite ? Math.sqrt(squareSum / finite) : null,
    zeroFraction: finite ? zero / finite : null, histogram: bins, histogramRange, basis };
}
function difference(a, b, total, tolerance = 0) {
  if (a.length !== b.length) throw new Error('Cannot compare different sample layouts.');
  let finite = 0, nonFinite = 0, nonFiniteMismatch = 0, changed = 0, absSum = 0, squareSum = 0, maxAbs = 0, normA = 0, normB = 0, dot = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) { nonFinite++; if (!(x === y || (Number.isNaN(x) && Number.isNaN(y)))) nonFiniteMismatch++; continue; }
    finite++; const delta = Math.abs(y - x);
    if (delta > tolerance) changed++;
    absSum += delta; squareSum += delta ** 2; maxAbs = Math.max(maxAbs, delta);
    normA += x ** 2; normB += y ** 2; dot += x * y;
  }
  const exact = a.length === total;
  const domain = histogramDomain(a, b);
  return { total, sampled: a.length, exact, finite, nonFinite, nonFiniteMismatch, changedValues: changed, tolerance,
    status: changed || nonFiniteMismatch ? 'changed' : exact ? 'equal' : 'sample-match',
    meanAbs: finite ? absSum / finite : null, maxAbs: finite ? maxAbs : null,
    rmse: finite ? Math.sqrt(squareSum / finite) : null,
    relativeL2: normA > 0 ? Math.sqrt(squareSum / normA) : null,
    cosine: normA > 0 && normB > 0 ? Math.max(-1, Math.min(1, dot / Math.sqrt(normA * normB))) : null,
    distributions: { ...domain, baseline: histogram(a, domain), target: histogram(b, domain) } };
}
function structureRow({ name, left, right, layout }) {
  const shape = t => t.logicalShape || t.shape;
  const info = t => ({ shape: shape(t), dtype: t.quantization?.decodedDtype ? t.quantization.scheme.toUpperCase() : t.dtype, elements: t.logicalElements ?? t.elements });
  const shapeChanged = left && right && JSON.stringify(shape(left)) !== JSON.stringify(shape(right));
  const dtypeChanged = left && right && info(left).dtype !== info(right).dtype;
  const basisChanged = left && right && (left.valueBasis || 'Weight values') !== (right.valueBasis || 'Weight values');
  return { name, ...(layout && { baseName: left.name, targetName: right.name, layout: layout.notes }), left: left ? info(left) : null,
    right: right ? info(right) : null,
    dtypeChanged: Boolean(dtypeChanged), status: !left ? 'added' : !right ? 'removed' : shapeChanged ? 'shape-changed' : basisChanged ? 'basis-changed' : 'unverified' };
}
function structureDiff(base, target) {
  return pairTensors(base, target).pairs.map(structureRow);
}
module.exports = { statistics, difference, structureRow, structureDiff };
