'use strict';
const { isDeepStrictEqual } = require('node:util');
function statistics(values, total, basis = 'Weight values') {
  let finite = 0, nan = 0, inf = 0, zero = 0, min = Infinity, max = -Infinity, mean = 0, m2 = 0, squareSum = 0;
  for (const v of values) {
    if (Number.isNaN(v)) { nan++; continue; }
    if (!Number.isFinite(v)) { inf++; continue; }
    finite++; if (v === 0) zero++;
    min = Math.min(min, v); max = Math.max(max, v);
    const delta = v - mean; mean += delta / finite; m2 += delta * (v - mean); squareSum += v * v;
  }
  const bins = Array(24).fill(0);
  if (finite) for (const v of values) if (Number.isFinite(v)) bins[min === max ? 12 : Math.min(23, Math.floor((v - min) / (max - min) * 24))]++;
  return { total, sampled: values.length, exact: values.length === total, finite, nan, inf, zero,
    min: finite ? min : null, max: finite ? max : null, mean: finite ? mean : null,
    std: finite ? Math.sqrt(m2 / finite) : null, rms: finite ? Math.sqrt(squareSum / finite) : null,
    zeroFraction: finite ? zero / finite : null, histogram: bins, basis };
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
  return { total, sampled: a.length, exact, finite, nonFinite, nonFiniteMismatch, changedValues: changed, tolerance,
    status: changed || nonFiniteMismatch ? 'changed' : exact ? 'equal' : 'sample-match',
    meanAbs: finite ? absSum / finite : null, maxAbs: finite ? maxAbs : null,
    rmse: finite ? Math.sqrt(squareSum / finite) : null,
    relativeL2: normA > 0 ? Math.sqrt(squareSum / normA) : null,
    cosine: normA > 0 && normB > 0 ? Math.max(-1, Math.min(1, dot / Math.sqrt(normA * normB))) : null };
}
function structureDiff(base, target) {
  const shape = t => t.logicalShape || t.shape;
  const info = t => ({ shape: shape(t), dtype: t.quantization?.decodedDtype ? t.quantization.scheme.toUpperCase() : t.dtype, elements: t.logicalElements ?? t.elements });
  const a = new Map(base.tensors.map(t => [t.name, t])), b = new Map(target.tensors.map(t => [t.name, t]));
  return [...new Set([...a.keys(), ...b.keys()])].sort((x, y) => x.localeCompare(y, undefined, { numeric: true })).map(name => {
    const left = a.get(name), right = b.get(name);
    const shapeChanged = left && right && JSON.stringify(shape(left)) !== JSON.stringify(shape(right));
    const dtypeChanged = left && right && info(left).dtype !== info(right).dtype;
    const basisChanged = left && right && (left.valueBasis || 'Weight values') !== (right.valueBasis || 'Weight values');
    return { name, left: left ? info(left) : null,
      right: right ? info(right) : null,
      dtypeChanged: Boolean(dtypeChanged), status: !left ? 'added' : !right ? 'removed' : shapeChanged ? 'shape-changed' : basisChanged ? 'basis-changed' : 'unverified' };
  });
}
function metadataDiff(base, target) {
  const left = base.metadata || {}, right = target.metadata || {};
  return [...new Set([...Object.keys(left), ...Object.keys(right)])].sort().flatMap(key => {
    const hasLeft = Object.hasOwn(left, key), hasRight = Object.hasOwn(right, key);
    if (hasLeft && hasRight && isDeepStrictEqual(left[key], right[key])) return [];
    return [{ key, status: !hasLeft ? 'added' : !hasRight ? 'removed' : 'changed', left: hasLeft ? left[key] : null, right: hasRight ? right[key] : null }];
  });
}
module.exports = { statistics, difference, structureDiff, metadataDiff };
