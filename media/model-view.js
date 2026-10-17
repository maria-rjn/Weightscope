'use strict';
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.WeightScopeView = api;
})(globalThis, function () {
  function groupKey(name) {
    const parts = name.split(/[./]/).filter(Boolean); parts.pop();
    const index = parts.findIndex(p => /^\d+$/.test(p));
    return (index < 0 ? parts : parts.slice(0, index + 1)).join('.') || 'Ungrouped tensors';
  }
  function groups(tensors) {
    const map = new Map();
    for (const t of tensors) {
      const key = groupKey(t.name);
      if (!map.has(key)) map.set(key, { key, tensors: [], elements: 0, bytes: 0, unknownBytes: false, dtypes: new Set() });
      const g = map.get(key); g.tensors.push(t); g.elements += t.elements;
      if (t.bytes == null) g.unknownBytes = true; else g.bytes += t.bytes;
      g.dtypes.add(t.dtype);
    }
    return [...map.values()].sort((a, b) => a.key.localeCompare(b.key, undefined, { numeric: true }));
  }
  const CHANGES = new Set(['changed', 'added', 'removed', 'shape-changed', 'basis-changed']);
  const hasChange = row => CHANGES.has(row.status) || Boolean(row.dtypeChanged);
  const needsAttention = row => hasChange(row) || ['unverified', 'unavailable'].includes(row.status);
  function diffOverview(rows) {
    const counts = Object.create(null), map = new Map();
    let analyzed = 0, changes = 0, pending = 0, unavailable = 0;
    for (const row of rows) {
      counts[row.status] = (counts[row.status] || 0) + 1;
      if (row.metrics) analyzed++; if (hasChange(row)) changes++;
      if (row.status === 'unverified') pending++; if (row.status === 'unavailable') unavailable++;
      const key = groupKey(row.name);
      if (!map.has(key)) map.set(key, { key, rows: [], changes: 0, pending: 0, unavailable: 0, sampleMatches: 0, maxAbs: null });
      const g = map.get(key); g.rows.push(row);
      if (hasChange(row)) g.changes++; if (row.status === 'unverified') g.pending++;
      if (row.status === 'unavailable') g.unavailable++; if (row.status === 'sample-match') g.sampleMatches++;
      if (row.metrics?.maxAbs != null) g.maxAbs = Math.max(g.maxAbs ?? 0, row.metrics.maxAbs);
    }
    const grouped = [...map.values()].sort((a, b) => a.key.localeCompare(b.key, undefined, { numeric: true }));
    for (const g of grouped) g.status = g.rows.every(r => r.status === 'added') ? 'added' : g.rows.every(r => r.status === 'removed') ? 'removed' : g.changes ? 'changed' : g.unavailable ? 'unavailable' : g.pending ? 'unverified' : g.sampleMatches ? 'sample-match' : 'equal';
    return { counts, groups: grouped, analyzed, changes, pending, unavailable, total: rows.length };
  }
  function tensorRole(name) {
    const parts = name.split(/[./]/).filter(Boolean), key = groupKey(name);
    return parts.slice(key === 'Ungrouped tensors' ? 0 : key.split('.').length).join('.') || name;
  }
  function comparisonMatrix(rows) {
    const map = new Map(), roles = new Set();
    let maxRelativeL2 = 0, minRelativeL2 = Infinity;
    for (const row of rows) {
      const key = groupKey(row.name), role = tensorRole(row.name);
      if (!map.has(key)) map.set(key, { key, cells: new Map() });
      const cells = map.get(key).cells;
      if (!cells.has(role)) cells.set(role, []);
      cells.get(role).push(row); roles.add(role);
      const relative = row.metrics?.relativeL2;
      if (Number.isFinite(relative)) { maxRelativeL2 = Math.max(maxRelativeL2, relative); if (relative > 0) minRelativeL2 = Math.min(minRelativeL2, relative); }
    }
    const natural = (a, b) => a.localeCompare(b, undefined, { numeric: true });
    return { groups: [...map.values()].sort((a, b) => natural(a.key, b.key)), roles: [...roles].sort(natural), maxRelativeL2, minRelativeL2: minRelativeL2 === Infinity ? 0 : minRelativeL2 };
  }
  // Position of a relative L2 value on the shared map scale, from 0 to 1, or null when it has none.
  // The log scale spans the smallest to the largest positive drift (at most six decades), so small
  // changes stay visible next to one large outlier. Zero drift is position 0 on both scales.
  function driftScale(matrix, mode = 'log') {
    const max = matrix.maxRelativeL2, low = Math.max(matrix.minRelativeL2, max * 1e-6), log = mode === 'log' && max > 0 && low > 0 && low < max;
    const position = v => !Number.isFinite(v) || !(max > 0) ? null : v <= 0 ? 0 : Math.min(1, log ? Math.max(0, Math.log(v / low) / Math.log(max / low)) : v / max);
    return { mode: log ? 'log' : 'linear', position, ticks: !(max > 0) ? [] : log ? [low, Math.sqrt(low * max), max] : [0, max / 2, max] };
  }
  // Magma, perceptually uniform. Light themes run it from pale to dark so more drift always means more contrast with the page.
  const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const RAMPS = { dark: ['#3b0f70', '#641a80', '#8c2981', '#b73779', '#de4968', '#f7705c', '#fe9f6d', '#fecf92', '#fcfdbf'].map(hex),
    light: ['#fecf92', '#fe9f6d', '#f7705c', '#de4968', '#b73779', '#8c2981', '#641a80', '#3b0f70', '#140e36'].map(hex) };
  function heatRgb(t, light = false) {
    const ramp = RAMPS[light ? 'light' : 'dark'], p = Math.max(0, Math.min(1, t)) * (ramp.length - 1), i = Math.min(ramp.length - 2, Math.floor(p)), f = p - i;
    return ramp[i].map((c, k) => Math.round(c + (ramp[i + 1][k] - c) * f));
  }
  // Signed values: blue below zero, red above. The center matches the theme, so near-zero weights recede into the page.
  const DIVERGING = { dark: ['#b9dcff', '#5aa4e8', '#1f5c94', '#14171c', '#8c2f1c', '#e0663f', '#ffc6a8'].map(hex),
    light: ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b'].map(hex) };
  function divergingRgb(t, light = false) {
    const ramp = DIVERGING[light ? 'light' : 'dark'], p = (Math.max(-1, Math.min(1, t)) + 1) / 2 * (ramp.length - 1), i = Math.min(ramp.length - 2, Math.floor(p)), f = p - i;
    return ramp[i].map((c, k) => Math.round(c + (ramp[i + 1][k] - c) * f));
  }
  const heatColor = (t, light) => `rgb(${heatRgb(t, light).join(', ')})`;
  const heatInk = (t, light) => { const [r, g, b] = heatRgb(t, light); return .2126 * r + .7152 * g + .0722 * b > 140 ? 'dark' : 'light'; };
  // Translated cross-format rows also answer to the tensor name in the other checkpoint.
  const matchesQuery = (row, query) => [row.name, row.baseName, row.targetName].some(n => n && n.toLowerCase().includes(query.toLowerCase()));
  function diffRows(rows, { query = '', status = 'attention', group = '', role = '' } = {}) {
    return rows.filter(r => matchesQuery(r, query) && (!group || groupKey(r.name) === group)
      && (!role || tensorRole(r.name) === role)
      && (!status || (status === 'attention' ? needsAttention(r) : r.status === status)))
      .sort((a, b) => Number(hasChange(b)) - Number(hasChange(a)) || (b.metrics?.maxAbs || 0) - (a.metrics?.maxAbs || 0) || a.name.localeCompare(b.name, undefined, { numeric: true }));
  }
  return { groupKey, groups, tensorRole, comparisonMatrix, driftScale, heatRgb, heatColor, heatInk, divergingRgb, hasChange, needsAttention, diffOverview, diffRows, matchesQuery };
});
