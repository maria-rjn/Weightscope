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
    let maxRelativeL2 = 0;
    for (const row of rows) {
      const key = groupKey(row.name), role = tensorRole(row.name);
      if (!map.has(key)) map.set(key, { key, cells: new Map() });
      const cells = map.get(key).cells;
      if (!cells.has(role)) cells.set(role, []);
      cells.get(role).push(row); roles.add(role);
      if (Number.isFinite(row.metrics?.relativeL2)) maxRelativeL2 = Math.max(maxRelativeL2, row.metrics.relativeL2);
    }
    const natural = (a, b) => a.localeCompare(b, undefined, { numeric: true });
    return { groups: [...map.values()].sort((a, b) => natural(a.key, b.key)), roles: [...roles].sort(natural), maxRelativeL2 };
  }
  // Translated cross-format rows also answer to the tensor name in the other checkpoint.
  const matchesQuery = (row, query) => [row.name, row.baseName, row.targetName].some(n => n && n.toLowerCase().includes(query.toLowerCase()));
  function diffRows(rows, { query = '', status = 'attention', group = '', role = '' } = {}) {
    return rows.filter(r => matchesQuery(r, query) && (!group || groupKey(r.name) === group)
      && (!role || tensorRole(r.name) === role)
      && (!status || (status === 'attention' ? needsAttention(r) : r.status === status)))
      .sort((a, b) => Number(hasChange(b)) - Number(hasChange(a)) || (b.metrics?.maxAbs || 0) - (a.metrics?.maxAbs || 0) || a.name.localeCompare(b.name, undefined, { numeric: true }));
  }
  return { groupKey, groups, tensorRole, comparisonMatrix, hasChange, needsAttention, diffOverview, diffRows, matchesQuery };
});
