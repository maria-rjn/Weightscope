'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const view = require('../media/model-view');

test('visual grouping aggregates named blocks without inventing an execution graph', () => {
  const tensors = [
    { name: 'model.layers.10.attn.weight', elements: 20, bytes: 40, dtype: 'F16' },
    { name: 'model.layers.2.attn.weight', elements: 10, bytes: 20, dtype: 'F16' },
    { name: 'model.layers.2.norm.weight', elements: 5, bytes: 20, dtype: 'F32' },
    { name: 'blk/0/weight', elements: 4, bytes: null, dtype: 'Q4_K' },
    { name: 'bias', elements: 1, bytes: 4, dtype: 'F32' }
  ];
  const groups = view.groups(tensors), layer = groups.find(g => g.key === 'model.layers.2');
  assert.equal(layer.elements, 15); assert.equal(layer.bytes, 40); assert.equal(layer.dtypes.size, 2);
  assert.equal(groups.reduce((n, g) => n + g.elements, 0), 40);
  assert.equal(groups.find(g => g.key === 'blk.0').unknownBytes, true);
  assert.ok(groups.findIndex(g => g.key === 'model.layers.2') < groups.findIndex(g => g.key === 'model.layers.10'));
  assert.equal(view.groupKey('bias'), 'Ungrouped tensors');
  assert.equal(view.groupKey('vision/encoder/block/3/attn/weight'), 'vision.encoder.block.3');
});
test('change map keeps pending, sampled and full matches distinct', () => {
  const rows = [
    { name: 'layer.0.weight', status: 'equal', metrics: { maxAbs: 0 } },
    { name: 'layer.1.weight', status: 'sample-match', metrics: { maxAbs: 0 } },
    { name: 'layer.2.weight', status: 'unverified' },
    { name: 'layer.3.weight', status: 'changed', metrics: { maxAbs: .1 } },
    { name: 'layer.3.bias', status: 'shape-changed' },
    { name: 'layer.4.weight', status: 'unavailable' }
  ];
  const summary = view.diffOverview(rows);
  assert.equal(summary.analyzed, 3); assert.equal(summary.changes, 2); assert.equal(summary.pending, 1); assert.equal(summary.unavailable, 1);
  assert.deepEqual(summary.groups.map(g => g.status), ['equal', 'sample-match', 'unverified', 'changed', 'unavailable']);
  assert.equal(summary.groups[3].maxAbs, .1); assert.equal(summary.groups[3].changes, 2);
});
test('precision changes remain visible when all numeric values match', () => {
  const row = { name: 'layer.weight', status: 'equal', dtypeChanged: true, metrics: { maxAbs: 0 } };
  assert.equal(view.diffOverview([row]).changes, 1);
  assert.equal(view.diffOverview([row]).groups[0].status, 'changed');
  assert.equal(view.diffRows([row]).length, 1);
});
test('added and removed groups have their own labels, while mixed structural changes stay changed', () => {
  const rows = [{ name: 'adapter.a.weight', status: 'added' }, { name: 'old.weight', status: 'removed' }, { name: 'mixed.a', status: 'added' }, { name: 'mixed.b', status: 'removed' }];
  const groups = view.diffOverview(rows).groups;
  assert.equal(groups.find(g => g.key === 'adapter.a').status, 'added');
  assert.equal(groups.find(g => g.key === 'old').status, 'removed');
  assert.equal(groups.find(g => g.key === 'mixed').status, 'changed');
});
test('focused diff filters preserve exhaustive search and order changed tensors first', () => {
  const rows = [
    { name: 'block.0.a', status: 'equal' },
    { name: 'block.0.b', status: 'changed', metrics: { maxAbs: 1 } },
    { name: 'block.1.c', status: 'changed', metrics: { maxAbs: 3 } },
    { name: 'block.1.d', status: 'unverified' },
    { name: 'block.2.e', status: 'sample-match' }
  ];
  assert.deepEqual(view.diffRows(rows).map(r => r.name), ['block.1.c', 'block.0.b', 'block.1.d']);
  assert.equal(view.diffRows(rows, { status: '' }).length, 5);
  assert.deepEqual(view.diffRows(rows, { group: 'block.1', query: '.c' }).map(r => r.name), ['block.1.c']);
  assert.equal(view.diffRows(rows, { status: 'sample-match' })[0].name, 'block.2.e');
});

test('drift matrix aligns actual tensor suffixes, preserves collisions and keeps a global finite scale', () => {
  const rows = [
    { name: 'model.layers.10.self_attn.q_proj.weight', status: 'changed', metrics: { relativeL2: .4 } },
    { name: 'model.layers.2.self_attn.q_proj.weight', status: 'sample-match', metrics: { relativeL2: 0 } },
    { name: 'model.layers.2.norm.weight', status: 'equal', dtypeChanged: true, metrics: { relativeL2: null } },
    { name: 'model/layers/2/norm/weight', status: 'unavailable' },
    { name: 'bias', status: 'added' },
    { name: 'encoder.weight', status: 'removed', metrics: { relativeL2: NaN } }
  ];
  const matrix = view.comparisonMatrix(rows);
  assert.equal(matrix.maxRelativeL2, .4); assert.equal(matrix.minRelativeL2, .4);
  assert.ok(matrix.groups.findIndex(g => g.key === 'model.layers.2') < matrix.groups.findIndex(g => g.key === 'model.layers.10'));
  assert.deepEqual(matrix.roles, ['bias', 'norm.weight', 'self_attn.q_proj.weight', 'weight']);
  const layer = matrix.groups.find(g => g.key === 'model.layers.2');
  assert.equal(layer.cells.get('norm.weight').length, 2);
  assert.equal(layer.cells.get('self_attn.q_proj.weight')[0], rows[1]);
  assert.equal(view.diffRows(rows, { status: '', group: 'model.layers.2', role: 'norm.weight' }).length, 2);
  assert.equal(rows[2].status, 'equal');
});

test('drift matrix retains every group and role beyond either visible page', () => {
  const rows = Array.from({ length: 135 }, (_, block) => Array.from({ length: 12 }, (_, role) => ({ name: `block.${block}.module.${role}.weight`, status: 'unverified' }))).flat();
  const matrix = view.comparisonMatrix(rows);
  assert.equal(matrix.groups.length, 135); assert.equal(matrix.roles.length, 12);
  assert.equal(matrix.groups.at(-1).key, 'block.134'); assert.equal(matrix.roles.at(-1), 'module.11.weight');
  assert.equal(matrix.groups.reduce((n, group) => n + [...group.cells.values()].reduce((m, cell) => m + cell.length, 0), 0), rows.length);
  assert.deepEqual(view.comparisonMatrix([]), { groups: [], roles: [], maxRelativeL2: 0, minRelativeL2: 0 });
});

test('drift scale keeps small changes visible on a log scale and stays honest about zero and undefined drift', () => {
  const matrix = view.comparisonMatrix([
    { name: 'block.0.weight', status: 'changed', metrics: { relativeL2: .001 } },
    { name: 'block.1.weight', status: 'changed', metrics: { relativeL2: .1 } },
    { name: 'block.2.weight', status: 'equal', metrics: { relativeL2: 0 } },
    { name: 'block.3.weight', status: 'equal', metrics: { relativeL2: null } }
  ]);
  assert.equal(matrix.minRelativeL2, .001); assert.equal(matrix.maxRelativeL2, .1);
  const log = view.driftScale(matrix, 'log'), linear = view.driftScale(matrix, 'linear');
  assert.equal(log.mode, 'log'); assert.equal(log.position(.001), 0); assert.ok(Math.abs(log.position(.01) - .5) < 1e-12); assert.equal(log.position(.1), 1);
  assert.equal(log.position(0), 0); assert.equal(log.position(null), null); assert.equal(log.position(NaN), null);
  assert.deepEqual(log.ticks.map(t => Number(t.toPrecision(6))), [.001, .01, .1]);
  assert.equal(linear.mode, 'linear'); assert.equal(linear.position(.05), .5); assert.deepEqual(linear.ticks, [0, .05, .1]);
  // One decade-spanning outlier cannot push the rest of the map off a six-decade log scale.
  const wide = view.driftScale({ maxRelativeL2: 1, minRelativeL2: 1e-12 }, 'log');
  assert.equal(wide.position(1e-9), 0); assert.equal(wide.ticks[0], 1e-6);
  assert.equal(view.driftScale({ maxRelativeL2: .2, minRelativeL2: .2 }, 'log').mode, 'linear');
  assert.deepEqual(view.driftScale({ maxRelativeL2: 0, minRelativeL2: 0 }).ticks, []);
  assert.equal(view.driftScale({ maxRelativeL2: 0, minRelativeL2: 0 }).position(0), null);
});

test('drift colors use one perceptual ramp, reversed for light themes, with readable ink', () => {
  assert.equal(view.heatColor(0), 'rgb(59, 15, 112)'); assert.equal(view.heatColor(1), 'rgb(252, 253, 191)');
  assert.equal(view.heatColor(1, true), 'rgb(20, 14, 54)'); assert.equal(view.heatColor(2), view.heatColor(1)); assert.equal(view.heatColor(-1), view.heatColor(0));
  assert.equal(view.heatInk(0), 'light'); assert.equal(view.heatInk(1), 'dark'); assert.equal(view.heatInk(0, true), 'dark'); assert.equal(view.heatInk(1, true), 'light');
});
