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
