'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { statistics, difference } = require('../src/math');
const sum = counts => counts.reduce((a, b) => a + b, 0);

test('comparison histograms share a domain and do not disguise shifted distributions', () => {
  const result = difference([0, 1], [100, 101], 100);
  const d = result.distributions;
  assert.deepEqual([d.min, d.max], [0, 101]);
  assert.equal(d.baseline.length, 24); assert.equal(d.target.length, 24);
  assert.equal(d.baseline[0], 2); assert.equal(d.target.at(-1), 2);
  assert.equal(sum(d.baseline), 2); assert.equal(sum(d.target), 2);
  assert.equal(result.sampled, 2); assert.equal(result.total, 100); assert.equal(result.exact, false);
});

test('constant distributions have a nonzero axis range and preserve every finite value', () => {
  for (const value of [0, -3, Number.MIN_VALUE, Number.MAX_VALUE, -Number.MAX_VALUE]) {
    const result = statistics([value, value, NaN, Infinity, -Infinity], 5);
    assert.ok(result.histogramRange.min < result.histogramRange.max);
    assert.ok(Number.isFinite(result.histogramRange.min)); assert.ok(Number.isFinite(result.histogramRange.max));
    assert.equal(sum(result.histogram), 2); assert.equal(result.nan, 1); assert.equal(result.inf, 2);
  }
});

test('histograms with an overflowing span still place extreme values in their correct bins', () => {
  const d = difference([-Number.MAX_VALUE, 0, Number.MAX_VALUE], [0, 0, 0], 3).distributions;
  assert.equal(d.baseline[0], 1); assert.equal(d.baseline.at(-1), 1);
  assert.equal(sum(d.baseline), 3); assert.equal(sum(d.target), 3);
  assert.equal(d.target[12], 3);
});

test('empty and non-finite samples do not fabricate a distribution or a zero-norm metric', () => {
  for (const values of [[], [NaN, Infinity, -Infinity]]) {
    const stats = statistics(values, values.length);
    assert.deepEqual(stats.histogramRange, { min: null, max: null }); assert.equal(sum(stats.histogram), 0);
    const d = difference(values, values, values.length).distributions;
    assert.equal(d.min, null); assert.equal(d.max, null); assert.equal(sum(d.baseline), 0); assert.equal(sum(d.target), 0);
  }
  const result = difference([0, 0], [1, 2], 2);
  assert.equal(result.relativeL2, null); assert.equal(result.status, 'changed');
  assert.equal(sum(result.distributions.baseline), 2); assert.equal(sum(result.distributions.target), 2);
});
