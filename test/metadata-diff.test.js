'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { metadataDiff } = require('../src/math');
test('metadata comparison reports additions, removals and typed changes without treating object order as a change', () => {
  const rows = metadataDiff({ metadata: { removed: 'before', changed: 1, same: { a: 1, b: 2 }, null: null } }, { metadata: { added: 'after', changed: '1', same: { b: 2, a: 1 }, null: null } });
  assert.deepEqual(rows, [{ key: 'added', status: 'added', left: null, right: 'after' }, { key: 'changed', status: 'changed', left: 1, right: '1' }, { key: 'removed', status: 'removed', left: 'before', right: null }]);
  assert.deepEqual(metadataDiff({}, {}), []);
});
