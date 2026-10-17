'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const backend = require('../src/backend');
const { translate, sourceIndex, ARCHITECTURES } = require('../src/tensor-names');
const view = require('../media/model-view');
const F = require('./fixtures');
let dir;
before(async () => { dir = await fs.mkdtemp(path.join(os.tmpdir(), 'weightscope-names-')); });
after(async () => { await fs.rm(dir, { recursive: true, force: true }); });
const file = name => path.join(dir, name);
const values = (n, seed) => Array.from({ length: n }, (_, i) => Math.fround(Math.sin(seed * 100 + i)));
const count = shape => shape.reduce((n, d) => n * d, 1);
// llama.cpp LlamaModel.permute, written as the literal reshape(heads, 2, half, cols).swapaxes(1, 2).
function permute(data, rows, heads) {
  const cols = data.length / rows, half = rows / heads / 2, out = [];
  for (let h = 0; h < heads; h++) for (let k = 0; k < half; k++) for (let j = 0; j < 2; j++) { const r = h * 2 * half + j * half + k; out.push(...data.slice(r * cols, (r + 1) * cols)); }
  return out;
}
const meta = (architecture, entries = []) => [['general.architecture', 8, F.str(architecture)], ...entries.map(([key, n]) => [key, 4, F.u32(n)])];
// A tiny GQA llama: 4 query heads, 2 key/value heads, head size 4.
const LLAMA = [
  ['model.embed_tokens.weight', 'token_embd.weight', [10, 16]], ['model.norm.weight', 'output_norm.weight', [16]], ['lm_head.weight', 'output.weight', [10, 16]],
  ['model.layers.0.input_layernorm.weight', 'blk.0.attn_norm.weight', [16]], ['model.layers.0.post_attention_layernorm.weight', 'blk.0.ffn_norm.weight', [16]],
  ['model.layers.0.self_attn.q_proj.weight', 'blk.0.attn_q.weight', [16, 16], 4], ['model.layers.0.self_attn.q_proj.bias', 'blk.0.attn_q.bias', [16], 4],
  ['model.layers.0.self_attn.k_proj.weight', 'blk.0.attn_k.weight', [8, 16], 2], ['model.layers.0.self_attn.v_proj.weight', 'blk.0.attn_v.weight', [8, 16]],
  ['model.layers.0.self_attn.o_proj.weight', 'blk.0.attn_output.weight', [16, 16]], ['model.layers.0.mlp.gate_proj.weight', 'blk.0.ffn_gate.weight', [24, 16]],
  ['model.layers.0.mlp.up_proj.weight', 'blk.0.ffn_up.weight', [24, 16]], ['model.layers.0.mlp.down_proj.weight', 'blk.0.ffn_down.weight', [16, 24]]
];
async function llamaPair(name, { permuteRows = true, metadata = meta('llama', [['llama.attention.head_count', 4], ['llama.attention.head_count_kv', 2]]) } = {}) {
  const hf = LLAMA.map(([n, , shape], i) => ({ name: n, shape, values: values(count(shape), i) }));
  hf.push({ name: 'model.layers.0.self_attn.rotary_emb.inv_freq', shape: [2], values: [1, .5] });
  const gguf = LLAMA.map(([, n, shape, heads], i) => ({ name: n, shape, values: heads && permuteRows ? permute(values(count(shape), i), shape[0], heads) : values(count(shape), i) }));
  gguf.push({ name: 'rope_freqs.weight', shape: [2], values: [1, 1] });
  await F.safetensors(file(`${name}.safetensors`), hf); await F.gguf(file(`${name}.gguf`), gguf, metadata);
  return [await backend.inspect(file(`${name}.safetensors`)), await backend.inspect(file(`${name}.gguf`))];
}

test('Hugging Face names pair with llama.cpp GGUF names and permuted Q/K rows compare equal', async () => {
  const [hf, gguf] = await llamaPair('llama');
  const rows = backend.structureDiff(hf, gguf), matching = backend.tensorMatching(hf, gguf);
  assert.equal(matching.kind, 'gguf'); assert.equal(matching.architecture, 'llama'); assert.equal(matching.matched, LLAMA.length);
  assert.match(matching.description, /llama architecture/); assert.match(matching.description, /rotary/);
  assert.deepEqual(rows.filter(r => r.status === 'removed').map(r => r.name), ['model.layers.0.self_attn.rotary_emb.inv_freq']);
  assert.deepEqual(rows.filter(r => r.status === 'added').map(r => r.name), ['rope_freqs.weight']);
  const q = rows.find(r => r.name === 'model.layers.0.self_attn.q_proj.weight'), k = rows.find(r => r.name === 'model.layers.0.self_attn.k_proj.weight');
  assert.equal(q.baseName, 'model.layers.0.self_attn.q_proj.weight'); assert.equal(q.targetName, 'blk.0.attn_q.weight');
  assert.match(q.layout.join(' '), /4 query heads/); assert.match(k.layout.join(' '), /2 key\/value heads/);
  assert.deepEqual(rows.find(r => r.name === 'model.layers.0.self_attn.v_proj.weight').layout, []);
  for (const row of rows.filter(r => r.status === 'unverified')) {
    const d = await backend.compareTensor(hf, gguf, row.name);
    assert.equal(d.status, 'equal', row.name); assert.equal(d.maxAbs, 0);
  }
  // Bounded samples gather the matching GGUF positions across row and head boundaries.
  const sampled = await backend.compareTensor(hf, gguf, q.name, { sampleSize: 64 });
  assert.equal(sampled.status, 'sample-match'); assert.equal(sampled.sampled, 64); assert.equal(sampled.total, 256);
});
test('GGUF baselines compare against Hugging Face targets with the same pairing', async () => {
  const [hf, gguf] = await llamaPair('reverse');
  const rows = backend.structureDiff(gguf, hf), q = rows.find(r => r.name === 'model.layers.0.self_attn.q_proj.weight');
  assert.equal(q.baseName, 'blk.0.attn_q.weight'); assert.equal(q.targetName, 'model.layers.0.self_attn.q_proj.weight');
  assert.equal(rows.find(r => r.name === 'rope_freqs.weight').status, 'removed');
  assert.equal((await backend.compareTensor(gguf, hf, q.name)).status, 'equal');
  assert.equal((await backend.compareTensor(gguf, hf, 'model.layers.0.self_attn.k_proj.weight', { sampleSize: 32 })).status, 'sample-match');
});
test('Q/K row restoration is applied, not assumed: unpermuted GGUF rows show drift', async () => {
  const [hf, gguf] = await llamaPair('unpermuted', { permuteRows: false });
  assert.equal((await backend.compareTensor(hf, gguf, 'model.layers.0.self_attn.q_proj.weight')).status, 'changed');
  assert.equal((await backend.compareTensor(hf, gguf, 'model.layers.0.self_attn.v_proj.weight')).status, 'equal');
});
test('missing or unusable head counts make Q/K analysis unavailable with an explanation', async () => {
  const [hf, gguf] = await llamaPair('noheads', { metadata: meta('llama') });
  await assert.rejects(backend.compareTensor(hf, gguf, 'model.layers.0.self_attn.q_proj.weight'), /llama\.attention\.head_count/);
  assert.equal((await backend.compareTensor(hf, gguf, 'model.layers.0.self_attn.v_proj.weight')).status, 'equal');
  const [hf2, odd] = await llamaPair('oddheads', { metadata: meta('llama', [['llama.attention.head_count', 3], ['llama.attention.head_count_kv', 2]]) });
  await assert.rejects(backend.compareTensor(hf2, odd, 'model.layers.0.self_attn.q_proj.weight'), /16 rows do not split into 3 heads/);
  // Variable-attention models store one head count per block as an array.
  const perBlock = n => Buffer.concat([F.u32(4), F.u64(1), F.u32(n)]);
  const [hf3, arrays] = await llamaPair('arrayheads', { metadata: [...meta('llama'), ['llama.attention.head_count', 9, perBlock(4)], ['llama.attention.head_count_kv', 9, perBlock(2)]] });
  assert.equal((await backend.compareTensor(hf3, arrays, 'model.layers.0.self_attn.k_proj.weight')).status, 'equal');
});
test('Gemma sandwich norms are renamed and llama.cpp’s +1 norm offset is removed', async () => {
  const norms = [['model.layers.0.input_layernorm.weight', 'blk.0.attn_norm.weight'], ['model.layers.0.post_attention_layernorm.weight', 'blk.0.post_attention_norm.weight'],
    ['model.layers.0.pre_feedforward_layernorm.weight', 'blk.0.ffn_norm.weight'], ['model.layers.0.post_feedforward_layernorm.weight', 'blk.0.post_ffw_norm.weight'], ['model.norm.weight', 'output_norm.weight']];
  const exact = i => Array.from({ length: 8 }, (_, j) => ((i * 8 + j) % 17 - 8) / 64);
  await F.safetensors(file('gemma.safetensors'), [...norms.map(([n], i) => ({ name: n, values: exact(i) })), { name: 'model.layers.0.self_attn.v_proj.weight', shape: [2, 4], values: exact(9) }]);
  await F.gguf(file('gemma.gguf'), [...norms.map(([, n], i) => ({ name: n, shape: [8], values: exact(i).map(v => v + 1) })), { name: 'blk.0.attn_v.weight', shape: [2, 4], values: exact(9) }], meta('gemma2'));
  const hf = await backend.inspect(file('gemma.safetensors')), gguf = await backend.inspect(file('gemma.gguf'));
  const rows = backend.structureDiff(hf, gguf);
  assert.ok(rows.every(r => r.status === 'unverified')); assert.match(backend.tensorMatching(hf, gguf).description, /\+1/);
  assert.equal(rows.find(r => r.name === norms[1][0]).targetName, 'blk.0.post_attention_norm.weight');
  assert.match(rows.find(r => r.name === norms[0][0]).layout[0], /1 \+ w/);
  for (const row of rows) assert.equal((await backend.compareTensor(hf, gguf, row.name)).status, 'equal', row.name);
});
test('stacked GGUF experts are compared one Hugging Face expert at a time', async () => {
  const experts = [values(12, 1), values(12, 2), values(12, 3)];
  await F.safetensors(file('moe.safetensors'), [...experts.map((v, e) => ({ name: `model.layers.0.block_sparse_moe.experts.${e}.w1.weight`, shape: [3, 4], values: v })), { name: 'model.layers.0.block_sparse_moe.gate.weight', shape: [3, 4], values: values(12, 4) }]);
  await F.gguf(file('moe.gguf'), [{ name: 'blk.0.ffn_gate_exps.weight', shape: [3, 3, 4], values: experts.flat() }, { name: 'blk.0.ffn_gate_inp.weight', shape: [3, 4], values: values(12, 4) }], meta('llama', [['llama.attention.head_count', 4]]));
  const hf = await backend.inspect(file('moe.safetensors')), gguf = await backend.inspect(file('moe.gguf'));
  const rows = backend.structureDiff(hf, gguf);
  assert.deepEqual(rows.map(r => r.status), ['unverified', 'unverified', 'unverified', 'unverified']);
  const second = rows.find(r => r.name === 'model.layers.0.block_sparse_moe.experts.1.w1.weight');
  assert.equal(second.targetName, 'blk.0.ffn_gate_exps.weight'); assert.deepEqual(second.right.shape, [3, 4]); assert.equal(second.right.elements, 12);
  assert.match(second.layout[0], /Expert 1 of 3/); assert.match(backend.tensorMatching(hf, gguf).description, /one expert at a time/);
  for (const row of rows) assert.equal((await backend.compareTensor(hf, gguf, row.name)).status, 'equal', row.name);
  assert.equal((await backend.compareTensor(hf, gguf, 'model.layers.0.block_sparse_moe.experts.2.w1.weight', { sampleSize: 4 })).status, 'sample-match');
});
test('unsupported architectures and ONNX keep exact matching and explain why', async () => {
  await F.safetensors(file('falcon.safetensors'), [{ name: 'model.layers.0.self_attn.v_proj.weight', values: [1, 2] }]);
  await F.gguf(file('falcon.gguf'), [{ name: 'blk.0.attn_v.weight', shape: [2], values: [1, 2] }], meta('falcon'));
  const hf = await backend.inspect(file('falcon.safetensors')), falcon = await backend.inspect(file('falcon.gguf'));
  const matching = backend.tensorMatching(hf, falcon);
  assert.equal(matching.kind, 'unsupported'); assert.match(matching.description, /“falcon”/);
  assert.deepEqual(backend.structureDiff(hf, falcon).map(r => r.status), ['added', 'removed']);
  await F.onnx(file('names.onnx'), [{ name: 'model.layers.0.self_attn.v_proj.weight', values: [1, 2] }]);
  await F.gguf(file('qwen.gguf'), [{ name: 'blk.0.attn_v.weight', shape: [2], values: [1, 2] }], meta('qwen2'));
  const onnx = await backend.inspect(file('names.onnx')), qwen = await backend.inspect(file('qwen.gguf'));
  assert.equal(backend.tensorMatching(onnx, qwen).kind, 'exact'); assert.equal(backend.tensorMatching(hf, qwen).kind, 'gguf');
  // Checkpoints in the same format still pair by exact name.
  assert.equal(backend.tensorMatching(hf, hf).kind, 'exact');
});
test('name translation covers multimodal prefixes, fused projections and per-architecture conversions', () => {
  assert.equal(translate('language_model.model.layers.3.self_attn.q_proj.weight', ARCHITECTURES.llama).name, 'blk.3.attn_q.weight');
  assert.equal(translate('model.language_model.layers.3.self_attn.q_norm.weight', ARCHITECTURES.gemma3).shift, 1);
  assert.equal(translate('language_model.lm_head.weight', ARCHITECTURES.gemma3).name, 'output.weight');
  assert.equal(translate('model.layers.0.self_attn.qkv_proj.weight', ARCHITECTURES.phi3).name, 'blk.0.attn_qkv.weight');
  assert.equal(translate('model.layers.0.self_attn.q_proj.weight', ARCHITECTURES.qwen2).rope, undefined);
  assert.equal(translate('model.layers.0.self_attn.q_proj.bias', ARCHITECTURES.olmo).rope, undefined);
  assert.equal(translate('model.layers.0.mlp.experts.7.down_proj.weight', ARCHITECTURES.qwen3moe).fallback, 'blk.0.ffn_down.7.weight');
  assert.equal(translate('model.layers.0.self_attn.rotary_emb.inv_freq', ARCHITECTURES.llama), null);
  // The index map is a bijection within each tensor.
  const layout = { offset: 0, columns: 3, rows: 24, heads: 3 }, seen = new Set(Array.from({ length: 72 }, (_, i) => sourceIndex(layout, i)));
  assert.equal(seen.size, 72); assert.equal(Math.max(...seen), 71);
});
test('comparison search also finds translated rows by their GGUF name', () => {
  const rows = [{ name: 'model.layers.0.self_attn.q_proj.weight', baseName: 'model.layers.0.self_attn.q_proj.weight', targetName: 'blk.0.attn_q.weight', status: 'unverified' }, { name: 'rope_freqs.weight', status: 'added' }];
  assert.deepEqual(view.diffRows(rows, { query: 'attn_q', status: '' }).map(r => r.name), ['model.layers.0.self_attn.q_proj.weight']);
  assert.equal(view.matchesQuery(rows[1], 'ROPE'), true);
});
