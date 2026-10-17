'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const help = require('../media/help');
const { TYPES } = require('../src/parsers/gguf');

test('every explanation the inspector opens exists and belongs to a Guide section', () => {
  const app = fs.readFileSync(path.join(__dirname, '..', 'media', 'app.js'), 'utf8');
  const used = new Set([...app.matchAll(/\bhelp(?:Body)?\('([^']+)'/g)].map(m => m[1]));
  assert.ok(used.size >= 25, `only ${used.size} explanations are used`);
  for (const key of used) assert.ok(help.TOPICS[key], `missing explanation: ${key}`);
  const sections = new Set(help.SECTIONS.map(([id]) => id)), settings = { sampleSize: 32768, maxDiffTensors: 500 };
  for (const [key, topic] of Object.entries(help.TOPICS)) {
    assert.ok(topic.title && sections.has(topic.section), `${key} needs a title and a known section`);
    assert.ok(topic.what || topic.terms, `${key} needs a plain explanation`);
    for (const field of ['what', 'read', 'method']) if (typeof topic[field] === 'function') assert.match([topic[field](settings)].flat().join(' '), /\S/);
  }
  assert.match(help.TOPICS.scan.method(settings), /32,768 values/);
  assert.deepEqual(Object.keys(help.MAP_COLORS).sort(), ['peak', 'precision', 'rms', 'size']);
});
test('status explanations cover every comparison result shown in the inspector', () => {
  const terms = new Map(help.TOPICS.statuses.terms);
  for (const label of ['Not analyzed', 'Changed', 'Full match', 'Sample matches', 'Added', 'Removed', 'Shape changed', 'Numeric units differ', 'Unavailable', 'Precision changed']) assert.ok(terms.get(label), label);
});
test('precision notes state the same bits per weight as the GGUF block layouts', () => {
  for (const [name, count, bytes] of Object.values(TYPES)) {
    const note = help.precisionNote(name), stated = /\(([\d.]+) bits per weight\)/.exec(note || '');
    if (['F32', 'F16', 'BF16', 'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_K'].includes(name)) assert.ok(note, `${name} needs a note`);
    if (stated) assert.equal(Number(stated[1]), bytes * 8 / count, name);
  }
  assert.match(help.precisionNote('IQ4_XS'), /not decoded/);
  assert.match(help.precisionNote('F8_E4M3FN'), /4 exponent/);
  assert.match(help.precisionNote('I64'), /Integers/);
  assert.equal(help.precisionNote('STRING'), null);
});
test('role hints read Hugging Face, GPT-2 and llama.cpp names, and stay silent when unsure', () => {
  const role = name => help.tensorRole(name) || '';
  assert.match(role('model.layers.3.self_attn.q_proj.weight'), /query projection/);
  assert.match(role('blk.3.attn_q.weight'), /query projection/);
  assert.match(role('blk.0.attn_k_norm.weight'), /Query\/key norm/);
  assert.match(role('model.layers.0.self_attn.k_proj.bias'), /key projection.*bias/);
  assert.match(role('transformer.h.0.attn.c_attn.weight'), /Fused attention/);
  assert.match(role('transformer.h.0.attn.c_proj.weight'), /Attention output/);
  assert.match(role('transformer.h.0.mlp.c_proj.weight'), /down projection/);
  assert.match(role('blk.0.ffn_gate_inp.weight'), /router/);
  assert.match(role('blk.0.ffn_down_exps.weight'), /down projection.*stacks every expert/);
  assert.match(role('model.layers.1.mlp.experts.3.w2.weight'), /one expert/);
  assert.match(role('blk.0.post_attention_norm.weight'), /Sandwich/);
  assert.match(role('model.norm.weight'), /Final norm/);
  assert.match(role('model.layers.0.norm.weight'), /^Normalization scale/);
  assert.match(role('token_embd.weight'), /Token embedding/);
  assert.match(role('output.weight'), /LM head/);
  assert.match(role('v.blk.0.attn_q.weight'), /^In the vision encoder/);
  assert.match(role('blk.0.attn_q.weight.scale'), /Quantization scales/);
  assert.match(role('conv_proj.weight'), /Convolution/);
  assert.match(role('adapter.lora_A.weight'), /LoRA/);
  assert.equal(help.tensorRole('linear.weight'), null);
});
