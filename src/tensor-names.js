'use strict';
// Pairs Hugging Face transformers tensor names with the llama.cpp GGUF names
// they were converted to, and records how convert_hf_to_gguf.py rearranged the
// values, so a GGUF file can be compared with its source checkpoint.

// GGUF general.architecture values whose conversion is reproduced here.
// rope: suffixes of q_proj/k_proj tensors that LlamaModel.permute moves from the
// Hugging Face half-split rotary layout into llama.cpp's interleaved layout.
// normShift: added to every *norm.weight (Gemma RMSNorm scales by 1 + w).
// postNorms: post_attention_layernorm normalizes the attention output (sandwich
// norms) instead of being the FFN input norm.
const LLAMA_ROPE = ['weight', 'bias'];
const ARCHITECTURES = {
  llama: { rope: LLAMA_ROPE }, granite: { rope: LLAMA_ROPE }, granitemoe: { rope: LLAMA_ROPE }, smollm3: { rope: LLAMA_ROPE }, arcee: { rope: LLAMA_ROPE },
  olmo: { rope: ['weight'] }, olmo2: { postNorms: true }, qwen2: {}, qwen2moe: {}, qwen3: {}, qwen3moe: {}, phi3: {},
  gemma: { normShift: 1 }, gemma2: { normShift: 1, postNorms: true }, gemma3: { normShift: 1, postNorms: true }
};
const GLOBAL = { 'model.embed_tokens': 'token_embd', 'model.norm': 'output_norm', lm_head: 'output' };
const BLOCK = {
  input_layernorm: 'attn_norm', post_attention_layernorm: 'ffn_norm', pre_feedforward_layernorm: 'ffn_norm', post_feedforward_layernorm: 'post_ffw_norm',
  'self_attn.q_proj': 'attn_q', 'self_attn.k_proj': 'attn_k', 'self_attn.v_proj': 'attn_v', 'self_attn.o_proj': 'attn_output', 'self_attn.qkv_proj': 'attn_qkv',
  'self_attn.q_norm': 'attn_q_norm', 'self_attn.k_norm': 'attn_k_norm',
  'mlp.gate_proj': 'ffn_gate', 'mlp.up_proj': 'ffn_up', 'mlp.down_proj': 'ffn_down', 'mlp.gate_up_proj': 'ffn_up',
  'block_sparse_moe.gate': 'ffn_gate_inp', 'mlp.gate': 'ffn_gate_inp', 'mlp.shared_expert_gate': 'ffn_gate_inp_shexp',
  'mlp.shared_expert.gate_proj': 'ffn_gate_shexp', 'mlp.shared_expert.up_proj': 'ffn_up_shexp', 'mlp.shared_expert.down_proj': 'ffn_down_shexp'
};
const EXPERT_ROLES = { w1: 'gate', w3: 'up', w2: 'down', gate_proj: 'gate', up_proj: 'up', down_proj: 'down' };
const ROPE_ROLES = { 'self_attn.q_proj': 'query', 'self_attn.k_proj': 'key/value' };
const shapeOf = t => t.logicalShape || t.shape;

// The GGUF tensor a Hugging Face tensor becomes, or null when it has no counterpart.
function translate(name, config = {}) {
  const hf = name.replace(/^model\.language_model\./, 'model.').replace(/^language_model\./, '');
  const parts = /^(.+)\.(weight|bias)$/.exec(hf); if (!parts) return null;
  const [, stem, suffix] = parts, shift = config.normShift && hf.endsWith('norm.weight') ? config.normShift : 0;
  if (Object.hasOwn(GLOBAL, stem)) return { name: `${GLOBAL[stem]}.${suffix}`, shift };
  const layer = /^model\.layers\.(\d+)\.(.+)$/.exec(stem); if (!layer) return null;
  const [, block, module] = layer, expert = /^(?:block_sparse_moe|mlp)\.experts\.(\d+)\.(w[123]|gate_proj|up_proj|down_proj)$/.exec(module);
  // Current GGUF stacks experts into one tensor; early Mixtral files kept one per expert.
  if (expert) { const role = EXPERT_ROLES[expert[2]]; return { name: `blk.${block}.ffn_${role}_exps.${suffix}`, expert: Number(expert[1]), fallback: `blk.${block}.ffn_${role}.${expert[1]}.${suffix}`, shift }; }
  const target = module === 'post_attention_layernorm' && config.postNorms ? 'post_attention_norm' : Object.hasOwn(BLOCK, module) ? BLOCK[module] : null;
  if (!target) return null;
  return { name: `blk.${block}.${target}.${suffix}`, block: Number(block), rope: config.rope?.includes(suffix) ? ROPE_ROLES[module] : undefined, shift };
}
function headCount(metadata, key, block) {
  if (metadata?.[key] === undefined) return undefined;
  let value = metadata[key];
  if (value && typeof value === 'object') value = block < (value.preview?.length ?? 0) ? value.preview[block] : undefined;
  const n = Number(value);
  return Number.isSafeInteger(n) && n > 0 ? n : null;
}
// LlamaModel.permute reshapes rows to (heads, 2, rows / heads / 2) and swaps the
// last two axes; K uses the key/value head count.
function ropeLayout(gguf, architecture, match, rows) {
  const key = `${architecture}.attention.head_count`, kv = match.rope === 'key/value' ? headCount(gguf.metadata, `${key}_kv`, match.block) : undefined;
  const heads = kv === undefined ? headCount(gguf.metadata, key, match.block) : kv, declared = kv === undefined ? key : `${key}_kv`;
  if (!heads) return { error: `Cannot restore llama.cpp’s rotary row order: the GGUF file does not declare a usable ${declared} for block ${match.block}.` };
  if (!Number.isSafeInteger(rows) || rows % (heads * 2)) return { error: `Cannot restore llama.cpp’s rotary row order: ${rows} rows do not split into ${heads} heads of even size.` };
  return { heads, rows, note: `Rows were restored from llama.cpp’s interleaved rotary layout (${heads} ${match.rope} heads) before comparing.` };
}
// Element index in the GGUF tensor that holds element `index` of the Hugging Face tensor.
function sourceIndex(layout, index) {
  let row = Math.floor(index / layout.columns);
  if (layout.heads) {
    const size = layout.rows / layout.heads, half = size / 2, within = row % size;
    row += (within % half) * 2 + Math.floor(within / half) - within;
  }
  return layout.offset + row * layout.columns + index % layout.columns;
}
function exactPairs(base, target) {
  const a = new Map(base.tensors.map(t => [t.name, t])), b = new Map(target.tensors.map(t => [t.name, t]));
  return [...new Set([...a.keys(), ...b.keys()])].map(name => ({ name, left: a.get(name), right: b.get(name) }));
}
// Rows are named after the Hugging Face tensor: merged GGUF experts have one row per expert.
function translatedPairs(hf, gguf, ggufSide, architecture, config) {
  const available = new Map(gguf.tensors.map(t => [t.name, t])), used = new Set(), pairs = [];
  let matched = 0;
  for (const tensor of hf.tensors) {
    const match = translate(tensor.name, config);
    let source = match && available.get(match.name), view = source, offset = 0, notes = [];
    if (source && match.expert !== undefined) {
      const experts = shapeOf(source)[0];
      if (shapeOf(source).length < 2 || match.expert >= experts) source = view = undefined;
      else {
        view = { ...source, shape: source.shape.slice(1), elements: source.elements / experts }; delete view.logicalShape; delete view.logicalElements;
        offset = match.expert * view.elements; notes.push(`Expert ${match.expert} of ${experts} in the stacked GGUF tensor.`);
      }
    }
    if (!source && match?.fallback) source = view = available.get(match.fallback);
    if (!source || (used.has(source.name) && match.expert === undefined)) { pairs.push({ name: tensor.name, [ggufSide === 'left' ? 'right' : 'left']: tensor }); continue; }
    used.add(source.name); matched++;
    const shape = shapeOf(tensor), rows = shape[0], columns = shape.length ? shape.slice(1).reduce((n, d) => n * d, 1) : 1;
    const layout = { side: ggufSide, tensor: source.name, offset, columns, notes };
    if (match.rope) { const rope = ropeLayout(gguf, architecture, match, rows); if (rope.error) layout.error = rope.error; else { Object.assign(layout, { heads: rope.heads, rows }); notes.push(rope.note); } }
    if (match.shift) { layout.shift = match.shift; notes.push(`llama.cpp stores these norm weights as ${match.shift} + w; ${match.shift} was subtracted from the GGUF values.`); }
    pairs.push(ggufSide === 'left' ? { name: tensor.name, left: view, right: tensor, layout } : { name: tensor.name, left: tensor, right: view, layout });
  }
  for (const t of gguf.tensors) if (!used.has(t.name)) pairs.push({ name: t.name, [ggufSide]: t });
  return { pairs, matched };
}
function describe(architecture, config, pairs, matched) {
  const notes = [`Hugging Face tensor names are translated to llama.cpp GGUF names for the ${architecture} architecture (${matched} matched).`];
  if (config.rope && pairs.some(p => p.layout?.heads || p.layout?.error)) notes.push('Query and key rows are restored from llama.cpp’s interleaved rotary layout before comparing.');
  if (pairs.some(p => p.layout?.shift)) notes.push(`The +${config.normShift} llama.cpp adds to norm weights is removed.`);
  if (pairs.some(p => p.layout?.offset)) notes.push('Stacked GGUF expert tensors are compared one expert at a time.');
  return notes.join(' ');
}
function buildPairs(base, target) {
  const exact = exactPairs(base, target), exactMatched = exact.filter(p => p.left && p.right).length;
  const ggufSide = base.format === 'gguf' && target.format !== 'gguf' ? 'left' : target.format === 'gguf' && base.format !== 'gguf' ? 'right' : null;
  const exactResult = (description = 'Tensor names are matched exactly.', kind = 'exact') => ({ pairs: exact, matching: { kind, description } });
  const gguf = ggufSide === 'left' ? base : target, hf = ggufSide === 'left' ? target : base;
  // ONNX exports may transpose weights, so only native checkpoint formats are translated.
  if (!ggufSide || !['safetensors', 'pytorch'].includes(hf.format)) return exactResult();
  const architecture = gguf.metadata?.['general.architecture'], config = typeof architecture === 'string' && Object.hasOwn(ARCHITECTURES, architecture) ? ARCHITECTURES[architecture] : null;
  if (!config) {
    if (translatedPairs(hf, gguf, ggufSide, '', {}).matched <= exactMatched) return exactResult();
    return exactResult(`Tensor names are matched exactly. Translating Hugging Face names is not supported for the GGUF architecture “${architecture || 'unknown'}”, so renamed tensors appear as added and removed.`, 'unsupported');
  }
  const { pairs, matched } = translatedPairs(hf, gguf, ggufSide, architecture, config), names = new Set(pairs.map(p => p.name));
  if (matched <= exactMatched || names.size !== pairs.length) return exactResult();
  return { pairs, matching: { kind: 'gguf', architecture, matched, description: describe(architecture, config, pairs, matched) } };
}
const cache = new WeakMap();
// Models are immutable once inspected, so pairing is computed once per baseline/target.
function pairTensors(base, target) {
  let targets = cache.get(base); if (!targets) cache.set(base, targets = new WeakMap());
  let result = targets.get(target);
  if (!result) {
    result = buildPairs(base, target);
    result.pairs.sort((x, y) => x.name.localeCompare(y.name, undefined, { numeric: true }));
    result.byName = new Map(result.pairs.map(p => [p.name, p]));
    targets.set(target, result);
  }
  return result;
}
module.exports = { pairTensors, translate, sourceIndex, ARCHITECTURES };
