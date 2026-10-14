'use strict';
const path = require('node:path');
const fs = require('node:fs/promises');
const F = require('../test/fixtures');
async function makeDemo(directory = path.join(__dirname, '..', 'examples')) {
  await fs.mkdir(directory, { recursive: true });
  const tensors = [];
  function weight(name, shape, scale = 1) { const count = shape.reduce((a, b) => a * b, 1); tensors.push({ name, shape, values: Array.from({ length: count }, (_, i) => Math.sin(i * 12.9898 + tensors.length * 78.233) * Math.cos(i * .018) * scale) }); }
  weight('model.embed_tokens.weight', [512, 64], .04);
  for (let i = 0; i < 4; i++) {
    for (const p of ['q_proj', 'k_proj', 'v_proj', 'o_proj']) weight(`model.layers.${i}.self_attn.${p}.weight`, [64, 64], .08);
    for (const p of ['gate_proj', 'up_proj']) weight(`model.layers.${i}.mlp.${p}.weight`, [128, 64], .06);
    weight(`model.layers.${i}.mlp.down_proj.weight`, [64, 128], .06);
    weight(`model.layers.${i}.input_layernorm.weight`, [64], .02);
  }
  weight('model.norm.weight', [64], .01); weight('lm_head.weight', [512, 64], .04);
  const base = path.join(directory, 'tiny-transformer.safetensors'), target = path.join(directory, 'tiny-transformer-finetuned.safetensors');
  await F.safetensors(base, tensors, { architecture: 'Synthetic tiny transformer', purpose: 'Demo fixture, not a trained model', layers: '4' });
  const changed = tensors.filter(t => t.name !== 'model.norm.weight').map(t => ({ ...t, values: t.values.map((v, j) => t.name.includes('layers.2.') ? v + Math.sin(j) * .008 : v) }));
  changed.push({ name: 'adapter.lora_A.weight', shape: [4, 64], values: Array(256).fill(.001) });
  await F.safetensors(target, changed, { architecture: 'Synthetic tiny transformer', training_step: '1200' });
  await F.gguf(path.join(directory, 'tiny-model.gguf'), [{ name: 'blk.0.attn_q.weight', shape: [2, 2], values: [-1, 0, 1, 2] }]);
  await F.onnx(path.join(directory, 'tiny-model.onnx'), [{ name: 'linear.weight', shape: [2, 2], values: [-1, 0, 1, 2] }, { name: 'linear.bias', shape: [2], values: [0, .5] }], [{ name: 'linear', type: 'Gemm', inputs: ['input', 'linear.weight', 'linear.bias'], outputs: ['output'] }]);
  return { base, target };
}
if (require.main === module) makeDemo().then(({ base, target }) => console.log(`Created demo checkpoints:\n${base}\n${target}`)).catch(e => { console.error(e); process.exitCode = 1; });
module.exports = { makeDemo };
