"""Generate independently serialized test checkpoints using upstream libraries."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import onnx
from onnx import helper, numpy_helper, TensorProto
from safetensors.torch import save_file
import gguf
import hashlib
from safetensors.numpy import save_file as save_numpy

directory = Path(sys.argv[1])
directory.mkdir(parents=True, exist_ok=True)
tensors = {"layer.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4), "layer.bias": torch.tensor([-1, 0, 1], dtype=torch.bfloat16), "noncontiguous": torch.arange(12, dtype=torch.float32).reshape(4, 3).t(), "empty": torch.empty(0)}
torch.save({"state_dict": tensors, "epoch": 7, "optimizer": {"slot": torch.ones(100)}}, directory / "official.pt")
torch.save(tensors, directory / "legacy.pth", _use_new_zipfile_serialization=False)
save_file({k: v.contiguous() for k, v in tensors.items()}, directory / "official.safetensors", metadata={"source": "upstream library"})

class PicklePayload:
    def __reduce__(self):
        import os
        return (os.system, (f'touch "{directory / "should-not-exist"}"',))

torch.save({"payload": PicklePayload()}, directory / "rejected.pt")
scripted = torch.jit.trace(torch.nn.Linear(4, 3), torch.ones(1, 4))
torch.jit.save(scripted, directory / "scripted.pt")
torch.save(torch.tensor([1.0, float("nan"), float("inf")]), directory / "nonfinite.pt")

weights = numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(3, 4), "layer.weight")
packed = helper.make_tensor("packed", TensorProto.FLOAT, [3], [1, 2, 3])
graph = helper.make_graph([helper.make_node("MatMul", ["input", "layer.weight"], ["output"])], "Official graph", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])], [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])], [weights, packed])
model = helper.make_model(graph, producer_name="upstream ONNX")
onnx.save_model(model, directory / "official.onnx")
onnx.save_model(model, directory / "external.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="external.bin", size_threshold=0)

expected = {}
writer = gguf.GGUFWriter(str(directory / "official.gguf"), "test")
writer.add_name("Official GGUF fixture")
writer.add_tensor("float.weight", np.arange(12, dtype=np.float32).reshape(3, 4))
expected["float.weight"] = list(map(float, range(12)))
source = np.linspace(-3, 3, 64, dtype=np.float32).reshape(2, 32)
for kind in (gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q4_1, gguf.GGMLQuantizationType.Q5_0, gguf.GGMLQuantizationType.Q5_1, gguf.GGMLQuantizationType.Q8_0):
    raw = gguf.quantize(source, kind)
    name = kind.name + ".weight"
    writer.add_tensor(name, raw, raw_dtype=kind)
    expected[name] = gguf.dequantize(raw, kind).ravel().tolist()
for kind, scale_offset, min_offset in ((gguf.GGMLQuantizationType.Q2_K, 80, 82), (gguf.GGMLQuantizationType.Q3_K, 108, None), (gguf.GGMLQuantizationType.Q4_K, 0, 2), (gguf.GGMLQuantizationType.Q5_K, 0, 2), (gguf.GGMLQuantizationType.Q6_K, 208, None)):
    # Exercise every packed bit and scale in four independent blocks.
    block, size = gguf.GGML_QUANT_SIZES[kind]
    raw = np.random.default_rng(int(kind)).integers(0, 256, size=(4, size), dtype=np.uint8)
    if scale_offset is not None:
        raw[:, scale_offset:scale_offset + 2] = np.frombuffer(np.float16(.0625).tobytes(), dtype=np.uint8)
    else:
        raw[:, :4] = np.frombuffer(np.float32(.0625).tobytes(), dtype=np.uint8)
    if min_offset is not None:
        raw[:, min_offset:min_offset + 2] = np.frombuffer(np.float16(.015625).tobytes(), dtype=np.uint8)
    name = kind.name + ".weight"
    writer.add_tensor(name, raw, raw_dtype=kind)
    expected[name] = gguf.dequantize(raw, kind).ravel().tolist()
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
(directory / "expected.json").write_text(json.dumps(expected))

# Convert a tiny GQA Llama the way llama.cpp's convert_hf_to_gguf.py does: gguf-py's
# tensor name map, LlamaModel.permute on Q/K, and Q8_0 for every matrix.
def permute(weights, n_head, n_head_kv):
    # Copied from llama.cpp conversion/llama.py (MIT).
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))
n_head, n_kv, head_dim, hidden, ff, blocks = 4, 2, 8, 32, 64, 2
shapes = {"model.embed_tokens.weight": (64, hidden), "model.norm.weight": (hidden,), "lm_head.weight": (64, hidden)}
for b in range(blocks):
    p = f"model.layers.{b}."
    shapes |= {p + "input_layernorm.weight": (hidden,), p + "post_attention_layernorm.weight": (hidden,), p + "self_attn.q_proj.weight": (hidden, hidden),
               p + "self_attn.k_proj.weight": (n_kv * head_dim, hidden), p + "self_attn.v_proj.weight": (n_kv * head_dim, hidden), p + "self_attn.o_proj.weight": (hidden, hidden),
               p + "mlp.gate_proj.weight": (ff, hidden), p + "mlp.up_proj.weight": (ff, hidden), p + "mlp.down_proj.weight": (hidden, ff)}
rng = np.random.default_rng(7)
source = {name: torch.from_numpy(rng.standard_normal(shape).astype(np.float32)) for name, shape in shapes.items()}
save_file(source, directory / "hf-llama.safetensors")
tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, blocks)
writer = gguf.GGUFWriter(str(directory / "llama-q8_0.gguf"), "llama")
writer.add_head_count(n_head)
writer.add_head_count_kv(n_kv)
cross = {}
for name, tensor in source.items():
    if name.endswith("q_proj.weight"):
        tensor = permute(tensor, n_head, n_head)
    if name.endswith("k_proj.weight"):
        tensor = permute(tensor, n_head, n_kv)
    data, target = tensor.numpy(), tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
    if data.ndim == 2:
        raw = gguf.quantize(data, gguf.GGMLQuantizationType.Q8_0)
        writer.add_tensor(target, raw, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        stored = gguf.dequantize(raw, gguf.GGMLQuantizationType.Q8_0).reshape(data.shape)
    else:
        writer.add_tensor(target, data)
        stored = data
    # Mean and max absolute error do not depend on element order, so they are
    # measured here in the permuted GGUF layout.
    delta = np.abs(stored.astype(np.float64) - data.astype(np.float64))
    cross[name] = {"target": target, "meanAbs": float(delta.mean()), "maxAbs": float(delta.max())}
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
(directory / "cross-format.json").write_text(json.dumps(cross))

# gguf-py's name map for each architecture WeightScope translates. Per-expert
# Hugging Face names are looked up by their merged form, as the converter does.
probe = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"] + [f"model.layers.2.{module}.weight" for module in (
    "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.qkv_proj", "self_attn.q_norm", "self_attn.k_norm",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp.gate_up_proj", "block_sparse_moe.gate", "mlp.gate", "mlp.shared_expert_gate",
    "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj",
    "block_sparse_moe.experts.3.w1", "block_sparse_moe.experts.3.w2", "block_sparse_moe.experts.3.w3",
    "mlp.experts.3.gate_proj", "mlp.experts.3.up_proj", "mlp.experts.3.down_proj")] + ["model.layers.2.self_attn.q_proj.bias", "model.layers.2.self_attn.k_proj.bias"]
architectures = {value: key for key, value in gguf.MODEL_ARCH_NAMES.items()}
names = {}
for arch in ("llama", "granite", "granitemoe", "smollm3", "arcee", "olmo", "olmo2", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "phi3", "gemma", "gemma2", "gemma3"):
    arch_map = gguf.get_tensor_name_map(architectures[arch], 4)
    names[arch] = {name: arch_map.get_name(name.replace(".experts.3.", ".experts."), try_suffixes=(".weight", ".bias")) for name in probe}
(directory / "gguf-names.json").write_text(json.dumps(names))

# Independently check every E4M3FN code against every E8M0 scale using
# PyTorch's dtype conversions, including NaNs and float32 overflow/underflow.
codes = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
scale_codes = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 8, axis=1)
packed = codes.copy().view("<u4")
fp8 = torch.from_numpy(codes.copy()).view(torch.float8_e4m3fn).float()
scales = torch.from_numpy(scale_codes.copy()).view(torch.float8_e8m0fnu).float()
decoded = (fp8 * scales.repeat_interleave(32, dim=-1)).flatten().tolist()
def json_float(value):
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "Infinity"
    if np.isneginf(value):
        return "-Infinity"
    return value
(directory / "mxfp8-expected.json").write_text(json.dumps(list(map(json_float, decoded)), allow_nan=False))
ollama_root = directory / "ollama"
blob_dir = ollama_root / "blobs"
blob_dir.mkdir(parents=True)
blob_file = blob_dir / "temporary"
save_numpy({"packed.weight": packed, "packed.weight.scale": scale_codes}, str(blob_file), metadata={"quant_type": "mxfp8", "group_size": "32"})
digest = hashlib.sha256(blob_file.read_bytes()).hexdigest()
size = blob_file.stat().st_size
blob_file.rename(blob_dir / f"sha256-{digest}")
manifest_dir = ollama_root / "manifests" / "registry.ollama.ai" / "library" / "mxfp8"
manifest_dir.mkdir(parents=True)
(manifest_dir / "latest").write_text(json.dumps({"schemaVersion": 2, "layers": [{"mediaType": "application/vnd.ollama.image.tensor", "digest": f"sha256:{digest}", "size": size}]}))
print("Generated official framework fixtures.")
