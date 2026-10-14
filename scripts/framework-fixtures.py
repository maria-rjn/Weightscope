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
