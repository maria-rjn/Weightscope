"""Restricted PyTorch inspection and explicit Save As edits. JSON in/out; no checkpoint code."""
import json
import math
import os
import re
import sys
import zipfile
from collections.abc import Mapping


def runtime():
    try:
        import torch
    except ImportError as exc:
        raise ValueError("PyTorch is not installed in this interpreter. Install torch>=2.6 and set weightscope.pythonPath. See WeightScope: PyTorch Setup Instructions.") from exc
    match = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if not match or tuple(map(int, match.groups())) < (2, 6):
        raise ValueError("WeightScope requires PyTorch >= 2.6 for restricted weights-only loading. Upgrade this interpreter's torch package.")
    return torch


def load(file, torch, metadata=False):
    modern = zipfile.is_zipfile(file)
    if modern:
        with zipfile.ZipFile(file) as archive:
            names = archive.namelist()
            if any("/code/" in n or n.endswith("/constants.pkl") for n in names):
                raise ValueError("TorchScript archives are not supported. Export a plain state_dict or ONNX model; WeightScope does not execute model code.")
    elif os.path.getsize(file) > 512 * 1024 * 1024:
        raise ValueError("Legacy non-ZIP checkpoints over 512 MiB are not supported. Re-save as a modern torch.save state_dict or safetensors.")
    kwargs = dict(map_location="cpu", weights_only=True)
    if modern:
        kwargs["mmap"] = True
    if metadata:
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            return torch.load(file, **kwargs)
    return torch.load(file, **kwargs)


def tensors(root, torch):
    # Select the model state rather than adding optimizer slots to the model count.
    if isinstance(root, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in root and isinstance(root[key], Mapping):
                root = root[key]
                break
    result = {}

    def visit(value, prefix="", depth=0):
        if depth > 32:
            raise ValueError("Checkpoint nesting exceeds the limit.")
        if torch.is_tensor(value):
            name = prefix or "tensor"
            if name in result:
                raise ValueError(f"Ambiguous tensor name: {name}")
            if len(result) >= 200000:
                raise ValueError("Checkpoint tensor count exceeds the limit.")
            result[name] = value
        elif isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(key, (str, int)):
                    visit(item, f"{prefix}.{key}" if prefix else str(key), depth + 1)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                visit(item, f"{prefix}.{i}" if prefix else str(i), depth + 1)

    visit(root)
    if not result:
        raise ValueError("No tensors found. Use a plain tensor/state_dict checkpoint. Custom pickled modules are not supported.")
    return result


DTYPES = {"float64": "F64", "float32": "F32", "float16": "F16", "bfloat16": "BF16", "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8", "uint8": "U8", "bool": "BOOL", "float8_e4m3fn": "F8_E4M3FN", "float8_e5m2": "F8_E5M2", "float8_e4m3fnuz": "F8_E4M3FNUZ", "float8_e5m2fnuz": "F8_E5M2FNUZ"}


def inspect(file, torch):
    # FakeTensorMode cannot construct quantized fake tensors in some torch versions.
    # Do not fall back to an eager full checkpoint load just to open the inspector.
    root = load(file, torch, metadata=True)
    rows = []
    for name, tensor in tensors(root, torch).items():
        dtype = str(tensor.dtype).removeprefix("torch.")
        rows.append(dict(name=name, shape=list(tensor.shape), dtype=DTYPES.get(dtype, dtype.upper()), elements=tensor.numel(), bytes=tensor.numel() * tensor.element_size(), storage=dict(kind="pytorch", editable=tensor.layout == torch.strided and not tensor.is_quantized and not tensor.is_complex())))
    metadata = {"torchVersion": torch.__version__, "loading": "weights_only=True, CPU, FakeTensorMode"}
    if isinstance(root, Mapping):
        for key, value in root.items():
            if isinstance(key, str) and isinstance(value, (str, int, float, bool)) and len(str(value)) < 4096:
                metadata[key] = value
    return dict(format="pytorch", tensors=rows, metadata=metadata, warnings=["Counts describe stored tensors, including buffers and aliases; they do not establish unique trainable parameter counts."])


def sample(file, name, windows, torch):
    root = load(file, torch)
    tensor = tensors(root, torch).get(name)
    if tensor is None:
        raise ValueError("Unknown tensor.")
    if tensor.layout != torch.strided or tensor.is_complex() or tensor.is_quantized:
        raise ValueError("Numerical analysis currently supports dense real, non-quantized PyTorch tensors. Export quantized models as GGUF or ONNX for quantization inspection.")
    if sum(w["count"] for w in windows) > 262144:
        raise ValueError("Sample exceeds the limit.")
    output = []
    with torch.no_grad():
        for window in windows:
            start, count = window["start"], window["count"]
            if start < 0 or count < 0 or start + count > tensor.numel():
                raise ValueError("Invalid sample window.")
            if tensor.is_contiguous():
                values = tensor.view(-1)[start:start + count]
            else:
                coordinates = torch.unravel_index(torch.arange(start, start + count), tuple(tensor.shape))
                values = tensor[coordinates]
            values = values.to(torch.float64).tolist()
            for value in values:
                if str(tensor.dtype) in ("torch.int64", "torch.uint64") and abs(value) > 2**53 - 1:
                    raise ValueError("Integer weights exceed exact JavaScript precision.")
                output.append("NaN" if math.isnan(value) else "Infinity" if value == math.inf else "-Infinity" if value == -math.inf else value)
    return output


def edit(file, output, edits, torch):
    root = load(file, torch)
    originals = tensors(root, torch)
    replacements = {}
    with torch.no_grad():
        for change in edits:
            name = change['name']
            tensor = originals.get(name)
            if tensor is None or tensor.layout != torch.strided or tensor.is_quantized or tensor.is_complex():
                raise ValueError('Editing requires dense real, non-quantized PyTorch tensors.')
            operation = change['operation']
            if operation == 'replace':
                donor = tensors(load(change['sourcePath'], torch), torch).get(change['sourceName'])
                if donor is None or donor.shape != tensor.shape or donor.dtype != tensor.dtype or donor.layout != torch.strided or donor.is_quantized or donor.is_complex():
                    raise ValueError('Replacement tensors must have the same shape and dtype with dense real storage.')
                replacements[name] = donor.clone()
            elif operation == 'zero':
                replacements[name] = torch.zeros_like(tensor)
            elif operation == 'scale':
                factor = change['factor']
                if not isinstance(factor, (int, float)) or not math.isfinite(factor) or not tensor.is_floating_point():
                    raise ValueError('Scaling requires floating-point tensors and a finite factor.')
                # Operate on a separate tensor so aliased, unselected weights remain unchanged.
                result = (tensor.to(torch.float64) * factor).to(tensor.dtype)
                if not torch.isfinite(tensor).all().item() or not torch.isfinite(result).all().item():
                    raise ValueError('Scaling requires finite weights and must not overflow the tensor dtype.')
                replacements[name] = result
            else:
                raise ValueError('Unknown tensor edit operation.')

    def replace(value, prefix=''):
        if torch.is_tensor(value):
            return replacements.get(prefix or 'tensor', value)
        if isinstance(value, Mapping):
            for key in list(value):
                if isinstance(key, (str, int)):
                    value[key] = replace(value[key], f'{prefix}.{key}' if prefix else str(key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                value[i] = replace(item, f'{prefix}.{i}' if prefix else str(i))
        elif isinstance(value, tuple):
            return tuple(replace(item, f'{prefix}.{i}' if prefix else str(i)) for i, item in enumerate(value))
        return value

    model_key = next((key for key in ('state_dict', 'model_state_dict', 'model') if isinstance(root, Mapping) and key in root and isinstance(root[key], Mapping)), None)
    if model_key is not None:
        root[model_key] = replace(root[model_key])
    else:
        root = replace(root)
    torch.save(root, output)
    return {'edited': len(replacements)}


def main():
    try:
        request = json.load(sys.stdin)
        torch = runtime()
        if request["action"] == "inspect":
            result = inspect(request["path"], torch)
        elif request["action"] == "sample":
            result = sample(request["path"], request["name"], request["windows"], torch)
        elif request['action'] == 'edit':
            result = edit(request['path'], request['output'], request['edits'], torch)
        else:
            raise ValueError("Unknown action.")
        print(json.dumps(dict(ok=True, data=result), allow_nan=False))
    except Exception as exc:
        message = str(exc)
        if "Weights only load failed" in message:
            message = "Restricted weights-only loading rejected this checkpoint. Only plain tensors/state_dicts and supported scalar metadata are accepted. Re-export it from its trusted training environment."
        print(json.dumps(dict(ok=False, error=message)))


if __name__ == "__main__":
    main()
