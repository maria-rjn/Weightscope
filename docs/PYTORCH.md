# PyTorch inspection setup

safetensors, GGUF and ONNX use native readers with no Python dependencies. For `.pt`/`.pth`, install a maintained PyTorch version **>= 2.6** in your chosen Python environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install 'torch>=2.6'
```

In VS Code settings, set `weightscope.pythonPath` to the full path of `.venv/bin/python`. On Windows, use `.venv\Scripts\python.exe`. If the setting is empty, the extension tries `python3`, or `py -3` on Windows. For Remote SSH, install Python/PyTorch on the remote host and use that host's executable path.

Use a trusted workspace. The helper uses `torch.load(..., weights_only=True, map_location='cpu')` and **never falls back to unrestricted pickle loading**. Metadata inspection uses PyTorch FakeTensorMode to avoid allocating complete tensor storage. Modern ZIP checkpoints use memory mapping. Each operation runs in a cancellable Python subprocess with a two-minute timeout.

Supported checkpoint layouts include a plain tensor, a plain state dict, or dictionaries containing `state_dict`, `model_state_dict`, or a mapping-valued `model`. Optimizer slots in other sections are excluded when a recognized model wrapper exists. Dense non-contiguous tensors are sampled at logical row-major indices without copying their entire storage.

If a checkpoint contains custom classes, TorchScript code, sparse/complex/quantized tensors, or objects incompatible with restricted loading, export a plain dense state dict or safetensors checkpoint in its trusted training environment. Restricted loading reduces code-execution risks; use a current patched PyTorch release for files from unfamiliar sources. No model functions are called.
