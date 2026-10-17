# Changelog

## Unreleased

- Compare GGUF files with the Hugging Face safetensors or PyTorch checkpoint they were converted from. Tensor names are translated to llama.cpp's GGUF names for supported architectures, Q/K rows permuted into llama.cpp's rotary layout are restored using the GGUF head counts, Gemma's `1 + w` norm offset is removed, and stacked GGUF experts are compared per expert. The drawer shows both tensor names and each reversed step; unsupported architectures keep exact-name matching with an explanation.
- Align model layer size bars on a shared scale, with stored-value and storage-byte measures and explicit unknown sizes.
- Replace comparison cards with a searchable, paginated layer-by-tensor-role heatmap of relative L2 drift, preserving structural, precision, non-finite, and sampling states.
- Add labeled SVG distributions, accessible bin-count tables, and baseline/target overlays computed with shared bins from the existing samples.
- Improve narrow layouts, keyboard pagination, chart theme colors, and cleanup when inspectors close or refresh.

- Decode Ollama MXFP8 packed weights with E4M3FN values and companion E8M0 scales for bounded statistics, histograms, and checkpoint comparison.
- Show logical dimensions and weight counts in the tensor inspector while preserving stored shapes, counts, and dtypes in model summaries and exports.
- Honor per-tensor quantization metadata in mixed blobs and keep scale tensors and unquantized floating-point tensors available for inspection.
- Explain missing scales, incompatible block layouts, and unsupported packed formats without computing misleading statistics.

## 0.3.0

- Add Open Ollama Model and Compare Ollama Models commands with searchable local model/tag pickers and custom store selection.
- Inspect original GGUF hash blobs and aggregate newer safetensors tensor blobs without copying, renaming, or running models.
- Preserve manifest tensor names, shared aliases, local variant references, and model/tag identity when restoring editors.
- Display Ollama prompt templates, system prompts, generation parameters, licenses, small model configurations, and additional weight files under Model details.
- Identify packed safetensors quantization with explicit limits on numerical analysis and stored-count interpretation.
- Support `OLLAMA_MODELS` and the machine-scoped `weightscope.ollamaModelsPath` setting.
- Protect selected Ollama manifests and associated blobs from report overwrites and detect changes before numerical analysis.

## 0.2.0

- Replace the initial dense table with clickable layer groups and relative size bars.
- Show tensor details and weight distributions in a focused, keyboard-accessible drawer.
- Keep the name tree, exact table columns, advanced statistics, and metadata available on request.
- Add a checkpoint change map with separate states for changes, pending analysis, full matches, sample matches, and unavailable analysis.
- Start comparisons with a visual baseline/target workflow; keep tensor results collapsed until requested.
- Preserve visibility of precision changes even when numerical values match.
- Move refresh and export into the Tools menu and simplify the header.

## 0.1.0

Initial local model inspector with native safetensors, GGUF and ONNX readers, restricted PyTorch loading, on-demand tensor diagnostics, numerical checkpoint comparison, and JSON export.
