# Changelog

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
