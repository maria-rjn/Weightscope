# Changelog

## Unreleased

- Add a **Fingerprint** view to the layer drift map. It draws every layer group (columns) and tensor role (rows) as one dense canvas image, with hover details, arrow-key navigation, and a screen-reader description for each cell. The **Table** view keeps written values.
- Add a **Log scale** for relative L2 drift, on by default, spanning up to six decades so small changes stay visible next to a large outlier. **Linear scale** remains available, and both stay shared across pages and filters.
- Color drift with a perceptually uniform ramp (reversed in light themes) in both views, and give exact zero drift its own neutral color.
- Open **Explore** on a **Model map**: one cell per tensor, layer groups as columns and tensor roles as rows, colored by precision (with bits per value) or stored values without reading weights. **Scan weights** samples every tensor once on request, streams progress, and colors the map by weight RMS or largest magnitude; scanned statistics also appear in the tensor inspector. Layer size bars remain below the map.
- Share one canvas grid between the drift fingerprint and the model map, including keyboard navigation and hover details.
- Replace the README screenshots with images of real models: a SmolLM2-135M base-vs-instruct drift fingerprint, and the model map, RMS scan and weight image of Gemma 3 4B Q4_K_M. Add `npm run screenshots`, which regenerates them from the browser preview with headless Chrome and no new dependencies.
- Add **Draw weight image** to the tensor inspector and **Draw change image** to the tensor difference drawer. Each draws the tensor (or target minus baseline) as up to 192 × 192 tiles colored by RMS or signed peak, with hover readouts, clipped or full color range, and the location of the largest value. Images read up to 4,194,304 values (262,144 for PyTorch) as evenly spaced runs of whole rows and state their coverage; change images use the same translated positions as GGUF comparisons.

## 0.4.2

- Add a WeightScope Activity Bar icon with a **Start** view. It has buttons to inspect a model file, open an Ollama model, compare checkpoints or Ollama models, and set up PyTorch, so you can start without opening a model first.

## 0.4.1

- Use the Matrix Lens icon for the extension listing, inspector header, and editor tab.
- Name Safetensors and GGUF in the extension display name and improve model-inspection search keywords.
- Exclude local development worktrees from the packaged extension.

## 0.4.0

- Compare GGUF files with the Hugging Face safetensors or PyTorch checkpoint they were converted from. Tensor names are translated to llama.cpp's GGUF names for supported architectures, Q/K rows permuted into llama.cpp's rotary layout are restored using the GGUF head counts, Gemma's `1 + w` norm offset is removed, and stacked GGUF experts are compared per expert. The drawer shows both tensor names and each reversed step; unsupported architectures keep exact-name matching with an explanation.
- Open sharded Hugging Face safetensors checkpoints as one model from the `.safetensors.index.json` index or any `-0000X-of-0000N` shard, including Hugging Face cache snapshots. Totals, layer groups, statistics, and drift comparisons cover every shard.
- Validate index weight maps against shard headers. Report missing shards, misplaced or duplicate tensors, and file names outside the model folder. Group complete shard sets that have no index by file name.
- Show shard counts, a per-shard table, and each tensor's shard. Detect index and shard changes before analysis, and protect every shard from report overwrites.
- Open sharded PyTorch checkpoints (`pytorch_model-0000X-of-0000N.bin` with `pytorch_model.bin.index.json`) as one model from the index or any `.bin` shard, with the same validation, fallback, change detection, and report protection as safetensors shards. Shards are read through the restricted PyTorch helper in trusted workspaces, and each tensor is sampled from its own shard.
- Use only an index of the shard's own format when safetensors and `.bin` shards share a folder. Name the shard that could not be read when a set fails to open.

- Align model layer size bars on a shared scale, with stored-value and storage-byte measures and explicit unknown sizes.
- Replace comparison cards with a searchable, paginated layer-by-tensor-role heatmap of relative L2 drift, preserving structural, precision, non-finite, and sampling states.
- Add labeled SVG distributions, accessible bin-count tables, and baseline/target overlays computed with shared bins from the existing samples.
- Improve narrow layouts, keyboard pagination, chart theme colors, and cleanup when inspectors close or refresh.

- Decode Ollama MXFP8 packed weights with E4M3FN values and companion E8M0 scales for bounded statistics, histograms, and checkpoint comparison.
- Show logical dimensions and weight counts in the tensor inspector while preserving stored shapes, counts, and dtypes in model summaries and exports.
- Honor per-tensor quantization metadata in mixed blobs and keep scale tensors and unquantized floating-point tensors available for inspection.
- Explain missing scales, incompatible block layouts, and unsupported packed formats without computing misleading statistics.

## 0.3.1

- Open Ollama tag manifests directly in the WeightScope inspector instead of the JSON editor.
- Document how to switch an already-open JSON manifest to the inspector.

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
