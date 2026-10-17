# Product direction and demand

## Assessment

There is evidence of demand for local model inspection, but no validated evidence yet for willingness to pay for WeightScope. Existing products already serve this category: [Netron](https://github.com/lutzroeder/netron), [ONNX Model Inspector](https://marketplace.visualstudio.com/items?itemName=JackMa-Tools.onnx-model-inspector), [H5 Viewer](https://marketplace.visualstudio.com/items?itemName=meijiahao.vscode-h5-viewer), and [Weightless](https://marketplace.visualstudio.com/items?itemName=kipngenokoech.weightless). The defensible opportunity is a focused debugging workflow, not format support alone. Assessment researched on October 14, 2026.

## Target workflow

An ML engineer receives a checkpoint, verifies its structure and precision, investigates whether a training run changed the intended layers, and checks whether problematic tensors contain invalid or unusual weights. Quantization users can inspect storage schemes and compare compatible decoded weights.

The product promise is: **Understand a checkpoint and its drift without writing a notebook, uploading a model, or loading all its weights.**

## Implemented first release

1. Local read-only custom editor for five file extensions across four format families.
2. Fast metadata inspection, tensor name hierarchy, dtype summaries, quantization details, and searchable paged tables.
3. On-demand bounded tensor diagnostics with visible sampling coverage.
4. Structural comparison followed by per-tensor or batched numerical drift analysis.
5. Exportable JSON evidence and a remote-extension-host architecture.

Version 0.2.0 simplifies the first-use workflow: visual layer groups lead into tensor inspection, a focused drawer holds diagnostics, and a change map precedes detailed comparison rows. Advanced statistics, raw metadata, the tensor tree, and extra table columns remain available on request.

Version 0.3.0 adds local Ollama catalog discovery, original GGUF blob inspection, aggregation of safetensors tensor blobs, and catalog-based comparison. Prompt settings and source metadata stay inside Model details. Packed tensor formats expose storage metadata with explicit numerical-decoding limits.

The visualization update replaces overview cards with aligned size bars and comparison cards with a layer-by-tensor-role matrix. Roles are literal suffixes of tensor names, with a generic fallback and collision-preserving cells; they do not imply a model architecture or execution graph. Relative L2 uses one scale across all pages and filters, while coverage, precision, structural changes, and unavailable analysis retain separate labels. Tensor distributions have labeled axes, accessible bin counts, and shared-bin baseline/target overlays based on existing bounded samples. All rendering stays local with no runtime plotting dependencies.

Sharded Hugging Face safetensors checkpoints open as one model from their index or any shard, so drift analysis works on multi-gigabyte models without merging files. Index weight maps are checked against shard headers; inconsistent sets fail explicitly instead of silently comparing a partial model.

Explicit boundaries keep results trustworthy: name hierarchy is not a recovered execution graph; stored-element counts are not unique trainable counts; sample matches are not exact equality. Unsupported numerical decoders produce clear explanations rather than fabricated values.

## Candidate next releases

- GGUF IQ and newer quantization decoders.
- LoRA-aware comparison and configurable tensor-name mapping.
- Full-tensor streaming comparisons and checksums for exact verification.
- ONNX graph visualization, sparse tensors, and dequantized numerical analysis.
- Persistent worker processes for repeated PyTorch operations.
- CSV reports and experiment-run links.

## Validate demand before investing further

Recruit 8–12 ML engineers who train, fine-tune, or quantize models. Ask them to inspect two of their own checkpoints and observe their workflow. Measure time to find a changed layer, repeated usage, and whether reports help resolve actual debugging questions. Distribution can begin with an open-source core and a Marketplace release after an owned publisher ID is configured. Paid features should follow demonstrated repeat use and explicit willingness to pay, rather than assuming demand from competitor existence.
