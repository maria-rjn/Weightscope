# WeightScope

A local VS Code extension for inspecting model checkpoints and answering **what changed, and are the weights healthy?** Original checkpoints stay unchanged.

Open `.safetensors`, `.gguf`, `.onnx`, `.pt`, or `.pth` files, or select a downloaded Ollama model from its local catalog. Explore a layer tree, shapes, stored parameter counts, dtype distribution, quantization metadata, tensor statistics, and checkpoint differences. Inspection and analysis run on demand. There are no uploads, network requests, or telemetry in the extension.

![Model overview with layer groups and size bars](docs/images/overview.png)

![Checkpoint comparison with a map of changed layers](docs/images/checkpoint-diff.png)

## Install

1. Clone this repository and run `npm ci`, then `npm run package` (Node.js 22 or newer is required for the packaging tools).
2. In VS Code, run **Extensions: Install from VSIX…** and select the generated `artifacts/weightscope-0.3.0.vsix`.
3. Open a supported model file, or run **WeightScope: Inspect Model**.

If another extension owns these file types, use **Reopen Editor With… → WeightScope**. Requires VS Code 1.90 or newer. Works locally and with Remote SSH when the extension and model are on the remote host.

## A simple first workflow

1. Open a checkpoint. **Explore** shows clickable layer groups with size bars.
2. Select a group, then a tensor. Its details open in a focused drawer.
3. Click **Analyze weights** to see the distribution and health check. Advanced statistics stay under **More statistics & sampling details**.
4. Use **Compare checkpoint** to choose a target, then **Analyze weight changes**. A change map shows where differences are concentrated; select a group for individual results.

**Browse tree** opens the tensor name hierarchy. **More columns** reveals exact values and storage bytes. **Tools** contains refresh, JSON export, and Python setup. **Model details** holds full metadata and precision distribution.

## Open downloaded Ollama models

Run **WeightScope: Open Ollama Model** from the Command Palette (`Cmd+Shift+P` on macOS, `Ctrl+Shift+P` on Windows/Linux), then select a model/tag. The picker reads the local catalog and opens the original weights in place. Ollama does not need to be running, and no rename, copy, download, or model execution is needed.

Both GGUF model blobs and newer safetensors tensor blobs are supported. Manifest tensor names are retained, including image model namespaces and shared weight aliases. Local manifest-list references resolve to a downloaded variant, preferring GGUF when available. Model names, tags, registry, prompt template, system prompt, generation parameters, licenses, and small model configurations appear under **Model details → Ollama source & prompt settings**. Additional adapter/projector files are listed separately and excluded from primary weight totals.

The extension uses `weightscope.ollamaModelsPath` if set, then the `OLLAMA_MODELS` environment variable visible to VS Code. Otherwise it searches the default user store (`~/.ollama/models`) and Linux service stores. **Choose Ollama models folder…** in the picker selects a different store for this session; choose the folder containing both `manifests` and `blobs`. Set `weightscope.ollamaModelsPath` to keep that location across restarts. For Remote SSH, the store must be accessible on the extension host. See [Ollama's storage documentation](https://docs.ollama.com/faq#where-are-models-stored).

Run **WeightScope: Compare Ollama Models** to select a baseline and target from the catalog. Inside an inspector, **Tools → Compare Ollama model** or **Compare → Choose Ollama model** compares the current checkpoint with a downloaded model. Regular file checkpoints and Ollama models can be compared with each other.

GGUF numerical analysis uses the supported decoders listed below. Ollama MXFP8 safetensors weights support numerical analysis and checkpoint comparison: four E4M3FN values are unpacked from each U32 word and multiplied by their companion E8M0 scale, shared across 32 weights. Decoded values use float32 rounding, including signed zeros, NaNs, overflow, and underflow. The tensor inspector shows the logical shape and weight count alongside a separate stored-layout disclosure. Model totals, size bars, and tensor tables continue to describe stored entries, including scale codes; they are not unique trainable parameter counts. Other packed schemes (including FP4), missing scales, and incompatible MXFP8 layouts expose metadata with an explicit numerical-decoding explanation. Missing/corrupt downloads, unsupported blobs, and cloud-only entries show explanations. Text/configuration previews are limited to 64 KiB. JSON reports can include local prompts and settings; review them before sharing. Export cannot overwrite the selected manifests or their weight/configuration blobs.

Layer groups follow the first numbered module in tensor names, or the enclosing named module when there is no numbered block. Bars describe stored value counts; they do not infer execution order. Both the optional tree and search retain access to individual tensors.

## What you can do

- Browse the tensor name hierarchy and search all layers, with dtype filtering and paged results.
- See tensor shapes, stored elements, storage bytes, precision distribution, and model metadata.
- Analyze a selected tensor: min, max, mean, population standard deviation, RMS, zero fraction, NaN/Inf counts, and histogram.
- Compare a baseline and target checkpoint by tensor name. Detect added/removed tensors, changed shapes, and changed dtypes immediately.
- Analyze numerical differences for one tensor or a batch: mean absolute delta, maximum absolute delta, RMSE, relative L2, cosine similarity, changed-value count, and non-finite mismatches. The tensor difference drawer and JSON report contain the full metrics.
- Export a JSON report containing metadata, computed statistics, comparison results, coverage, and tolerance.
- View ONNX operators and input/output names.
- Save a validated checkpoint copy under a new name with **Tools → Save checkpoint as…**.

## Save a checkpoint copy

**Tools → Save checkpoint as…** streams a complete copy to a temporary file, validates its tensor structure, and publishes it under a new name. Existing files and aliases are never overwritten. Cancellation and validation failures remove the temporary file. Memory usage stays bounded while copying large models.

Single-file safetensors, GGUF (including downloaded Ollama GGUF blobs), ONNX, and supported PyTorch checkpoints retain their format. PyTorch validation requires the same trusted workspace and interpreter as inspection. ONNX files with external weights must stay in the original directory to preserve relative references; external weights are shared, not copied. Multi-blob Ollama safetensors export is not supported.

## Format support

| Format | Metadata / tree | Numerical analysis | Runtime |
| --- | --- | --- | --- |
| safetensors | Shapes, counts, dtypes, metadata, validated ranges | F64/F32/F16/BF16, supported FP8, integers, boolean | Native Node.js, bundled with VS Code |
| Ollama MXFP8 safetensors | Stored U32 layout and logical shape/count, scale reference, block metadata | Sampled dequantized E4M3FN × E8M0 values and cross-format comparisons | Native Node.js; no MLX/Python dependency |
| GGUF | Little-endian v2/v3, tensor metadata, quantization blocks, effective bits per weight | Plain scalar types, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K | Native Node.js |
| ONNX | Initializers, tensor-valued Constant operators, nested graphs, quantization annotations, metadata, operator list | Common scalar raw and typed storage, external weights confined to the model directory | Native streaming protobuf reader; no Python |
| PyTorch `.pt`/`.pth` | Plain tensors/state dicts, common training checkpoint wrappers | Dense real non-quantized tensors, including non-contiguous layouts | Python with PyTorch **>= 2.6**, trusted workspace |

GGUF IQ and newer schemes expose their type, count, block geometry, and byte size. Their numerical decoders are not included in this release. Unknown GGML types and version-dependent Q8_1 block layouts show unknown byte sizes. Big-endian GGUF and version 1 are unsupported.

ONNX quantized integer statistics describe **stored numeric codes**, not dequantized floating-point weights. Comparisons reject mismatched numerical units. Sparse initializers, segmented tensors, string/complex and newer sub-byte types do not have numerical analysis. Sparse initializers are excluded from totals, with a visible warning. ONNX inspection is a focused reader, not a full ONNX validator or inference engine.

PyTorch inspection does not execute forward passes or accept custom pickled modules/TorchScript archives. Quantized PyTorch tensors are not supported by this release's metadata reader; export them as ONNX or GGUF. Setup: [PyTorch instructions](docs/PYTORCH.md).

## Accurate interpretation

The tree follows tensor names; it does not recover the forward-pass graph from a state dict. Counts are **stored tensor elements**, including buffers and aliases, not guaranteed unique trainable parameters.

Opening native-format checkpoints reads bounded metadata and skips weight payloads. Large ONNX messages are scanned without materializing raw weights. GGUF metadata arrays show the first 32 entries while preserving their full count. PyTorch metadata uses FakeTensorMode and weights-only loading; numerical analysis maps modern ZIP checkpoints on CPU. Legacy non-ZIP PyTorch checkpoints up to 512 MiB may require an eager load when analyzing values.

Statistics use up to 32 evenly spaced contiguous windows across flattened row-major indices, bounded by `weightscope.sampleSize`. Small tensors are analyzed in full. Results explicitly say **Full tensor** or **Sampled estimate**; sampled extrema and health checks can miss outliers outside the sample.

Comparisons use identical logical sample positions across formats. **Sample matches** means no differences were found in analyzed positions; it does not prove whole-tensor equality. **Full match** means every value was compared and agreed within tolerance. The JSON status for a full match remains `equal`. Precision changes remain visible even when numerical values match. NaN/Inf pairs are counted separately. Equal NaNs and equal infinities agree for this comparison policy; mismatched non-finite values mark the tensor changed. Relative L2 is undefined for a zero baseline norm, and cosine is undefined for zero norms.

Files changed since inspection must be refreshed before analysis. No background model scans or weight analysis occur.

## Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `weightscope.pythonPath` | Platform Python | Executable with PyTorch installed; machine-scoped |
| `weightscope.ollamaModelsPath` | Auto-detect | Directory containing Ollama `manifests` and `blobs`; machine-scoped |
| `weightscope.sampleSize` | `32768` | Maximum sampled values per tensor (`256`–`262144`) |
| `weightscope.maxDiffTensors` | `500` | Compatible tensors per batch (`1`–`10000`) |
| `weightscope.absoluteTolerance` | `0` | Absolute numerical difference tolerance |

Use **Analyze next batch** again to continue through large comparisons. Analysis can be cancelled. Reports contain the paths of inspected local files; review them before sharing.

## Develop and verify

The extension has no runtime npm dependencies or build step.

```sh
npm install
npm run check
npm test
npm run demo
npm run preview
npm run package
```

Open this folder in VS Code and press **F5** to start the Extension Development Host. Open one of the generated files in `examples/`. The browser preview at `http://127.0.0.1:4173` runs the same frontend against real synthetic safetensors checkpoints. Add `?ollama=1` for the generated Ollama catalog demo. File dialogs, the catalog picker, export, and cancellation belong to the VS Code integration.

Framework integration verification is optional and uses an isolated environment:

```sh
python3 -m venv .test-runtime
.test-runtime/bin/python -m pip install 'torch>=2.6' onnx safetensors gguf
WEIGHTSCOPE_TEST_PYTHON="$PWD/.test-runtime/bin/python" npm run test:integration
```

On Windows, use the environment's `Scripts/python.exe`. The test suite covers corrupt/truncated files, sampling semantics, cross-format drift, GGUF decoding, external-data path confinement, file mutation, and framework-written checkpoints. See [product direction](docs/PRODUCT.md).

## References

Parsers follow the upstream [safetensors specification](https://github.com/safetensors/safetensors#format), [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [GGML quantization layouts](https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h), and [ONNX protobuf definitions](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3). Ollama MXFP8 decoding follows [Ollama's packing and block parameters](https://github.com/ollama/ollama/blob/main/mlx/quant/quant.go), [companion naming and quantization metadata](https://github.com/ollama/ollama/blob/main/create/quantize.go), and the [OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). VS Code integration uses its [Custom Editor API](https://code.visualstudio.com/api/extension-guides/custom-editors).

MIT licensed. The local package uses a placeholder publisher ID; Marketplace publication requires an owned publisher ID and release metadata.
