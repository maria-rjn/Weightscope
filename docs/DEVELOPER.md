# WeightScope developer README

This guide covers developing, testing, and packaging WeightScope. For operating the extension, use the [user guide](USER_GUIDE.md). The [project README](https://github.com/maria-rjn/Weightscope#readme) is the project entry point.

## Requirements

| Component | Requirement |
| --- | --- |
| Node.js | 22 or newer for the packaging tools. |
| npm | Install development dependencies from the committed lockfile with `npm ci`. |
| VS Code | 1.90 or newer for the extension and Extension Development Host. |
| Python | Optional for native formats; needed for PyTorch operations and framework integration tests. |
| PyTorch | 2.6 or newer, with a trusted workspace for extension-host operations. |

The extension uses CommonJS JavaScript and has no build step or runtime npm dependencies. VS Code supplies its Node.js runtime. `@vscode/vsce` is a development dependency used for packaging.

## Quick start

Clone the repository, then work from its root directory:

```sh
git clone https://github.com/maria-rjn/Weightscope.git
cd Weightscope
npm ci
npm run check
npm test
npm run demo
```

If you are testing an unmerged feature, check out its branch before running these commands. Generated checkpoints appear in `examples/` and are ignored by Git. They contain synthetic data and are not trained models.

Open the repository in VS Code and press **F5**. The supplied **Run WeightScope** launch configuration opens an Extension Development Host. In that window, open an example checkpoint or run **WeightScope: Inspect Model**.

No compilation or file watcher is required. Reload the development host after changing extension-host code. Reopen or reload the custom editor when checking frontend changes.

## Repository map

| Path | Responsibility |
| --- | --- |
| `package.json` | Commands, file associations, settings, runtime capabilities, and npm scripts. |
| `src/extension.js` | Activation, custom editor, dialogs, webview messages, staged edits, progress, and report export. |
| `src/backend.js` | Inspection, summaries, sampling, numerical comparisons, file identity checks, and Python requests. |
| `src/binary.js` | Bounded reads, scalar decoding, count validation, cancellation, and sample windows. |
| `src/parsers/` | Native safetensors, GGUF, and streaming ONNX readers. |
| `src/ollama.js` | Store discovery, manifests, catalog entries, blob resolution, source settings, and protected paths. |
| `src/quantization/mxfp8.js` | Packed Ollama MXFP8 layout validation and dequantized sampling. |
| `src/checkpoint.js` | Save As transactions, format constraints, bounded copying, validation, and exclusive publication. |
| `src/metadata.js` | Metadata edit validation and native-format metadata writers. |
| `src/tensor-edits.js` | Tensor capabilities, replacement compatibility, native transformations, and floating-point encoding. |
| `src/math.js` | Statistics, structural differences, numerical metrics, and metadata differences. |
| `python/inspect_pt.py` | Restricted CPU loading, PyTorch inspection/sampling, and Save As tensor edits. |
| `media/app.js` | Production webview UI and host-message handling. |
| `media/model-view.js` | Layer grouping and change-map helpers shared with Node tests. |
| `media/styles.css` | Webview layout, theme integration, and responsive styles. |
| `test/` | Native fixtures, unit tests, mocked VS Code host tests, and framework tests. |
| `scripts/` | Syntax/manifest checks, synthetic checkpoints, browser preview, packaging, and framework fixtures. |
| `docs/` | User/developer guides, product direction, and verification notes. |

## Runtime flow

The workspace extension runs on the host that owns the model files. With Remote SSH, that is the remote host. The webview sends structured messages to the extension; filesystem access, dialogs, and Python execution belong to the extension host.

1. The webview sends `ready`. The host inspects metadata and returns the public model, settings, and edit state.
2. `stats` requests deterministic samples for a selected tensor. Numerical payloads are read on demand.
3. `compare` or `compareOllama` selects a target. Structural rows and metadata differences are available before numerical analysis.
4. `analyzeDiff` compares a selected tensor or the next batch, returning `diffRow` and `diffDone` messages.
5. `stageMetadata` and `stageTensor` update in-memory edit maps. Replacement selects its donor through host-owned dialogs.
6. `saveCheckpoint` chooses a new destination and runs the Save As transaction. `checkpointSaved` exposes the resulting file in the UI.
7. `compareSaved` reloads that saved file as the target. `openSaved` opens a separate inspector.

Internal models include storage descriptors, a tensor lookup map, and file identities. `backend.publicModel()` omits private storage details. The extension adds editing capabilities to the webview payload. Keep filesystem access and donor selection under host control; saved comparison uses the host's last saved URI.

The custom document has no in-place save, undo stack, or backup implementation. Staged edits live in the editor's host-side closure and are lost when the inspector closes. **Save checkpoint as…** is the explicit write path; ordinary VS Code Save does not apply the plan. Analysis continues to use the original model after staging or saving edits.

## Reader and comparison contracts

Native inspection reads bounded metadata and skips tensor payloads. Numerical operations use up to 32 evenly spaced contiguous windows over logical row-major indices, bounded by `sampleSize`. Cross-format comparisons use identical logical sample positions.

Preserve these distinctions when changing code or labels:

- Counts describe stored elements, including buffers and aliases, rather than guaranteed unique trainable parameters.
- Tensor names define the hierarchy. A state dict does not supply a recovered execution graph.
- `equal` means a full numerical comparison agreed within tolerance; `sample-match` applies only to sampled positions.
- Precision changes remain visible even if numerical values match.
- Stored quantization codes and dequantized weights have different numerical units. Incompatible bases must not be compared as equivalent.
- GGUF metadata arrays expose previews of up to 32 entries. Metadata comparison uses those previews.
- Unknown layouts expose limitations. Do not infer a byte size, decoder, or arithmetic capability from a similar format name.

The PyTorch helper uses `weights_only=True`, CPU loading, and FakeTensorMode for metadata. It rejects TorchScript and never falls back to unrestricted pickle loading. Requests run in cancellable subprocesses with a 120-second timeout. See [PyTorch setup](PYTORCH.md).

## Save As and editing contracts

`checkpoint.saveCheckpoint(model, destination, options)` coordinates saving. Optional `metadata` and `tensors` lists describe the plan. Replacement carries an inspected donor model and tensor name inside the host; the public plan contains review information.

A native save writes a temporary file in the destination directory, copies data in bounded chunks, applies supported edits, syncs and reparses the result, and verifies tensor names, shapes, and dtypes. Publication uses an exclusive hard link, so existing destination files and aliases are not overwritten. Temporary files are removed on cancellation and failure. The destination filesystem must support hard links.

The temporary and published files share the same data during publication; a second complete output copy is not required. The output still needs disk space for a full checkpoint. Structural validation does not establish inference correctness or model quality.

| Area | Contract to preserve |
| --- | --- |
| File identity | Recheck the source and replacement donors while they are used. Changed inputs require refresh or reselection. |
| Metadata | Safetensors/ONNX property values are strings. GGUF preserves existing scalar types and copies complete untouched entries, including arrays. |
| GGUF layout | Recompute header padding while retaining tensor-relative offsets. `general.alignment` cannot be edited. |
| Tensor plans | One operation per tensor, applied to original values. Restaging replaces its previous operation. |
| Replacement | Match shape, dtype, byte length, and encoded layout. PyTorch donors must also be PyTorch checkpoints. Companion scales are not copied implicitly. |
| Arithmetic | Native F64/F32/F16/BF16 and supported GGUF blocks only. F16/BF16 writes round to nearest, ties to even. Reject non-finite scaling and overflow. |
| ONNX | Edit supported inline payloads. External references remain relative to the original directory; external weight tensors are not modified. |
| Ollama | Single GGUF blobs can be exported. Aggregated safetensors collections are not writable as one checkpoint. Preserve store manifests/blobs. |
| PyTorch | Reserialize with restricted CPU loading. Preserve wrappers, scalar metadata, optimizer state, and unselected tensors; give edited tensors separate storage. This can need substantial RAM. |
| Reports | Protect inspected files, selected donors, saved checkpoints, and protected Ollama paths against report overwrites, including aliases. |

Use the [user support tables](USER_GUIDE.md#format-and-editing-support) as the capability reference. Update the UI explanations and both guides if those boundaries change.

## Commands for development

| Command | Result |
| --- | --- |
| `npm ci` | Install the lockfile's development dependencies. |
| `npm run check` | Check JavaScript syntax, manifest fields, and required assets. |
| `npm test` | Run Node tests; framework tests skip without their configured interpreter. |
| `npm run demo` | Generate small native checkpoints in `examples/`. |
| `npm run preview` | Start the production frontend with a synthetic local backend at `http://127.0.0.1:4173`. |
| `npm run test:integration` | Run framework tests; requires `WEIGHTSCOPE_TEST_PYTHON` to actually execute them. |
| `npm run package` | Build `artifacts/weightscope-<version>.vsix`, using the version in `package.json`. |

### Browser preview

```sh
npm run preview
```

Open `http://127.0.0.1:4173`. Add `?ollama=1` for a generated Ollama catalog entry. The server generates its own demo checkpoints and binds to localhost. Select another port with:

```sh
WEIGHTSCOPE_PREVIEW_PORT=4174 npm run preview
```

The preview can stage edits, save generated safetensors files under `output/preview/`, and compare them. Replacement uses the predefined demo target and matching tensor name. Edit state is shared within the preview server process; use it as a development fixture.

Real catalog/file pickers, opening a saved VS Code editor, report export, cancellation, workspace trust, and Remote SSH need extension-host verification. The preview is not a complete host substitute. Stop it with `Ctrl+C`.

## Testing

Use checks relevant to the change. Parser/writer changes need malformed-input and round-trip coverage; precision or quantization changes need an independent numerical reference; host workflows belong in `test/extension.test.js`.

| Tests | Main coverage |
| --- | --- |
| `backend.test.js` | Native readers, decoding, sampling, metrics, malformed inputs, file changes, and trust. |
| `ollama.test.js`, `mxfp8.test.js` | Catalog/manifests, aliases, source details, packed layouts, scale companions, and mutation. |
| `checkpoint.test.js` | Copies, original preservation, exclusive destinations, cleanup, and ONNX references. |
| `metadata.test.js`, `metadata-diff.test.js` | Round trips, GGUF types, preserved arrays/payloads, validation, and differences. |
| `tensor-edits.test.js` | Combined edits, compatibility, rounding, quantized arithmetic, chunk boundaries, and rejection. |
| `extension.test.js` | Activation, messages, dialogs, staging, saving, saved comparison, reports, and alias protection with a mocked host. |
| `model-view.test.js` | Grouping, change-map states, precision changes, and filtering. |
| `integration.test.js` | Upstream readers/writers, ONNX checker, GGUF dequantizers, PyTorch rounding, aliases/wrappers, and restricted loading. |

### Framework tests: macOS/Linux

```sh
python3 -m venv .test-runtime
.test-runtime/bin/python -m pip install 'torch>=2.6' onnx safetensors gguf
WEIGHTSCOPE_TEST_PYTHON="$PWD/.test-runtime/bin/python" npm run test:integration
```

To run the entire suite with framework checks enabled:

```sh
WEIGHTSCOPE_TEST_PYTHON="$PWD/.test-runtime/bin/python" npm test
```

### Framework tests: Windows PowerShell

```powershell
py -3 -m venv .test-runtime
.\.test-runtime\Scripts\python.exe -m pip install 'torch>=2.6' onnx safetensors gguf
$env:WEIGHTSCOPE_TEST_PYTHON = (Resolve-Path .\.test-runtime\Scripts\python.exe).Path
npm run test:integration
```

This environment variable configures tests. It does not set the extension's `weightscope.pythonPath`; configure that setting separately in the development host for PyTorch UI flows.

### Manual verification

In an Extension Development Host, inspect a synthetic checkpoint, analyze a tensor, compare the demo pair, and export a report. For editing changes, stage an edit, save under a fresh name, compare the result, and open it in another inspector. Check cancellation and unavailable operations when changing those paths.

For frontend changes, also check a narrow window, keyboard access, and the browser console. Browser checks and mocked-host tests do not replace a real Extension Development Host or a live Remote SSH check. See [validation notes](VALIDATION.md) for recorded evidence and limits.

## Packaging and installation

```sh
npm run check
npm test
npm run package
```

Install `artifacts/weightscope-<version>.vsix` with **Extensions: Install from VSIX…**. The version comes from `package.json`; building a branch does not give it a different version automatically.

`.vscodeignore` excludes development dependencies, tests, scripts, generated checkpoints, local environments, output files, and screenshots. Runtime JavaScript, the Python helper, styles, and Markdown guides are included. Keep user documentation available inside the package, and verify its file list after changing exclusions. The packaged extension needs no npm installation on the user's machine.

Packaging creates a local installable artifact. The placeholder publisher ID does not publish it to a Marketplace. Before a public release, use an owned publisher ID, review version/changelog metadata, and rerun relevant verification.

## References and design context

Native readers follow the [safetensors format](https://github.com/safetensors/safetensors#format), [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [GGML block layouts](https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h), and [ONNX protobuf definitions](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3).

Ollama MXFP8 follows [packing and block parameters](https://github.com/ollama/ollama/blob/main/mlx/quant/quant.go), [companion naming and quantization metadata](https://github.com/ollama/ollama/blob/main/create/quantize.go), and the [OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). VS Code integration uses the [Custom Editor API](https://code.visualstudio.com/api/extension-guides/custom-editors).

For product scope and candidate work, see [product direction](PRODUCT.md). User workflows belong in the [user guide](USER_GUIDE.md).
