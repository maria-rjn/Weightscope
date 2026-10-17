# WeightScope user guide

WeightScope helps you inspect a local model checkpoint, investigate its weights, make supported edits in a new file, and compare the saved result with the original. The extension keeps the original checkpoint unchanged.

This guide describes the current source version. If an installed package has fewer controls, obtain a VSIX built from a source revision that includes them. Developers can use the [developer README](DEVELOPER.md) to build one.

## Contents

- [Install and open your first checkpoint](#install-and-open-your-first-checkpoint)
- [Find a layer and inspect its weights](#find-a-layer-and-inspect-its-weights)
- [Understand counts and sampled results](#understand-counts-and-sampled-results)
- [Compare two checkpoints](#compare-two-checkpoints)
- [Save a checkpoint copy](#save-a-checkpoint-copy)
- [Edit metadata](#edit-metadata)
- [Replace, scale, or zero a tensor](#replace-scale-or-zero-a-tensor)
- [Compare and open the saved result](#compare-and-open-the-saved-result)
- [Open downloaded Ollama models](#open-downloaded-ollama-models)
- [Export a report](#export-a-report)
- [Format and editing support](#format-and-editing-support)
- [Settings](#settings)
- [Troubleshooting](#troubleshooting)

## Install and open your first checkpoint

You need VS Code 1.90 or newer and a WeightScope `.vsix` package.

1. Open the Command Palette: `Cmd+Shift+P` on macOS or `Ctrl+Shift+P` on Windows/Linux.
2. Run **Extensions: Install from VSIX…** and select the package.
3. Run **WeightScope: Inspect Model** and select a `.safetensors`, `.gguf`, `.onnx`, `.pt`, or `.pth` file.

You can also open a supported file from the Explorer. If it opens in another editor, choose **Reopen Editor With… → WeightScope**.

Safetensors, GGUF, and ONNX work without Python. For `.pt`/`.pth`, follow [PyTorch setup](PYTORCH.md) and use a trusted workspace. WeightScope reads model data without running a forward pass.

For Remote SSH, install the extension on the remote host. Model files, the Ollama store, and any Python executable must be accessible there. Virtual filesystem documents are unsupported.

### Where to find things

| Area | Use it for |
| --- | --- |
| **Explore** | Layer groups, tensor search, precision filters, and tensor details. |
| **Compare** | Baseline/target selection, the change map, metadata differences, and numerical comparisons. |
| **Model details** | Precision distribution, metadata, metadata editing, and Ollama source/prompt settings. |
| **Operators** | ONNX operators and their input/output names. Appears for ONNX models. |
| **Tools** | Ollama actions, Save As, refresh, JSON export, and PyTorch setup. |
| **Pending edits** | Review staged changes, save them to a new checkpoint, or discard the plan. Appears after staging an edit. |
| **Last saved checkpoint** | Compare the most recently saved file with the original or open another inspector. Appears after saving. |

## Find a layer and inspect its weights

1. In **Explore**, choose a layer group.
2. Select a tensor to open its drawer.
3. Click **Analyze weights**.
4. Read the distribution, range, mean, standard deviation, and NaN/Inf health result.

Use **Search tensors…** to find a name and the precision filter to narrow the list. **Browse tree** opens the tensor-name hierarchy. **More columns** shows additional exact counts and storage sizes. **More statistics & sampling details** contains RMS, zero fraction, finite counts, and coverage information.

Layer groups and size bars follow tensor names and stored value counts. They do not show the model's execution order.

## Understand counts and sampled results

A **tensor** is a stored array of values. Its **shape** gives its dimensions, and its **dtype** describes its stored precision or encoding. Stored counts can include buffers, aliases, and quantization scales; they are not a guaranteed count of unique trainable parameters.

| Label | Meaning |
| --- | --- |
| **Full tensor** | Every value was included in this tensor's statistics. |
| **Sampled estimate** | Statistics describe selected windows of values. Outliers elsewhere can be missed. |
| **Full match** | Every value was compared and agreed under the comparison policy and tolerance. |
| **Sample matches** | No differences were found at the sampled positions. Other positions remain unchecked. |
| **Not analyzed** | The structure is known, but numerical values have not been compared yet. |
| **Unavailable** | The operation failed or the storage cannot be decoded for numerical comparison. Open the row for its explanation. |

The default is at most 32,768 sampled values per tensor. Small tensors are analyzed in full. A clean sampled health result means no NaNs or infinities were found in that sample.

## Compare two checkpoints

1. Open the checkpoint you want to use as the **baseline**.
2. Click **Compare checkpoint** and choose the **target**.
3. Review structural differences: added/removed tensors, changed shapes, and precision changes.
4. Click **Analyze weight changes** to calculate numerical differences.
5. Select a group in the change map, or open **Tensor-by-tensor results** and select a tensor.

Differences are target minus baseline. Tensor names must match exactly; WeightScope does not map renamed layers automatically. Comparisons can cross supported file formats when tensor shapes and numerical units are compatible.

The detail drawer includes mean/max absolute change, RMSE, cosine similarity, relative L2, changed-value counts, non-finite mismatches, coverage, and tolerance. **Analyze next batch** continues through large models. Individual tensors can be analyzed separately. Use **Cancel** to stop an active analysis.

**Metadata differences** lists added, removed, and changed values separately from weights. Metadata arrays use their displayed previews, so differences beyond those previews are not detected. Precision changes stay visible even when numerical values match. Pairs of NaNs and equal infinities agree under the comparison policy; mismatched non-finite values count as changes. Relative L2 and cosine are undefined for zero norms.

You can also start with **WeightScope: Compare Checkpoints** in the Command Palette and choose both files.

## Save a checkpoint copy

Choose **Tools → Save checkpoint as…**, then select a new filename with the model's format extension. You can save an unchanged copy or a copy with staged edits.

WeightScope validates saved tensor names, shapes, and dtypes before publishing the output. Existing files, including aliases of existing files, are never overwritten. A failed or cancelled save removes its temporary file. Keep enough free disk space for a full new checkpoint.

Single-file safetensors, GGUF, ONNX, and supported PyTorch files retain their format. Changing the filename extension does not convert the model. For PyTorch, Save As validation needs the configured Python interpreter and a trusted workspace.

ONNX models with external weights must be saved in their original directory. Their external files are shared with the original model and are not copied or edited. Ollama safetensors collections cannot be saved as one consolidated checkpoint.

Validation checks checkpoint structure. It does not establish that an edited model will run correctly or produce a desired response. WeightScope does not train or evaluate models.

## Edit metadata

1. Open **Model details → Edit metadata**.
2. Enter a metadata key and value.
3. For a new GGUF key, select its type.
4. Click **Stage metadata edit**, or **Stage removal** to delete a key from the saved copy.
5. Review the pending list and use **Save checkpoint as…**.

| Format | What you can enter |
| --- | --- |
| Safetensors | String-to-string metadata. Enter text without JSON quotes. |
| ONNX | String metadata properties. Producer fields, graph names, and opsets are separate and remain unchanged. |
| GGUF | Scalar values. Existing keys retain their stored type; new keys use the type you select. |
| PyTorch | Metadata editing is unavailable. Scalar metadata is retained when saving tensor edits. |

Examples: set `description` to `Reviewed checkpoint`, set GGUF `general.name` to a model name, or add a Boolean GGUF key such as `custom.reviewed` with value `true`.

For GGUF numeric types, enter JSON numbers such as `42` or `0.5`. Booleans use `true` or `false`. For 64-bit integers, enter exact decimal digits; do not use scientific notation or quotation marks. Existing arrays are shown as previews and cannot be rewritten from the form. Untouched arrays, including complete tokenizer arrays, are preserved. `general.alignment` cannot be edited because it controls tensor storage.

These are checkpoint metadata edits. Ollama prompt templates, system prompts, and generation settings shown in source details are not edited by this form.

## Replace, scale, or zero a tensor

1. Select a tensor in **Explore**.
2. Find **Edit tensor** in its drawer.
3. Choose an available operation.
4. Enter a scale factor if needed, then click **Stage tensor edit**.
5. Review the pending plan and save it to a new checkpoint.

| Operation | What the saved copy receives |
| --- | --- |
| **Scale weights** | The original tensor multiplied by a finite factor, such as `0.5`. |
| **Set weights to zero** | Zero weights for the selected tensor, including clearing non-finite values where supported. |
| **Replace from checkpoint** | A compatible tensor from another checkpoint. You choose its file and then its tensor name. |

Replacement requires the same shape, dtype, byte length, and compatible storage layout. PyTorch replacement requires another PyTorch checkpoint. Target quantization metadata is retained; companion scale tensors are not copied automatically. If the donor changes after selection, select it again.

The drawer explains unavailable operations. Scaling/zeroing support ordinary F64/F32/F16/BF16 weights and the GGUF blocks listed in the [support section](#format-and-editing-support). Integer and FP8 inline tensors support compatible replacement, without arithmetic. Scaling rejects non-finite inputs or results that overflow the stored precision. Supported GGUF quantized tensors are transformed through block scales and offsets, retaining their encoded codes and quantization precision.

PyTorch edits load and resave data on CPU and may need substantial RAM. Checkpoint wrappers, scalar metadata, optimizer state, and unselected tensors are retained. An edited tensor gets separate storage so unselected aliases keep their original values.

### How staged edits behave

Each tensor has one staged operation, applied to its original values. Staging another operation for that tensor replaces the previous one. For example, scaling an original value of `8` by `0.5` produces `4`; staging `0.5` again still produces `4`.

Statistics and the current inspector's tensor details continue to describe the original checkpoint. Use a saved-file inspector to analyze modified values. Staged edits remain after saving so you can revise the plan or save it again. **Discard pending edits** clears the whole plan.

Pending edits are kept only while the inspector remains open. There is no autosave or persistent edit history. Use **Save checkpoint as…** to write them; closing the inspector loses the pending plan. Refresh requires discarding it first.

## Compare and open the saved result

After saving, find **Last saved checkpoint**:

1. Click **Compare saved with original**.
2. Review metadata and structural differences.
3. Click **Analyze weight changes** for numerical results.
4. Click **Open saved checkpoint** to inspect and analyze the modified file in a separate editor.

The original remains the baseline. Comparison reloads the saved file from disk, so later unsaved plan changes are not part of that target. To apply a further transformation to already modified values, open the saved checkpoint and stage the next operation there.

## Open downloaded Ollama models

Run **WeightScope: Open Ollama Model** from the Command Palette and choose a downloaded model/tag. Ollama does not need to be running. WeightScope reads original local blobs without downloading or executing the model.

The store is selected in this order: `weightscope.ollamaModelsPath`, then `OLLAMA_MODELS` as seen by VS Code, then the default user store (`~/.ollama/models`) and Linux service stores. Use **Choose Ollama models folder…** to choose another store for the session; select the folder containing both `manifests` and `blobs`. Set the VS Code setting to retain that location across restarts.

**Model details → Ollama source & prompt settings** shows source identity, available prompt templates/system prompts, generation parameters, licenses, and small configurations. Text previews are limited to 64 KiB. Additional adapter/projector files are listed separately and excluded from primary weight totals. Local manifest-list variants resolve to downloaded data, preferring GGUF when available.

Run **WeightScope: Compare Ollama Models** to choose two catalog entries. Within an inspector, use **Tools → Compare Ollama model** or **Compare → Choose Ollama model**. File checkpoints and Ollama entries can also be compared.

Single GGUF entries can be saved and edited as new standalone GGUF files. Saving does not replace a store blob, update a manifest, or change an Ollama model/tag. Loading the new file into a runtime is a separate step. Ollama safetensors collections remain inspect-only, including collections with supported MXFP8 numerical decoding.

## Export a report

Choose **Tools → Export JSON report**. The report contains model metadata, computed statistics, comparison rows, metadata differences, tolerance, and sampling coverage. Numerical results are included for analyses you have requested; export does not run a new scan.

A report is not a modified checkpoint and does not save the pending edit plan. Use Save As for model files. Export protects inspected checkpoints, current replacement sources, and checkpoints saved by the inspector from overwrites.

Reports contain local paths and can include Ollama prompts and settings. Review that content before sharing. The extension itself makes no uploads, network requests, or telemetry calls.

## Format and editing support

### Inspection and numerical analysis

| Format | Inspection | Numerical analysis | Extra runtime |
| --- | --- | --- | --- |
| Safetensors | Tensor names, shapes, stored counts, dtypes, and metadata. | F64/F32/F16/BF16, supported FP8, integers, and Boolean storage. | None. |
| GGUF | Little-endian v2/v3 metadata, shapes, and known quantization layouts. | Plain scalar types and supported GGUF blocks listed below. | None. |
| ONNX | Initializers, Constant tensors, nested graphs, quantization annotations, metadata, and operators. | Supported scalar raw/typed storage and confined external weights. | None. |
| PyTorch `.pt`/`.pth` | Plain tensors/state dicts and supported training checkpoint wrappers. | Dense real, non-quantized tensors, including non-contiguous layouts. | Python, PyTorch ≥ 2.6, trusted workspace. |
| Ollama safetensors collections | Aggregated stored tensors, manifest names, aliases, and source settings. | Ordinary supported dtypes and supported MXFP8 layouts with companion scales. | None. |

### Editing and saving

| Format/source | Save As | Metadata edits | Tensor replacement | Scale/zero |
| --- | --- | --- | --- | --- |
| Standalone safetensors | Yes. | String properties. | Compatible inline storage. | F64/F32/F16/BF16 weights. |
| GGUF, including a single Ollama GGUF blob | Yes, as a standalone GGUF file. | Scalar keys; existing types retained. | Compatible known-size storage. | F64/F32/F16/BF16 and supported GGUF blocks. |
| ONNX | Yes; external-data models must stay in the original directory. | String metadata properties. | Supported inline fixed-width storage, including typed F32/F64. | Supported inline F64/F32/F16/BF16 and typed F32/F64; quantization-code arithmetic excluded. |
| PyTorch `.pt`/`.pth` | Yes, with PyTorch validation. | Unavailable. | Dense real, non-quantized tensors from another PyTorch file. | F64/F32/F16/BF16 dense real tensors. |
| Ollama safetensors collections | Unavailable. | Unavailable. | Unavailable. | Unavailable. |

Supported GGUF blocks for decoding and scale/zero are **Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, and Q8_K**. Edits retain encoded weight codes and change block scales/offsets; they do not requantize the tensor. IQ and newer types expose available storage metadata, with numerical decoding and arithmetic unavailable. Q8_1 and unknown layouts have unavailable byte sizes. Big-endian GGUF and version 1 are unsupported.

ONNX quantized integer statistics describe stored codes, not dequantized weights. Sparse initializers are excluded from totals with a warning. Sparse/segmented, string/complex, newer sub-byte, and variable-length typed storage have limits on analysis or editing; each tensor explains what is available. External ONNX weight tensors are inspect-only, although supported inline tensors in the same file can be edited.

Ollama MXFP8 analysis decodes packed E4M3FN values with companion E8M0 scales using float32 rounding. The drawer distinguishes logical weight shapes/counts from stored U32 words and scales. Missing scales, incompatible layouts, FP4, and other unsupported packed schemes expose explanations instead of numerical results.

PyTorch custom pickled modules, TorchScript archives, and quantized checkpoints are unsupported. Metadata inspection uses FakeTensorMode; modern numerical reads use memory mapping. Legacy non-ZIP checkpoints above 512 MiB are unsupported. See [PyTorch setup](PYTORCH.md).

Sharded safetensors index files and general multi-file consolidation are not supported. A standalone shard can be inspected as a file; it is not an automatically assembled complete model.

## Settings

Open VS Code Settings and search for `WeightScope`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `weightscope.pythonPath` | Empty: `python3`, or `py -3` on Windows. | Python executable with PyTorch. Machine-scoped; only needed for PyTorch files. |
| `weightscope.ollamaModelsPath` | Empty: automatic discovery. | Store folder containing `manifests` and `blobs`. Machine-scoped. |
| `weightscope.sampleSize` | `32768` | Maximum values sampled per tensor, from `256` to `262144`. Higher values increase numerical analysis work. |
| `weightscope.maxDiffTensors` | `500` | Compatible tensors per batch, from `1` to `10000`. Individual rows can always be analyzed. |
| `weightscope.absoluteTolerance` | `0` | Non-negative absolute tolerance when classifying numerical changes. |

For Remote SSH, configure paths on the remote extension host. Sample size affects analysis and comparison coverage; supported native edits process the entire selected tensor.

### Command Palette reference

| Command | Action |
| --- | --- |
| **WeightScope: Inspect Model** | Choose a checkpoint file. |
| **WeightScope: Compare Checkpoints** | Choose a baseline file and target file. |
| **WeightScope: Open Ollama Model** | Choose a downloaded catalog entry. |
| **WeightScope: Compare Ollama Models** | Choose two downloaded entries. |
| **WeightScope: PyTorch Setup Instructions** | Open the interpreter setup guide. |

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| The file opens as binary/text or in another extension. | Use **Reopen Editor With… → WeightScope**. |
| An older installation has no editing controls. | Install a VSIX containing checkpoint editing. Also check the selected tensor's support explanation. |
| Save As says the destination exists. | Choose a fresh filename. Existing files are never overwritten, even if a file dialog offered to replace one. |
| Save As fails with a link/filesystem error. | Choose a writable destination on a filesystem that supports hard links. The original is unchanged. |
| A checkpoint changed on disk. | Discard pending edits if necessary, use **Tools → Refresh model**, then restage the intended changes. |
| A replacement source changed on disk. | Select the donor again before saving. |
| Scaling fails because of NaNs, infinities, or overflow. | Use supported zeroing to clear non-finite values. For overflow, choose a factor whose results fit the stored dtype. |
| Replacement offers no compatible tensors. | Choose a checkpoint with matching shapes, dtypes, byte lengths, and supported storage. |
| PyTorch cannot start or is missing. | Set `weightscope.pythonPath` and follow [PyTorch setup](PYTORCH.md). |
| PyTorch requires workspace trust. | Use a trusted workspace for PyTorch files. Native inspection works in Restricted Mode. |
| PyTorch edits fail or use too much memory. | Use dense real, non-quantized tensors. Editing loads and reserializes on CPU and may need substantial RAM. Each helper request has a two-minute timeout. |
| The Ollama catalog is empty or missing. | Choose the folder containing `manifests` and `blobs`, or set `weightscope.ollamaModelsPath` on the correct host. Cloud-only/missing downloads are not local weights. |
| Numerical analysis or an edit is unavailable. | Read the storage/quantization explanation and support tables. Some encodings expose metadata only. |
| Counts differ from a model card. | Counts describe stored entries, including buffers, aliases, and quantization scales. |
| No numerical changes appear after staging an edit. | Save the plan, then compare the saved file. The original inspector and its statistics remain unchanged. |
| An ONNX copy cannot be saved in another folder. | Keep models with external weights in their original directory so relative references remain valid. |

For repository setup, contributing changes, or test commands, continue with the [developer README](DEVELOPER.md).
