# Validation of version 0.3.0

## Sharded PyTorch update — October 17, 2026

The isolated sharded-PyTorch branch, stacked on the sharded-safetensors branch, passed `npm run check` and all **86 tests with no failures or skips**, using the existing Python framework environment (PyTorch 2.14.1). Without that environment, the 78 Node tests pass and the 8 framework tests are skipped. A fake PyTorch helper records each request. It confirms that opening a `.bin` index or shard inspects every shard once, and that each sample is read from the tensor's own shard and never from the index. It also confirms that untrusted workspaces start no helper at all. The Node tests also cover the safetensors-style validation cases for `.bin` indexes (missing, unreadable, misplaced, and duplicated shards and tensors, path traversal, and `.safetensors` entries in a `.bin` index), with each failure falling back to one shard. Further tests cover index and shard change detection, folders that hold both layouts, standalone `.bin` files, GGUF files named `.bin`, and the manifest patterns.

Framework tests write real `torch.save` shards and a transformers-style index beside the upstream safetensors shards. Their BF16, F32, and non-contiguous tensors match the safetensors shards exactly. Real shards are also checked for index disagreements, duplicated tensors, missing shards, and change detection. A shard carrying an executable pickle is rejected by restricted loading without running it, and the rest of its set opens alone. A real Extension Development Host session, the browser preview, and a downloaded multi-gigabyte `.bin` model were not run.

## Cross-format GGUF comparison — October 17, 2026

`WEIGHTSCOPE_TEST_PYTHON="$PWD/.test-runtime/bin/python" npm test` passed **78 tests with no failures or skips**, and `npm run check` passed. Native tests build Hugging Face-named safetensors and GGUF files whose Q/K rows were reordered by a literal `reshape → swapaxes` implementation of llama.cpp's `permute`, then verify exact matches for full and bounded samples in both comparison directions, per-block head-count arrays, explanations for missing or indivisible head counts, Gemma sandwich-norm naming and the `1 + w` offset, stacked Mixtral experts, unsupported architectures, and ONNX fallback. A control test confirms that unpermuted rows report drift, so the restoration is applied rather than assumed.

The framework fixtures convert a two-block GQA Llama with gguf 0.19.0: gguf-py's tensor name map, llama.cpp's `permute` copied verbatim, and gguf-py's Q8_0 quantizer. For all 21 pairs, WeightScope's mean and maximum absolute error match the values computed from gguf-py's dequantizer within 1e-9. Restored K rows have cosine similarity 0.99999; without restoration the same tensor shows cosine 0.30. A second fixture checks WeightScope's name translation against gguf-py's `TensorNameMap` for every supported architecture.

The production frontend was checked in the browser preview with the Hugging Face checkpoint as baseline and the Q8_0 GGUF as target. The comparison header explained the translation, 16 quantized matrices showed as changed and 5 F32 norms as full matches. The Q drawer showed both tensor names and the rotary note, and searching `attn_k` found the translated key rows. A real Extension Development Host and real multi-gigabyte models were not run for this update.

## Sharded safetensors update — October 17, 2026

The isolated sharded-safetensors branch passed `npm run check` and all **77 tests with no failures or skips**, using the existing Python framework environment. New coverage opens Hugging Face indexes and individual shards as one checkpoint. It also covers sampling each tensor from its own shard, comparisons with single-file checkpoints, symlinked Hugging Face cache snapshots and repository naming, index-free grouping, and incomplete sets that fall back to one shard with a note. Inconsistent indexes are covered too: a missing shard, an absent, misplaced, or duplicated tensor, path traversal, non-safetensors shards, invalid JSON, and unindexed tensors. The remaining tests cover index and shard change detection, cancellation, export protection for every shard, and the disabled Save As. An upstream fixture writes shards with safetensors 0.8.0 and a transformers-style index; its BF16 and F32 tensors match the single-file checkpoint exactly.

The real Qwen3-32B index in the local Hugging Face cache (17 shards, 707 tensors, 65,524,246,528 bytes) parsed and was named `Qwen/Qwen3-32B/model.safetensors`. Its shards were not downloaded, so opening it reported the incomplete download rather than inspecting weights. The browser preview was checked at desktop width and 320 px with a three-shard demo: the shard badge, per-shard table, the shard name in the tensor inspector, numerical analysis, and a sharded-to-single-file drift comparison. The page had no horizontal overflow or console errors. A real Extension Development Host session and a multi-gigabyte sharded model were not run.

## Visualization update — October 17, 2026

The isolated visualization branch, including main's checkpoint Save As changes, passed `npm run check` and all **67 tests with no failures or skips**, using the existing Python framework environment. New coverage checks shared histogram domains, constant/empty/non-finite distributions, overflowing numeric ranges, zero baseline norms, exported distribution data, natural matrix ordering, normalized-name collisions, and retention of 135 groups × 12 roles. The VSIX also builds with the existing packager and no added runtime dependencies.

The production frontend was checked against the real synthetic checkpoint preview at 1280 × 900 and 320 × 900. Stored-value/storage-byte switching, group-to-tensor navigation, numerical analysis, map search, eight-row group drill-down, shared-bin distribution overlays, Escape-to-close, and keyboard focus restoration worked. The narrow comparison drawer and document have no horizontal overflow; the matrix scrolls horizontally within its own table container. Captures: [overview](images/layer-overview.jpg), [drift map](images/layer-drift-map.jpg), and [comparison distributions](images/shared-distributions.jpg).

These checks use the browser preview and mocked VS Code host. A real Extension Development Host, live Remote SSH session, and multi-gigabyte performance benchmark were not run for this update. The earlier release validation below describes its own verification date and scope.

## Earlier release validation

Verified on October 14, 2026, on macOS ARM64 with Node.js 24 and Python 3.12.

## Automated checks

`npm run check` passed JavaScript syntax, manifest, and packaged-asset checks.

`WEIGHTSCOPE_TEST_PYTHON="$PWD/.test-runtime/bin/python" npm test` passed **57 tests with no failures or skips**. Tests cover native format readers, scalar/FP8 decoding, primary GGUF and K-quant numerical decoding, bounded sampling and coverage, structural/numerical comparison, NaN/Inf handling, external ONNX path confinement, file mutation, workspace trust, and extension message/export integration. Visualization tests check grouping totals, numeric group ordering, precision-only changes, structural labels, filtering, and the distinction between pending, sampled, and full comparisons.

Ollama tests exercise explicit/environment/default store discovery, the model/tag picker, custom-folder fallback, cancellation of the picker, missing/corrupt/cloud/invalid-digest entries, direct hash-named GGUF inspection, safetensors aggregation, manifest tensor names, shared aliases, packed quantization limits, bounded text previews, configuration metadata, local manifest-list references and cycles, duplicate names, stale manifests/blobs, cross-format comparison, editor restoration, and report overwrite/symlink protection.

Independent fixtures use torch 2.14.1, ONNX 1.23.2, safetensors 0.8.0, and gguf 0.19.0. Primary Q4/Q5/Q8_0 decoders and Q2_K through Q6_K decoders were checked against the official GGUF Python dequantizers at every value in generated blocks and at bounded sample positions. Q8_K uses an explicit signed-code fixture because that package does not supply a Q8_K dequantizer. Legacy and modern PyTorch state dictionaries, BF16, non-contiguous tensors, executable-pickle rejection, and TorchScript rejection were exercised with real PyTorch.

Ollama MXFP8 decoding was checked against independent PyTorch E4M3FN and E8M0 conversions for all 65,536 code/scale combinations, including signed zeros, NaNs, float32 overflow, and underflow, in full and bounded samples. Native tests also cover byte order, block/row/expert boundaries, one-value samples, per-tensor metadata in mixed blobs, malformed or missing scale companions, companion-blob mutation before and during sampling, and comparisons with floating-point and packed checkpoints. Read instrumentation verifies that sampling fetches only the requested code bytes and the scales for their blocks.

## Browser verification

The production frontend was tested using the Playwright CLI against a localhost preview connected to real synthetic checkpoints. The redesigned overview shows seven layer groups with no default tensor table or open drawer. Group selection, tensor analysis, histogram rendering, Escape-to-close with focus restoration, batch diff analysis, and selection of a changed group worked. The change map reported eight changed tensors in `model.layers.2`, one added tensor, one removed tensor, and 26 full numerical matches. Selecting that group revealed exactly eight detailed tensor rows.

A strict content-security policy permits only the local stylesheet and nonce-authorized scripts. The viewer produced no browser errors or warnings after tensor analysis and comparison. Layouts were checked at 1280 × 900 and 620 × 900; the narrow overview has no horizontal body overflow. The checked screenshots are included in [docs/images](images/).

The Ollama view uses a generated local catalog containing the same synthetic checkpoints. Friendly model/tag titles, the Ollama badge, collapsed source settings, generation parameters, prompt templates, and the Choose Ollama model comparison action were checked with the Playwright CLI. The source details screenshot is [ollama-details.png](images/ollama-details.png). The real VS Code catalog picker remains covered by the mocked-host test, rather than this browser preview.

The MXFP8 inspector was also exercised with the real local TEV embedding tensor through a temporary localhost preview using the production frontend and backend. Analyze weights rendered a 32,768-value histogram with the logical shape/count and sampling denominator, a separate stored-layout disclosure, and 8-bit/32-value block metadata. At 620 × 900, the drawer had no horizontal overflow. The browser console had no errors or warnings.

## Downloaded Ollama models

Read-only metadata inspection succeeded for all nine model/tag entries in the local Ollama store: six GGUF entries and three safetensors tensor collections. These included Gemma, Qwen, TranslateGemma, TEV, Flux, and Z-Image weights. Supported tensors were sampled directly from the original multi-gigabyte files; no copying, renaming, inference, or Ollama server was involved. All 201 packed MXFP8 weight tensors in the TEV model have validated decoding layouts. Its embedding tensor reports a logical shape of 248320 × 2560 (635,699,200 weights) while retaining its stored U32 shape of 248320 × 640 (158,924,800 words). A 32,768-value sample produced finite dequantized values spanning -0.0703125 to 0.078125. Other packed schemes such as FP4 retain explicit numerical-decoding limits. This was a compatibility check, not a cold-cache or full-weight performance benchmark.

## Packaging and limits of verification

The official `@vscode/vsce` packager produced `artifacts/weightscope-0.3.0.vsix`. Packaged runtime sources were checked byte-for-byte against workspace sources. No runtime npm dependencies, downloaded models, or test environments are included.

A real Extension Development Host was **not launched**. Extension activation, custom-editor setup, messaging, statistics, comparison, report export, and overwrite protection were verified using a mocked VS Code API. No multi-gigabyte performance benchmark, Windows/Linux host test, or live Remote SSH host test was run.

Runtime-format boundaries and sampling semantics are documented in the README. Unsupported numerical formats produce explicit errors or warnings. The extension has not been published to a Marketplace.
