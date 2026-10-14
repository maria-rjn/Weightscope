# Validation of version 0.2.0

Verified on October 14, 2026, on macOS ARM64 with Node.js 24 and Python 3.12.

## Automated checks

`npm run check` passed JavaScript syntax, manifest, and packaged-asset checks.

`WEIGHTSCOPE_TEST_PYTHON="$PWD/.test-runtime/bin/python" npm test` passed **30 tests with no failures or skips**. Tests cover native format readers, scalar/FP8 decoding, primary GGUF and K-quant numerical decoding, bounded sampling and coverage, structural/numerical comparison, NaN/Inf handling, external ONNX path confinement, file mutation, workspace trust, and extension message/export integration. Visualization tests check grouping totals, numeric group ordering, precision-only changes, structural labels, filtering, and the distinction between pending, sampled, and full comparisons.

Independent fixtures use torch 2.14.1, ONNX 1.23.2, safetensors 0.8.0, and gguf 0.19.0. Primary Q4/Q5/Q8_0 decoders and Q2_K through Q6_K decoders were checked against the official GGUF Python dequantizers at every value in generated blocks and at bounded sample positions. Q8_K uses an explicit signed-code fixture because that package does not supply a Q8_K dequantizer. Legacy and modern PyTorch state dictionaries, BF16, non-contiguous tensors, executable-pickle rejection, and TorchScript rejection were exercised with real PyTorch.

## Browser verification

The production frontend was tested using the Playwright CLI against a localhost preview connected to real synthetic checkpoints. The redesigned overview shows seven layer groups with no default tensor table or open drawer. Group selection, tensor analysis, histogram rendering, Escape-to-close with focus restoration, batch diff analysis, and selection of a changed group worked. The change map reported eight changed tensors in `model.layers.2`, one added tensor, one removed tensor, and 26 full numerical matches. Selecting that group revealed exactly eight detailed tensor rows.

A strict content-security policy permits only the local stylesheet and nonce-authorized scripts. The viewer produced no browser errors or warnings after tensor analysis and comparison. Layouts were checked at 1280 × 900 and 620 × 900; the narrow overview has no horizontal body overflow. The checked screenshots are included in [docs/images](images/).

## Packaging and limits of verification

The official `@vscode/vsce` packager produced `artifacts/weightscope-0.2.0.vsix`. Packaged runtime sources were checked byte-for-byte against workspace sources. No runtime npm dependencies or test environments are included.

A real Extension Development Host was **not launched**. Extension activation, custom-editor setup, messaging, statistics, comparison, report export, and overwrite protection were verified using a mocked VS Code API. No multi-gigabyte performance benchmark, Windows/Linux host test, or live Remote SSH host test was run.

Runtime-format boundaries and sampling semantics are documented in the README. Unsupported numerical formats produce explicit errors or warnings. The extension has not been published to a Marketplace.
