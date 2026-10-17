# WeightScope

Inspect model checkpoints locally, edit supported weights and metadata in a new copy, and compare the result with the original.

WeightScope is a VS Code extension for `.safetensors`, `.gguf`, `.onnx`, `.pt`, and `.pth` files, including downloaded Ollama models. Inspection and analysis run on demand. The extension makes no network requests, uploads, or telemetry calls.

## Choose your guide

| Guide | Start here if you want to… |
| --- | --- |
| [User guide](docs/USER_GUIDE.md) | Install the extension, inspect a model, edit a checkpoint, compare results, or troubleshoot a problem. |
| [Developer README](docs/DEVELOPER.md) | Set up the repository, understand the code, run tests, change the extension, or build a VSIX. |

These documents describe the current source tree. Older VSIX packages may have fewer features.

## What it does

- Explore layer groups, tensor names, shapes, stored value counts, precision, and model metadata.
- Analyze tensor distributions, zeros, NaNs, infinities, and other weight statistics.
- Compare checkpoint structure, numerical weight changes, and metadata differences.
- Stage metadata changes or compatible tensor replacement, scaling, and zeroing.
- Save a validated checkpoint under a new name, keeping the original unchanged.
- Export a JSON inspection and comparison report.

Editing depends on the format and tensor storage. Multi-blob Ollama safetensors collections and external ONNX weight tensors are inspect-only. See the [support tables](docs/USER_GUIDE.md#format-and-editing-support) before editing.

WeightScope edits serialized checkpoint data. It does not train models or evaluate their generated responses. Sampling coverage is shown explicitly; a sample match does not establish whole-tensor equality.

## Install and open a model

1. Obtain a VSIX built from the desired source revision. To build one yourself, follow the [developer setup](docs/DEVELOPER.md#quick-start).
2. In VS Code, run **Extensions: Install from VSIX…** and select the package.
3. Run **WeightScope: Inspect Model**, or open a supported file.

Requires VS Code 1.90 or newer. Native formats need no Python installation. PyTorch files require Python with PyTorch 2.6 or newer and a trusted workspace; see [PyTorch setup](docs/PYTORCH.md).

If another extension opens the file, select **Reopen Editor With… → WeightScope**. Remote SSH works when the extension, model files, and any Python interpreter are on the extension host.

## Documentation

| Document | Purpose |
| --- | --- |
| [User guide](docs/USER_GUIDE.md) | Complete workflows, supported operations, settings, and troubleshooting. |
| [Developer README](docs/DEVELOPER.md) | Setup, architecture, tests, development workflow, and packaging. |
| [PyTorch setup](docs/PYTORCH.md) | Interpreter configuration and restricted loading behavior. |
| [Validation notes](docs/VALIDATION.md) | Recorded checks and their verification limits. |
| [Product direction](docs/PRODUCT.md) | Intended workflow, product boundaries, and candidate improvements. |
| [Changelog](CHANGELOG.md) | Release history. |

MIT licensed. The repository uses a placeholder publisher ID; Marketplace publication requires an owned publisher ID and release metadata.
