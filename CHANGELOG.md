# Changelog

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
