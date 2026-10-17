'use strict';
// Explanations for every part of the inspector. Each topic serves two readers: `what` and `terms` teach the idea,
// `read` says what to look for, and `method` states how the numbers are made and where they stop being reliable.
// Fields can be functions of the current settings, so limits shown here match the ones in use.
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.WeightScopeHelp = api;
})(globalThis, function () {
  const n = value => new Intl.NumberFormat('en').format(value);
  const SECTIONS = [['start', 'Start here'], ['explore', 'Explore a model'], ['values', 'Weight values'], ['compare', 'Compare checkpoints'], ['formats', 'Formats & precision'], ['details', 'Model details']];
  const TOPICS = {
    basics: { section: 'start', title: 'Key ideas',
      what: 'A trained neural network is a large collection of numbers called weights. Training adjusts them; running the model only reads them.',
      terms: [
        ['Tensor', 'A named multi-dimensional array of weights. A matrix is a 2-D tensor and a vector is a 1-D tensor.'],
        ['Checkpoint', 'A saved snapshot of every tensor, taken during training or after a conversion. One model can have many checkpoints.'],
        ['Layer / block', 'A repeated unit of the network. A transformer stacks the same block (attention followed by a feed-forward network) many times, and tensor names usually carry the block number.'],
        ['Parameter count', 'How many weights a model has: the “7B” in a model name.'],
        ['Precision', 'How many bits store each weight: 32 (F32), 16 (BF16, F16), 8, or fewer after quantization.']
      ] },
    recipes: { section: 'start', title: 'Research recipes',
      what: 'Common questions, and the shortest path to an answer in WeightScope.',
      terms: [
        ['Read a quantization recipe', 'Explore → Model map → color by Precision. The legend lists each storage type with its bits per value and number of tensors.'],
        ['Find outlier weights', 'Click Scan weights, then color the map by Largest |value|. Open a bright cell and draw its weight image with Signed peak to locate the outlier row or column.'],
        ['See what fine-tuning changed', 'Compare the base model (baseline) with the fine-tuned model (target), click Analyze weight changes, and read the Fingerprint across depth and tensor roles.'],
        ['Measure quantization error per layer', 'Compare the original Hugging Face checkpoint with its GGUF conversion. Relative L2 per tensor is the error that conversion and quantization introduced.'],
        ['Check a checkpoint for corruption', 'Scan weights. Cells with a red corner contain NaN or infinite values; open them to see how many.'],
        ['Keep evidence', 'Tools → Export JSON report saves metadata, statistics and comparisons together with their sampling coverage and tolerance.']
      ] },
    local: { section: 'start', title: 'Local inspection',
      what: 'WeightScope reads model files on this machine. Nothing is uploaded, no network requests are made, and the original files are never modified.',
      read: 'Opening a model reads only its header: tensor names, shapes and storage types. Weight values are read only when you ask, with Analyze weights, Scan weights, Draw weight image or Analyze weight changes.',
      method: 'Because nothing is loaded up front, multi-gigabyte checkpoints open in seconds and memory use stays small. Private or unreleased models can be inspected safely.' },
    overview: { section: 'start', title: 'Checkpoint at a glance',
      what: 'The counts under the name come straight from the file header, so they are exact without reading any weights.',
      terms: [
        ['Stored values', 'How many numbers all tensors hold together. Close to the parameter count, but it also includes buffers, quantization scales and tensors stored twice (tied weights).'],
        ['Tensors', 'How many named arrays the checkpoint contains.'],
        ['File / weights size', 'Bytes on disk. Lower precision stores the same number of values in fewer bytes.']
      ],
      method: 'As a rule of thumb, size ≈ stored values × bits per value ÷ 8: a 7B model takes about 14 GB in BF16 and about 4 GB at 4.5 bits per value.' },
    explore: { section: 'explore', title: 'Explore the model',
      what: 'Explore starts with the whole model and narrows down: the model map shows every tensor at once, layer groups compare sizes, and the tensor inspector shows one tensor’s values.',
      read: ['Use the model map to see patterns across depth: which layers use which precision, where the large tensors are and, after a scan, how weight magnitudes change from layer to layer.', 'Select a layer group, then a tensor, to open the tensor inspector.', 'Search matches any part of a tensor name. The precision filter keeps one storage type.'],
      method: 'Groups and roles come from tensor names only. A checkpoint stores weights, not the code that runs them, so execution order and the connections between layers are not known.' },
    'model-map': { section: 'explore', title: 'Model map',
      what: 'A grid with one cell per tensor. Columns are layer groups, usually transformer blocks, in name order from left to right. Rows are tensor roles: the part of the name after the layer number, such as self_attn.q_proj.weight.',
      read: ['Read along a row to follow one kind of tensor through the depth of the network.', 'Read down a column to see everything inside one block.', 'A blank space means that group has no tensor with that role. Embeddings and the final norm, for example, appear only once.', 'Hover or focus a cell for its details. Click it, or press Enter, to open the tensor.'],
      method: 'Groups are cut at the first number in a name (model.layers.7). Names without a number form a group of their own. When several tensors share a group and role, the cell holds all of them and opens a list.' },
    'map-color': { section: 'explore', title: 'Map colors',
      what: 'The same map can show four different properties of each tensor.',
      terms: [
        ['Precision', 'The storage type, read from the header. Shows a quantization recipe at a glance, for example which layers a 4-bit file keeps at 6 bits.'],
        ['Stored values', 'How many numbers each tensor holds. Embeddings and feed-forward projections are usually the largest.'],
        ['Weight RMS', 'Root mean square, √(mean of w²): the typical size of a weight, ignoring its sign. Needs a scan.'],
        ['Largest |value|', 'The biggest absolute value found. Isolated bright cells point to outlier weights, a common source of quantization error. Needs a scan.']
      ],
      method: 'Size and statistics use a log scale from the smallest to the largest positive value (up to six decades), so equal color steps are equal ratios. Faint cells are exactly zero, gray cells have no finite values, and hatched cells have not been scanned yet.' },
    scan: { section: 'explore', title: 'Scan weights',
      what: 'Reads a sample of every tensor once and computes its statistics, so the model map can be colored by the weights themselves.',
      read: 'The scan uses the same sampling as Analyze weights. Tensors already analyzed are skipped, results appear as they arrive, and you can cancel at any time.',
      method: s => `Each tensor contributes up to ${n(s.sampleSize)} values (setting weightscope.sampleSize), taken from up to 32 evenly spaced windows. Smaller tensors are read in full. In a sampled tensor, an outlier between windows can be missed. PyTorch files start the Python helper once per tensor, so they scan more slowly.` },
    'layer-groups': { section: 'explore', title: 'Layer groups',
      what: 'Each row is a group of tensors that share a name prefix: usually one transformer block (model.layers.12), or a standalone part such as the embedding or the final norm.',
      terms: [
        ['Stored values', 'Bars compare how many numbers each group holds.'],
        ['Storage bytes', 'Bars compare bytes on disk. Switching between the two measures shows which groups were compressed more.'],
        ['Hatched bar', 'At least one tensor in the group has an unknown storage size, for example an unsupported quantization type.']
      ],
      method: 'All bars share one scale across every page, so lengths are comparable everywhere. Groups sort by name with numbers in numeric order, which usually, but not necessarily, matches execution order.' },
    'tensor-list': { section: 'explore', title: 'Tensor table',
      terms: [
        ['Tensor', 'The full name: a path through the model’s modules. The last part is usually weight (a learned matrix or vector) or bias (a learned offset).'],
        ['Shape', 'The size of each dimension. A linear layer’s weight is output features × input features in PyTorch, safetensors and GGUF (shown here in the same order). ONNX MatMul weights are often stored the other way round. A 1-D tensor is a vector, such as a norm scale or a bias.'],
        ['Precision', 'The storage type of each value, for example F32, BF16 or Q4_K.'],
        ['Values', 'The product of the shape: how many numbers the tensor holds.'],
        ['Storage', 'Bytes on disk, including any quantization scales.']
      ] },
    'name-tree': { section: 'explore', title: 'Name tree',
      what: 'Tensor names split at dots and slashes form a hierarchy, like folders. The count beside each branch is the number of tensors below it.',
      method: 'The tree mirrors how the training code named its modules. It is not a diagram of the computation.' },
    inspector: { section: 'values', title: 'Tensor inspector',
      what: 'Everything known about one tensor: its storage type, shape and size from the header, plus an image and statistics computed from its values when you ask for them.',
      read: 'The line under the size suggests what this tensor does, based on common naming conventions. The precision line explains how its values are stored.',
      method: 'Role hints follow the names used by Hugging Face transformers and llama.cpp. A custom model can reuse those names for something else, so treat the hint as a starting point, not a fact.' },
    'weight-image': { section: 'values', title: 'Weight image',
      what: 'Draws the tensor as a picture. The matrix is divided into at most 192 × 192 tiles, and each pixel shows one tile.',
      terms: [
        ['RMS', 'Each tile’s root mean square. Brighter means larger weights on average; good for seeing structure.'],
        ['Signed peak', 'Each tile’s value with the largest magnitude: blue below zero, red above. A single outlier stays visible.'],
        ['Clip outer 0.5%', 'Colors stop changing beyond the 0.5th and 99.5th percentile of tiles, so a few extreme tiles do not wash out the rest.'],
        ['Full range', 'Colors span every tile. Extremes are exact, but everything else can look flat.']
      ],
      read: ['Horizontal bands often follow attention heads, because the rows of a query, key or value projection are grouped by head.', 'A bright line across the whole image is one row or column with unusually large weights: an outlier channel.', 'Dark rows in an embedding often belong to vocabulary tokens that were rarely or never seen in training.', 'An even texture with no structure can mean weights that barely moved from their random initialization.'],
      method: `Leading dimensions form rows and trailing dimensions form columns, in row-major order, so a convolution kernel [out, in, kh, kw] draws as out × in·kh·kw. Tensors with more than ${n(4194304)} values (${n(262144)} for PyTorch) read evenly spaced runs of whole rows; tiles that were not read stay hatched, and the caption states how many rows were read.` },
    'weight-health': { section: 'values', title: 'Weight health',
      what: 'Checks the numbers inside the tensor: their range, how they are distributed, and whether any are invalid.',
      read: 'NaN (not a number) and infinite values break inference. They usually point to a training instability, an overflow during conversion or a damaged file. A healthy tensor contains none.',
      method: s => `Tensors up to ${n(s.sampleSize)} values are read in full. Larger ones are sampled at up to 32 evenly spaced windows, and every result says whether it is a full tensor or a sampled estimate.` },
    distribution: { section: 'values', title: 'Weight distribution',
      what: 'A histogram. The range from the smallest to the largest value is cut into 24 equal bins, and each bar shows the share of values that falls into its bin.',
      read: ['Weights of trained linear layers usually form a bell shape centered near zero.', 'Norm scales often cluster around 1, or around 0 in models that apply them as 1 + w (Gemma’s Hugging Face checkpoints, for example).', 'A tall spike at exactly 0 suggests pruning, padding or unused rows.', 'A long thin tail means a few large outliers. They stretch the range, so most values crowd into a few bars.'],
      method: 'Height is the percentage of finite analyzed values, so samples of different sizes are comparable. NaN and infinite values are excluded and counted separately. Distribution bin counts lists the exact numbers behind each bar.' },
    'summary-stats': { section: 'values', title: 'Summary statistics',
      terms: [
        ['Minimum / Maximum', 'The smallest and largest finite value. In a sample, the true extremes can lie further out.'],
        ['Mean', 'The average. Usually near zero for projection weights; a clear offset is more typical of norm scales and biases.'],
        ['Standard deviation', 'How widely values spread around the mean: √(mean of (w − mean)²).'],
        ['RMS', 'Root mean square, √(mean of w²): the typical magnitude. RMS² = standard deviation² + mean².'],
        ['Zeros among finite values', 'The share of values that are exactly 0. A high share suggests sparsity, pruning or padding.'],
        ['NaN / Infinite values', 'Invalid numbers. Any count above zero deserves a closer look.'],
        ['Finite values analyzed', 'How many valid numbers the statistics are based on.']
      ],
      method: 'Standard deviation divides by n (the population formula), not n − 1. In a sampled tensor, minimum, maximum and the zero share are estimates.' },
    quantization: { section: 'formats', title: 'Quantization',
      what: 'Quantization stores weights with fewer bits. Most formats cut a tensor into small blocks; each block keeps one or two shared numbers (a scale, sometimes a minimum) and a short integer code per weight. A weight is rebuilt as code × scale, plus the minimum when there is one.',
      terms: [
        ['Scheme', 'The block format, such as Q4_K or Q8_0.'],
        ['Bits per weight', 'Average storage per weight, including the shared scales. Q4_K takes 4.5 bits, not 4.'],
        ['Values per block', 'How many weights share one set of scales. K-quants use super-blocks of 256, split into smaller sub-blocks.']
      ],
      read: 'Fewer bits mean a smaller file and more rounding error. Mixed recipes such as Q4_K_M keep some sensitive tensors, often attention value and feed-forward down projections, at a higher precision.',
      method: 'Statistics and comparisons use dequantized values: the numbers the runtime computes with. ONNX quantized integers are the exception and are analyzed as stored codes.' },
    'stored-layout': { section: 'formats', title: 'Stored layout',
      what: 'This tensor is packed: several low-precision values share one stored word, and a companion scale tensor holds their shared scales. WeightScope unpacks them into the logical shape shown above.',
      method: 'Ollama MXFP8 stores four 8-bit E4M3 floats in each 32-bit word, with one power-of-two E8M0 scale per 32 weights (the OCP Microscaling format). Model totals count the stored words, not the unpacked weights.' },
    formats: { section: 'formats', title: 'File formats',
      terms: [
        ['SAFETENSORS', 'Hugging Face’s format: a JSON header followed by raw arrays. Loading it cannot run code. Large models are split into shards with an index file.'],
        ['GGUF', 'The llama.cpp and Ollama format: metadata key/value pairs plus tensors, often in block-quantized types such as Q4_K.'],
        ['ONNX', 'A portable graph format: the operators of the computation plus initializer tensors (the weights), used by many inference runtimes.'],
        ['PYTORCH', '.pt, .pth and .bin files written by torch.save. They are read through your Python with restricted weights-only loading, which refuses arbitrary pickled objects.']
      ] },
    'precision-distribution': { section: 'details', title: 'Precision distribution',
      what: 'The share of stored values held in each storage type.',
      read: 'A quantized model usually keeps norm scales, and often the embedding or output layer, at a higher precision than the bulk of its weights. A single type means the whole file was stored uniformly.' },
    details: { section: 'details', title: 'Checkpoint totals',
      terms: [
        ['Stored values', 'How many numbers all tensors hold together, including buffers and quantization scales.'],
        ['Tensors', 'How many named tensors the checkpoint contains.'],
        ['File size', 'Bytes on disk for the checkpoint’s files.'],
        ['Tensor storage', 'Bytes taken by tensor data alone. The rest of the file is header, metadata and alignment padding. “Partial” means some sizes are unknown.']
      ] },
    metadata: { section: 'details', title: 'Model metadata',
      what: 'Key/value pairs written into the file header by the tool that saved it. GGUF files carry architecture details here, such as layer and head counts, context length, rotary settings and the tokenizer. Safetensors metadata is usually short.',
      method: 'Long arrays, such as a tokenizer vocabulary, show their first 32 entries together with their full length.' },
    ollama: { section: 'details', title: 'Ollama source & prompt settings',
      terms: [
        ['Model / Registry', 'The model name and tag, and where Ollama downloaded it from.'],
        ['Runner', 'The inference engine the manifest asks for, when it names one.'],
        ['Manifest payload size', 'Total size of every file the manifest lists.'],
        ['Prompt template', 'How Ollama wraps your messages before the model sees them, including special tokens.'],
        ['System prompt', 'Default instructions added to every conversation.'],
        ['Default generation parameters', 'Sampling settings such as temperature and context length.'],
        ['Manifest layers', 'Every file the model is made of, identified by content hash.']
      ],
      method: 'JSON reports can include these prompts and settings. Review a report before sharing it.' },
    shards: { section: 'details', title: 'Shards',
      what: 'Large checkpoints are split across several files called shards. An index file maps each tensor to the shard that stores it, and WeightScope reads them all as one model.',
      method: 'The index is checked against every shard’s header. Each analysis reads only the shard that holds the tensor.' },
    operators: { section: 'details', title: 'ONNX operators',
      what: 'An ONNX file stores the computation itself: a list of operators such as MatMul, Add or Softmax, in the order they run.',
      read: 'Inputs and outputs are the names of values passed between operators. An input that matches a tensor name is a weight.',
      method: 'Subgraphs, for example inside If or Loop, use scoped names. The list stops at 1,000 operators; the JSON report contains all of them.' },
    'compare-start': { section: 'compare', title: 'Comparing checkpoints',
      what: 'A comparison lines up two checkpoints tensor by tensor. The baseline is the reference, the model open now. The target is measured against it, and every change is target minus baseline.',
      terms: [
        ['Base → fine-tuned', 'Which layers training changed, and by how much.'],
        ['Original → quantized', 'For example a Hugging Face checkpoint and its GGUF: how much error each layer picked up.'],
        ['Step → later step', 'Whether some layers stopped moving or changed suddenly during training.'],
        ['Release → release', 'Which tensors were added, removed or reshaped.']
      ],
      method: 'Structural differences (added, removed, reshaped and re-typed tensors) appear at once from the headers. Value differences are computed only when you click Analyze.' },
    comparison: { section: 'compare', title: 'Checkpoint comparison',
      what: 'Structure is compared as soon as the target opens. Analyze weight changes then reads matching values from both checkpoints and measures how far each tensor moved.',
      read: s => `Each click analyzes the next batch of up to ${n(s.maxDiffTensors)} tensors (setting weightscope.maxDiffTensors); the map fills in as results arrive. Select any cell or row to analyze that tensor alone.`,
      method: 'Tensors are matched by name. Between a GGUF file and its Hugging Face source, names are translated and conversion steps are reversed before comparing; the note under the file names says when this happens.' },
    statuses: { section: 'compare', title: 'Comparison results',
      terms: [
        ['Not analyzed', 'The structure matches; the values have not been compared yet.'],
        ['Changed', 'At least one analyzed value differs by more than the tolerance, or NaN/infinity appears on one side only.'],
        ['Full match', 'Every value was compared and agrees within the tolerance.'],
        ['Sample matches', 'Every sampled value agrees, but the rest of the tensor was not read. This is not proof of equality.'],
        ['Added', 'Present only in the target.'],
        ['Removed', 'Present only in the baseline.'],
        ['Shape changed', 'Present in both with different dimensions, so values cannot be paired.'],
        ['Numeric units differ', 'One side stores real values and the other raw integer codes, so a numerical difference would be meaningless.'],
        ['Unavailable', 'The values could not be read or decoded. Select the tensor for the reason.'],
        ['Precision changed', 'The storage type differs, for example BF16 → Q4_K. Shown next to the result, because the values themselves may still match.']
      ],
      read: 'Select an entry in the legend to list the tensors with that result.' },
    'drift-map': { section: 'compare', title: 'Layer drift map',
      what: 'The model map’s grid, colored by how much each tensor changed. The measure is relative L2 change: the size of the change divided by the size of the original weights, so 0.05 means 5%.',
      terms: [
        ['Fingerprint', 'One dense image of the whole comparison; best for patterns across depth.'],
        ['Table', 'The same cells with values written in; best for reading exact numbers.'],
        ['Log scale', 'Equal color steps are equal ratios, so a 0.1% change stays visible next to a 50% one.'],
        ['Linear scale', 'Color is proportional to drift; small changes fade next to a large one.']
      ],
      read: ['A bright column: one block changed much more than its neighbors.', 'A bright row: one kind of tensor, such as every value projection, changed throughout the network.', 'A gradient from left to right: change grows or fades with depth.', 'Only a few roles changed while everything else is exactly zero: the typical signature of merged LoRA adapters on those modules.'],
      method: 'Relative L2 = ‖target − baseline‖ / ‖baseline‖ over the analyzed values. One color scale is shared by every page and filter. Color shows the amount of change, not whether the change is good or bad. A baseline that is all zeros has no relative change and shows n/a.' },
    'drift-results': { section: 'compare', title: 'Tensor-by-tensor results',
      terms: [
        ['Result', 'The comparison status of the tensor (see Comparison results).'],
        ['Mean |Δ|', 'The average absolute difference between paired values, |target − baseline|.'],
        ['Max |Δ|', 'The largest absolute difference found.']
      ],
      read: 'The default filter, Changes & not analyzed, hides tensors that matched. Changed tensors come first, ordered by Max |Δ|.' },
    difference: { section: 'compare', title: 'Tensor difference',
      what: 'One tensor in both checkpoints: its storage type and shape on each side, and how much its values changed.',
      method: 'When names were translated between formats, both names and every reversed conversion step are listed here.' },
    'change-metrics': { section: 'compare', title: 'Change measures',
      terms: [
        ['Mean absolute change', 'The average of |target − baseline|, in the units of the weights.'],
        ['Largest absolute change', 'The single biggest difference found.'],
        ['RMSE', 'Root mean squared error, √(mean of (target − baseline)²). Large differences count more than in the mean.'],
        ['Cosine similarity', 'Whether the two tensors, read as long vectors, point the same way: 1 is the same direction, 0 unrelated, −1 opposite. It ignores overall scale.'],
        ['Relative L2', '‖target − baseline‖ / ‖baseline‖: the change relative to the size of the weights, comparable across tensors of different scale. 0.01 is 1%.'],
        ['Changed values', 'Paired values whose difference exceeds the absolute tolerance.'],
        ['Non-finite pairs / mismatches', 'Pairs where either side is NaN or infinite, and those where the two sides disagree.'],
        ['Absolute tolerance', 'Differences up to this size count as equal (setting weightscope.absoluteTolerance).']
      ],
      read: 'Absolute measures depend on the scale of the weights, so compare them within one tensor role. Relative L2 and cosine compare across layers. As a rough guide for bell-shaped weights, 8-bit block quantization adds well under 1% relative L2 and 4-bit formats several percent, while fine-tuning ranges from a fraction of a percent to tens of percent.',
      method: 'Values are paired at identical positions in both tensors. A sampled comparison reads the same positions on both sides, so changes elsewhere are not seen.' },
    'change-image': { section: 'compare', title: 'Change image',
      what: 'Draws target − baseline as a picture, with the same tiles as the weight image.',
      read: ['With Signed peak, blue tiles decreased and red tiles increased.', 'Change concentrated in some rows or columns points to specific heads, channels or vocabulary entries.', 'An even, noisy texture everywhere is typical of quantization rounding.'],
      method: 'Both tensors are read at the same positions, including GGUF tensors whose rows were reordered during conversion.' },
    'diff-distributions': { section: 'compare', title: 'Baseline & target distributions',
      what: 'Both histograms on the same bins and vertical scale: the baseline as filled bars, the target as a line.',
      read: 'A line that follows the bars means the overall distribution was kept, even if individual values moved. A narrower or wider line means the weights shrank or grew; a shifted line means their center moved.',
      method: 'Bins span both sides’ combined range, and heights are percentages of each side’s finite values.' }
  };

  // How each storage type represents a value. Bits include block scales.
  const PRECISION = {
    F64: '64-bit float (double precision). Rare for weights.',
    F32: '32-bit float: 8 exponent and 23 fraction bits. Full precision; norms and biases are often kept in F32.',
    F16: '16-bit float: 5 exponent and 10 fraction bits. Precise, but values above 65,504 overflow.',
    BF16: 'bfloat16: the exponent range of F32 with a 7-bit fraction. The usual precision for training and releasing large models.',
    F8_E4M3: '8-bit float with 4 exponent and 3 fraction bits, normally paired with separate scales.',
    F8_E5M2: '8-bit float with 5 exponent and 2 fraction bits: more range, less precision than E4M3.',
    BOOL: 'True/false values, typically masks.',
    Q4_0: '4-bit codes with one F16 scale per block of 32 weights (4.5 bits per weight).',
    Q4_1: '4-bit codes with an F16 scale and minimum per block of 32 weights (5 bits per weight).',
    Q5_0: '5-bit codes with one F16 scale per block of 32 weights (5.5 bits per weight).',
    Q5_1: '5-bit codes with an F16 scale and minimum per block of 32 weights (6 bits per weight).',
    Q8_0: '8-bit codes with one F16 scale per block of 32 weights (8.5 bits per weight). Close to lossless for most weights.',
    Q2_K: 'K-quant: 2-bit codes in super-blocks of 256 weights, with a 4-bit scale and minimum for every 16 (2.625 bits per weight).',
    Q3_K: 'K-quant: 3-bit codes in super-blocks of 256 weights, with a 6-bit scale for every 16 (3.4375 bits per weight).',
    Q4_K: 'K-quant: 4-bit codes in super-blocks of 256 weights, with a 6-bit scale and minimum for every 32 (4.5 bits per weight).',
    Q5_K: 'K-quant: 5-bit codes in super-blocks of 256 weights, with a 6-bit scale and minimum for every 32 (5.5 bits per weight).',
    Q6_K: 'K-quant: 6-bit codes in super-blocks of 256 weights, with an 8-bit scale for every 16 (6.5625 bits per weight).',
    Q8_K: 'K-quant: 8-bit codes with an F32 scale per 256 weights. Mostly an intermediate format for dot products.',
    TQ1_0: 'Ternary: every weight is −1, 0 or +1 times a block scale.',
    TQ2_0: 'Ternary: every weight is −1, 0 or +1 times a block scale, stored in 2 bits.',
    MXFP4: 'OCP microscaling: 4-bit E2M1 floats in blocks of 32 sharing one power-of-two scale.',
    MXFP8: 'OCP microscaling: 8-bit E4M3 floats in blocks of 32 sharing one power-of-two scale.',
    NVFP4: '4-bit E2M1 floats with an 8-bit scale for every 16 values.'
  };
  function precisionNote(label) {
    if (!label) return null;
    const key = String(label).toUpperCase();
    if (PRECISION[key]) return PRECISION[key];
    if (key.startsWith('F8_E4M3')) return PRECISION.F8_E4M3;
    if (key.startsWith('F8_E5M2')) return PRECISION.F8_E5M2;
    if (key === 'F8_E8M0') return '8-bit power-of-two exponent with no sign or fraction, used as a shared block scale.';
    if (/^F[46]_E\dM\d$/.test(key)) return `${key[1]}-bit float, normally stored in blocks that share a scale.`;
    if (key.startsWith('IQ')) return 'i-quant: non-uniform levels or lattice codebooks, often chosen with an importance matrix. Its layout is shown, but its values are not decoded in this release.';
    if (/^(U|I)(2|4|8|16|32|64)$/.test(key) || /^U?INT\d+$/.test(key)) return 'Integers. In model files these are usually quantized codes, token or position indices, or other buffers, rather than learned floating-point weights.';
    return null;
  }

  // What a tensor most likely does, from naming conventions in Hugging Face transformers, GPT-2 and llama.cpp.
  const ROLES = [
    [/lora_a\b/, 'LoRA adapter, matrix A: projects the input down to a small rank. The adapter adds B × A, times a scale, to a base weight.'],
    [/lora_b\b/, 'LoRA adapter, matrix B: projects the low-rank features back up. The adapter adds B × A, times a scale, to a base weight.'],
    [/inv_freq|rope_freqs/, 'Rotary position frequencies: a fixed buffer computed from the model configuration, not a trained weight.'],
    [/ffn_gate_inp|\.router\b|block_sparse_moe\.gate\b|mlp\.gate\b/, 'Mixture-of-experts router: scores which experts each token is sent to.'],
    [/qkv_proj|attn_qkv|c_attn|query_key_value|in_proj_(weight|bias)|\.qkv\b/, 'Fused attention projection: the query, key and value projections stored as one matrix.'],
    [/q_norm|k_norm|q_layernorm|k_layernorm/, 'Query/key norm: normalizes queries or keys head by head before attention scores are computed.'],
    [/post_attention_norm|post_ffw_norm|post_feedforward_layernorm|pre_feedforward_layernorm/, 'Sandwich norm scale: an extra normalization around the attention or feed-forward block.'],
    [/input_layernorm|attn_norm\b|\bln_1\b|attention_norm/, 'Pre-attention norm scale: one multiplier per feature, applied to the normalized hidden state before attention.'],
    [/post_attention_layernorm/, 'Norm scale between attention and the feed-forward block. Most models apply it to the feed-forward input; Gemma 2/3 and OLMo 2 apply it to the attention output.'],
    [/ffn_norm|\bln_2\b/, 'Pre-feed-forward norm scale: one multiplier per feature, applied before the feed-forward block.'],
    [/^(?!.*\.\d+\.).*(^|\.)(norm|output_norm|ln_f|final_layernorm|final_layer_norm)\.weight$/, 'Final norm: normalizes the last hidden state before the output projection.'],
    [/norm/, 'Normalization scale: one multiplier per feature, applied after normalizing.'],
    [/(^|[._])q_proj|attn_q\b|\.wq\b|\.query\b/, 'Attention query projection: turns each token’s hidden state into queries, which are matched against keys to decide where to attend. Rows are grouped by attention head.'],
    [/(^|[._])k_proj|attn_k\b|\.wk\b|\.key\b/, 'Attention key projection: produces the keys that queries are compared with. With grouped-query attention it has fewer heads than the query projection.'],
    [/(^|[._])v_proj|attn_v\b|\.wv\b|\.value\b/, 'Attention value projection: produces the content each attended position contributes.'],
    [/(^|[._])o_proj|attn_output|out_proj|attn\.c_proj|attention\.output\.dense|attention\.dense|\.wo\b/, 'Attention output projection: mixes the results of all heads back into the hidden state.'],
    [/gate_up_proj/, 'Fused feed-forward gate and up projections, stored as one matrix.'],
    [/gate_proj|ffn_gate(_exps|_shexp)?\b|\.w1\b/, 'Feed-forward gate projection: in a gated (SwiGLU or GeGLU) block, controls how much of each intermediate feature passes through.'],
    [/up_proj|ffn_up(_exps|_shexp)?\b|\.w3\b|c_fc|\bfc1\b|dense_h_to_4h|intermediate\.dense/, 'Feed-forward up projection: expands the hidden state to a wider intermediate size.'],
    [/down_proj|ffn_down(_exps|_shexp)?\b|\.w2\b|mlp\.c_proj|\bfc2\b|dense_4h_to_h|output\.dense/, 'Feed-forward down projection: maps the intermediate features back to the hidden size.'],
    [/embed_tokens|\bwte\b|word_embeddings|tok_embeddings|token_embd/, 'Token embedding: one row per vocabulary token, turning each input token into a vector.'],
    [/\bwpe\b|position_embeddings|embed_positions/, 'Position embedding: one learned row per position in the context.'],
    [/(^|\.)(lm_head|embed_out)\b|^output\.weight$/, 'Output projection (LM head): turns the final hidden state into a score for every vocabulary token. Some models reuse the token embedding here instead (tied weights).'],
    [/mm_projector|multi_modal_projector|^mm\./, 'Multimodal projector: maps image features into the language model’s embedding space.'],
    [/conv/, 'Convolution kernel: output channels × input channels × kernel size.']
  ];
  function tensorRole(name) {
    const lower = String(name).toLowerCase();
    if (/(weight|input)[._]scales?$|[._](scales|biases|qbias)$/.test(lower)) return 'Quantization scales or offsets stored beside a packed weight; they rebuild its values when it is decoded.';
    const rule = ROLES.find(([pattern]) => pattern.test(lower)); if (!rule) return null;
    const parts = [/(^|\.)(vision|visual|vit|vision_tower|vision_model|image_encoder)\b|^v\.blk\./.test(lower) && 'In the vision encoder.', rule[1],
      /shared_expert|_shexp/.test(lower) ? 'It belongs to a shared expert that every token uses.' : /_exps\b/.test(lower) ? 'It stacks every expert of a mixture of experts in one tensor; each token uses only a few experts.' : /experts?\.\d+/.test(lower) && 'It belongs to one expert of a mixture of experts; each token uses only a few experts.',
      /[._]bias$/.test(lower) && !/lora/.test(lower) && 'This is the bias: a learned offset added after the multiplication.'];
    return parts.filter(Boolean).join(' ');
  }
  // One line under the model map legend for each coloring.
  const MAP_COLORS = {
    precision: 'Each color is one storage type. Bits per value include block scales; fewer bits mean a smaller file and more rounding error.',
    size: 'Brighter cells hold more values. On the log scale, equal color steps are equal ratios.',
    rms: 'Brighter cells have larger typical weights (RMS). Norm scales often sit near 1, far above projection weights.',
    peak: 'Brighter cells contain a larger single weight. An isolated bright cell is an outlier worth opening.'
  };
  return { SECTIONS, TOPICS, MAP_COLORS, precisionNote, tensorRole };
});
