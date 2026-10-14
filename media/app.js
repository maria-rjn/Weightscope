/* global acquireVsCodeApi, WeightScopeView */
'use strict';
(() => {
  const vscode = acquireVsCodeApi(), view = WeightScopeView, root = document.getElementById('app'), saved = vscode.getState() || {}, PAGE = 40;
  const state = { tab: saved.tab || 'weights', query: saved.query || '', dtype: '', page: 0, groupPage: 0, group: '', selected: saved.selected || '',
    stats: new Map(), rows: [], diffQuery: '', diffFilter: 'attention', diffGroup: '', diffPage: 0, diffDetailsOpen: false,
    busy: false, showTree: false, moreColumns: false, drawerOpen: false, drawerMode: 'tensor' };
  const send = (type, data = {}) => vscode.postMessage({ type, ...data });
  const persist = () => vscode.setState({ tab: state.tab, query: state.query, selected: state.selected });
  const number = n => n == null ? '—' : new Intl.NumberFormat('en', { maximumSignificantDigits: 5 }).format(n);
  const integer = n => new Intl.NumberFormat('en', { maximumFractionDigits: 0 }).format(n);
  const compact = n => new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 }).format(n);
  const percent = fraction => fraction > 0 && fraction * 100 < .01 ? '<0.01%' : `${new Intl.NumberFormat('en', { maximumFractionDigits: 2 }).format(fraction * 100)}%`;
  const bytes = n => { if (n == null) return 'Unknown'; if (!n) return '0 B'; const i = Math.min(4, Math.floor(Math.log(n) / Math.log(1024))); return `${number(n / 1024 ** i)} ${['B', 'KiB', 'MiB', 'GiB', 'TiB'][i]}`; };
  const shape = t => t.shape.length ? t.shape.join(' × ') : 'scalar';
  const labels = { unverified: 'Not analyzed', changed: 'Changed', equal: 'Full match', 'sample-match': 'Sample matches', added: 'Added', removed: 'Removed', 'shape-changed': 'Shape changed', 'basis-changed': 'Numeric units differ', unavailable: 'Unavailable' };
  const el = (tag, className, text) => { const n = document.createElement(tag); if (className) n.className = className; if (text !== undefined) n.textContent = text; return n; };
  const add = (parent, ...children) => { parent.append(...children.filter(Boolean)); return parent; };
  function button(text, click, cls = '') { const n = el('button', cls, text); n.type = 'button'; n.addEventListener('click', click); return n; }
  function action(text, type, data = {}, cls = '') { const n = button(text, () => send(type, data), cls); n.dataset.action = type; n.disabled = state.busy && type !== 'cancel'; return n; }
  function input(placeholder, value, changed) { const n = el('input'); n.type = 'search'; n.placeholder = placeholder; n.value = value; n.setAttribute('aria-label', placeholder); n.addEventListener('input', () => changed(n.value)); return n; }
  function select(choices, value, changed, label) { const n = el('select'); n.setAttribute('aria-label', label); for (const [v, title] of choices) { const o = el('option', '', title); o.value = v; n.append(o); } n.value = value; n.addEventListener('change', () => changed(n.value)); return n; }
  function pairs(items) { const n = el('dl', 'detail-pairs'); for (const [k, v] of items) add(n, el('dt', '', k), el('dd', '', v)); return n; }
  function disclosure(title, ...children) { return add(el('details', 'disclosure'), el('summary', '', title), ...children); }
  function brokenName(name, cls = 'detail-name') { const n = el('div', cls); for (const part of name.split(/([./])/)) { n.append(document.createTextNode(part)); if (part === '.' || part === '/') n.append(el('wbr')); } return n; }
  function badge(status) { return el('span', `status ${status}`, labels[status] || status); }
  function table(headers) { const wrap = el('div', 'table-wrap'), t = el('table'), head = el('thead'), row = el('tr'), body = el('tbody'); headers.forEach(h => row.append(el('th', '', h))); add(t, add(head, row), body); wrap.append(t); return { wrap, body }; }
  function pager(parent, total, page, update, size = PAGE) {
    if (total <= size) return; const last = Math.ceil(total / size) - 1, n = el('div', 'pager');
    const prev = button('← Previous', () => update(page - 1)); prev.disabled = !page;
    const next = button('Next →', () => update(page + 1)); next.disabled = page >= last;
    parent.append(add(n, prev, el('span', '', `${page + 1} / ${last + 1}`), next));
  }
  function heading(title, description, actions) { return add(el('div', 'section-heading'), add(el('div'), el('h2', '', title), description && el('p', 'subtle', description)), actions); }
  function showError(message) { for (const id of ['error', ...(state.drawerOpen ? ['drawer-error'] : [])]) { const n = document.getElementById(id); if (n) { n.textContent = message; n.hidden = false; } } }
  function buildShell() {
    root.replaceChildren(); const shell = el('div', 'shell'), top = el('header', 'topbar'), brand = el('div', 'brand'), logo = el('span', 'logo');
    add(logo, el('i'), el('i'), el('i')); add(brand, logo, el('span', '', 'WeightScope'));
    const tools = el('details', 'tools-menu'); add(tools, el('summary', '', 'Tools'), add(el('div', 'tools-popover'), action('Open Ollama model', 'openOllama'), action('Compare Ollama model', 'compareOllama'), action('Refresh model', 'refresh'), action('Export JSON report', 'export'), action('PyTorch setup', 'setup')));
    add(top, brand, add(el('div', 'actions'), add(el('span', 'local-label'), el('i', 'dot'), document.createTextNode('Local inspection')), tools, action('Compare checkpoint', 'compare', {}, 'primary')));
    const error = el('div', 'notice error'); error.id = 'error'; error.setAttribute('role', 'alert'); error.hidden = true;
    const busy = el('div', 'busy-line'); busy.id = 'busy'; busy.hidden = !state.busy; busy.setAttribute('role', 'status');
    const label = el('span', '', state.busyLabel || ''); label.id = 'busy-label'; add(busy, label, action('Cancel', 'cancel')); shell.append(top);
    if (!state.model) { add(shell, add(el('main', 'loading'), el('h1', '', 'Open a checkpoint. See what’s inside.'), el('p', 'subtle', 'Reading model metadata…'), action('Retry', 'refresh'), action('Open Ollama model', 'openOllama'), action('PyTorch setup', 'setup')), error, busy); root.append(shell); return; }
    const m = state.model, s = m.summary, hero = el('section', 'hero');
    add(hero, add(el('div', 'file-line'), el('h1', '', m.name), el('span', 'badge', m.format.toUpperCase()), m.source?.kind === 'ollama' && el('span', 'badge', 'Ollama')), add(el('div', 'model-facts'), el('span', '', `${compact(s.parameters)} stored values`), el('span', '', `${integer(s.tensorCount)} tensors`), el('span', '', `${bytes(s.fileBytes)} ${m.source ? 'weights' : 'file'}`))); shell.append(hero);
    if (m.warnings.length) shell.append(disclosure(`${m.warnings.length} format ${m.warnings.length === 1 ? 'note' : 'notes'}`, el('p', 'subtle', m.warnings.join(' ')))); add(shell, error, busy);
    const tabs = el('nav', 'tabs'); tabs.setAttribute('aria-label', 'Inspector sections');
    const choices = [['weights', 'Explore'], ['compare', 'Compare'], ['info', 'Model details']]; if (m.format === 'onnx') choices.push(['operators', 'Operators']);
    for (const [id, title] of choices) { const b = button(title, () => { state.tab = id; persist(); renderTab(); }); b.dataset.tab = id; b.className = `tab${state.tab === id ? ' active' : ''}`; b.setAttribute('aria-pressed', String(state.tab === id)); tabs.append(b); }
    const content = el('main'); content.id = 'content'; add(shell, tabs, content, add(el('footer', 'footer'), el('span', '', 'Read only · Models stay on your machine'), el('span', '', 'Weight analysis runs when you request it')));
    const drawer = el('dialog', 'drawer'); drawer.id = 'tensor-detail'; drawer.setAttribute('aria-labelledby', 'drawer-title'); drawer.addEventListener('close', () => { state.drawerOpen = false; });
    drawer.addEventListener('click', e => { if (e.target === drawer && e.clientX < drawer.getBoundingClientRect().left) drawer.close(); });
    shell.append(drawer); root.append(shell); renderTab();
  }
  function renderTab() {
    const content = document.getElementById('content'); if (!content) return; content.replaceChildren();
    document.querySelectorAll('[data-tab]').forEach(b => { b.classList.toggle('active', b.dataset.tab === state.tab); b.setAttribute('aria-pressed', String(b.dataset.tab === state.tab)); });
    if (state.tab === 'weights') renderWeights(content); else if (state.tab === 'compare') renderComparison(content); else if (state.tab === 'operators') renderOperators(content); else renderInfo(content);
  }
  function layerTree(container) {
    const sidebar = el('aside', 'sidebar'); sidebar.setAttribute('aria-label', 'Tensor name tree'); sidebar.append(el('div', 'eyebrow', 'NAME TREE'));
    const tree = { groups: new Map(), tensors: [], count: 0 };
    for (const t of state.model.tensors) { const parts = t.name.split(/[./]/).filter(Boolean); parts.pop(); let branch = tree; branch.count++; for (const p of parts) { if (!branch.groups.has(p)) branch.groups.set(p, { groups: new Map(), tensors: [], count: 0 }); branch = branch.groups.get(p); branch.count++; } branch.tensors.push(t); }
    function branch(node, parent, level = 0) {
      let count = 0;
      for (const [name, child] of node.groups) { if (++count > 100) { parent.append(el('p', 'subtle', 'Use search to find more tensors.')); break; } const d = el('details', 'tree-group'); add(d, add(el('summary'), document.createTextNode(name), el('span', 'tree-count', compact(child.count)))); let loaded = false; d.addEventListener('toggle', () => { if (d.open && !loaded) { loaded = true; branch(child, d, level + 1); } }); parent.append(d); if (!level) d.open = true; }
      for (const t of node.tensors.slice(0, 100)) { const b = button(t.name.split(/[./]/).at(-1), () => selectTensor(t.name), 'tree-row'); b.title = t.name; parent.append(b); } if (node.tensors.length > 100) parent.append(el('p', 'subtle', 'Find more tensors through search.'));
    }
    branch(tree, sidebar); container.append(sidebar);
  }
  function renderWeights(content) {
    const pane = el('section', 'explore-pane'), toggle = button(state.showTree ? 'Hide tree' : 'Browse tree', () => { state.showTree = !state.showTree; renderTab(); }, 'ghost'); toggle.setAttribute('aria-pressed', String(state.showTree));
    pane.append(heading('Explore the model', 'Choose a layer group, then select a tensor to inspect its weights.', toggle));
    const toolbar = el('div', 'toolbar'); add(toolbar, input('Search tensors…', state.query, v => { state.query = v; state.page = 0; persist(); renderWeightResults(); }), select([['', 'All precision types'], ...Object.keys(state.model.summary.dtypes).sort().map(d => [d, d])], state.dtype, v => { state.dtype = v; state.page = 0; renderWeightResults(); }, 'Filter dtype')); pane.append(toolbar);
    const workspace = el('div', state.showTree ? 'workspace with-tree' : 'workspace'); if (state.showTree) layerTree(workspace); const results = el('div', 'main'); results.id = 'weight-results'; workspace.append(results); pane.append(workspace); content.append(pane); renderWeightResults();
  }
  function renderWeightResults() {
    const parent = document.getElementById('weight-results'); if (!parent) return; parent.replaceChildren(); const group = state.groups.find(g => g.key === state.group);
    if (group) parent.append(add(el('div', 'breadcrumb'), button('All layer groups', () => { state.group = ''; state.page = 0; renderWeightResults(); }, 'text-button'), el('span', '', '/'), brokenName(group.key, 'crumb-current')));
    if (!state.query && !state.dtype && !group) {
      const total = state.model.summary.parameters, max = state.groups.reduce((n, g) => Math.max(n, g.elements), 1), grid = el('div', 'layer-grid'); parent.append(el('p', 'map-hint', 'Size bars show stored value counts. Groups follow tensor names.'));
      state.groupPage = Math.min(state.groupPage, Math.max(0, Math.ceil(state.groups.length / 24) - 1));
      for (const g of state.groups.slice(state.groupPage * 24, (state.groupPage + 1) * 24)) {
        const card = button('', () => { state.group = g.key; state.page = 0; renderWeightResults(); }, 'layer-card'); const tensorLabel = `${g.tensors.length} tensor${g.tensors.length === 1 ? '' : 's'}`; card.setAttribute('aria-label', `Explore ${g.key}, ${tensorLabel}`);
        const bar = el('div', 'size-bar'), fill = el('span'); fill.style.width = `${g.elements / max * 100}%`; bar.append(fill);
        add(card, add(el('div', 'layer-card-top'), brokenName(g.key, 'group-name'), el('span', 'layer-arrow', '→')), el('div', 'layer-value', compact(g.elements)), el('div', 'subtle', `${tensorLabel} · ${g.dtypes.size === 1 ? [...g.dtypes][0] : g.dtypes.size + ' precision types'}`), bar, el('div', 'subtle', `${percent(total ? g.elements / total : 0)} of stored values`)); grid.append(card);
      }
      parent.append(grid); pager(parent, state.groups.length, state.groupPage, p => { state.groupPage = p; renderWeightResults(); }, 24); if (!state.groups.length) parent.append(el('div', 'empty', 'No supported stored tensors. See Model details for format notes.')); return;
    }
    const query = state.query.toLowerCase(), tensors = state.model.tensors.filter(t => t.name.toLowerCase().includes(query) && (!state.dtype || t.dtype === state.dtype) && (!group || view.groupKey(t.name) === group.key));
    parent.append(add(el('div', 'list-heading'), el('p', 'subtle', `${integer(tensors.length)} tensors · Select one to inspect it`), button(state.moreColumns ? 'Fewer columns' : 'More columns', () => { state.moreColumns = !state.moreColumns; renderWeightResults(); }, 'text-button')));
    if (!tensors.length) { parent.append(el('div', 'empty', 'No tensors match. Try a different name or precision type.')); return; }
    state.page = Math.min(state.page, Math.max(0, Math.ceil(tensors.length / PAGE) - 1)); const { wrap, body } = table(state.moreColumns ? ['Tensor', 'Shape', 'Precision', 'Values', 'Storage'] : ['Tensor', 'Shape / precision', 'Values']); const max = tensors.reduce((n, t) => Math.max(n, t.elements), 1);
    for (const t of tensors.slice(state.page * PAGE, (state.page + 1) * PAGE)) {
      const row = el('tr', t.name === state.selected ? 'selected' : ''); row.dataset.tensorRow = t.name; const title = group && t.name.startsWith(group.key + '.') ? t.name.slice(group.key.length + 1) : t.name; const link = button(title, () => selectTensor(t.name), 'tensor-link'); link.title = t.name; row.append(add(el('td', 'name-cell'), link));
      if (state.moreColumns) add(row, el('td', 'shape', shape(t)), el('td', 'dtype', t.dtype), el('td', 'numeric', integer(t.elements)), el('td', 'numeric', bytes(t.bytes)));
      else { const bar = el('div', 'inline-size'), fill = el('span'); fill.style.width = `${t.elements / max * 100}%`; bar.append(fill); add(row, add(el('td'), el('div', 'shape', shape(t)), el('div', 'dtype', t.dtype)), add(el('td', 'numeric'), el('div', '', compact(t.elements)), bar)); } body.append(row);
    }
    parent.append(wrap); pager(parent, tensors.length, state.page, p => { state.page = p; renderWeightResults(); });
  }
  function selectTensor(name) { state.selected = name; state.drawerMode = 'tensor'; state.drawerOpen = true; persist(); renderDetail(); document.querySelectorAll('[data-tensor-row]').forEach(r => r.classList.toggle('selected', r.dataset.tensorRow === name)); }
  function showDifference(name) { state.selectedDiff = name; state.drawerMode = 'diff'; state.drawerOpen = true; renderDetail(); }
  function drawerHeader(parent, title, name) { const h = el('h2', '', title); h.id = 'drawer-title'; h.tabIndex = -1; add(parent, add(el('div', 'drawer-header'), h, button('Close', () => parent.close(), 'ghost')), brokenName(name)); const error = el('div', 'notice error'); error.id = 'drawer-error'; error.setAttribute('role', 'alert'); error.hidden = true; parent.append(error); }
  function tiles(items) { const n = el('div', 'metric-tiles'); for (const [label, value] of items) n.append(add(el('div'), el('div', 'subtle', label), el('div', 'metric-value', value))); return n; }
  function renderDetail() { const parent = document.getElementById('tensor-detail'); if (!parent || !state.drawerOpen) return; const wasOpen = parent.open, hadFocus = parent.contains(document.activeElement); parent.replaceChildren(); if (state.drawerMode === 'diff') renderDiffDetail(parent); else renderTensorDetail(parent); if (!wasOpen && state.drawerOpen) parent.showModal(); else if (wasOpen && hadFocus) parent.querySelector('#drawer-title')?.focus({ preventScroll: true }); updateBusy(); }
  function renderTensorDetail(parent) {
    const t = state.model.tensors.find(t => t.name === state.selected); if (!t) { state.drawerOpen = false; if (parent.open) parent.close(); return; }
    const decoded = t.logicalShape != null;
    drawerHeader(parent, 'Tensor inspector', t.name); add(parent, add(el('div', 'tensor-facts'), el('span', 'badge', decoded ? t.quantization.scheme.toUpperCase() : t.dtype), el('span', 'shape', shape(decoded ? { shape: t.logicalShape } : t))), el('p', 'subtle', `${integer(t.logicalElements ?? t.elements)} ${decoded ? 'weight' : 'stored'} values · ${bytes(t.bytes)}`)); const data = state.stats.get(t.name);
    if (t.analysisUnavailable) parent.append(add(el('div', 'analyze-prompt'), el('h3', '', 'Storage metadata available'), el('p', 'subtle', t.analysisUnavailable)));
    else if (!data) parent.append(add(el('div', 'analyze-prompt'), el('div', 'eyebrow', 'WEIGHT HEALTH'), el('h3', '', 'Check the values inside this tensor.'), el('p', 'subtle', 'See the distribution, range, and whether any values are NaN or infinite.'), action('Analyze weights', 'stats', { name: t.name }, 'primary full'), el('p', 'subtle', `Up to ${integer(state.settings.sampleSize)} values will be checked. Small tensors are checked in full.`)));
    else {
      add(parent, el('h3', '', 'Weight distribution'), el('p', 'subtle', `${data.exact ? 'Full tensor' : 'Sampled estimate'} · ${integer(data.sampled)} / ${integer(data.total)} values`)); const histogram = el('div', 'histogram'); histogram.setAttribute('role', 'img'); histogram.setAttribute('aria-label', `Weight distribution: ${data.histogram.join(', ')} values per bin`); const max = Math.max(1, ...data.histogram);
      data.histogram.forEach((count, i) => { const bin = el('span'); bin.style.height = `${count / max * 100}%`; bin.title = `Bin ${i + 1}: ${count} values`; histogram.append(bin); }); add(parent, histogram, add(el('div', 'histogram-labels'), el('span', '', number(data.min)), el('span', '', number(data.max))));
      const message = data.nan || data.inf ? `${integer(data.nan + data.inf)} NaN or infinite values found in the analyzed values.` : data.sampled ? `No NaN or infinite values found ${data.exact ? 'in this tensor' : 'in this sample'}.` : 'This tensor is empty.';
      const health = el('div', `health${data.nan || data.inf ? ' warning' : ''}`, message); health.setAttribute('role', 'status');
      add(parent, health, tiles([['Minimum', number(data.min)], ['Maximum', number(data.max)], ['Mean', number(data.mean)], ['Standard deviation', number(data.std)]]), disclosure('More statistics & sampling details', pairs([['RMS', number(data.rms)], ['Zeros among finite values', data.zeroFraction == null ? '—' : `${number(data.zeroFraction * 100)}%`], ['NaN values', integer(data.nan)], ['Infinite values', integer(data.inf)], ['Finite values analyzed', integer(data.finite)]]), el('p', 'subtle', `${data.basis}. ${data.exact ? 'Standard deviation uses the full population.' : 'Samples use evenly spaced windows. Values outside the sample can contain outliers or invalid values.'}`)), action('Analyze again', 'stats', { name: t.name }, 'full'));
    }
    if (t.valueBasis) parent.append(el('p', 'subtle', t.valueBasis));
    if (decoded) parent.append(el('p', 'subtle', 'Statistics use dequantized weight values.'), disclosure('Stored layout', pairs([['Storage dtype', t.dtype], ['Stored shape', shape(t)], ['Stored values', integer(t.elements)], ['Scale tensor', t.quantization.scaleTensor], ['Decoded dtype', t.quantization.decodedDtype]])));
    if (t.quantization) parent.append(disclosure('Quantization', pairs([['Scheme', t.quantization.scheme || t.dtype], ['Bits per weight', number(t.quantization.bitsPerWeight)], ['Values per block', number(t.quantization.blockSize)]]), disclosure('Raw quantization details', el('pre', '', JSON.stringify(t.quantization, null, 2)))));
  }
  function renderDiffDetail(parent) {
    const row = state.rows.find(r => r.name === state.selectedDiff); if (!row) { state.drawerOpen = false; if (parent.open) parent.close(); return; }
    drawerHeader(parent, 'Tensor difference', row.name); add(parent, badge(row.status), pairs([['Baseline', row.left ? `${row.left.dtype} · ${shape(row.left)}` : 'Not present'], ['Target', row.right ? `${row.right.dtype} · ${shape(row.right)}` : 'Not present']])); if (row.dtypeChanged) parent.append(el('p', 'subtle', 'The stored precision changed between checkpoints.')); const m = row.metrics;
    if (m) add(parent, el('h3', '', 'How much did it change?'), el('p', 'subtle', `${m.exact ? 'Full comparison' : 'Sampled comparison'} · ${integer(m.sampled)} / ${integer(m.total)} values`), tiles([['Mean absolute change', number(m.meanAbs)], ['Largest absolute change', number(m.maxAbs)], ['RMSE', number(m.rmse)], ['Cosine similarity', number(m.cosine)]]), disclosure('More comparison details', pairs([['Relative L2', number(m.relativeL2)], ['Changed values', integer(m.changedValues)], ['Non-finite pairs', integer(m.nonFinite)], ['Non-finite mismatches', integer(m.nonFiniteMismatch)], ['Absolute tolerance', number(m.tolerance)]]), el('p', 'subtle', 'Full match means all values agree within tolerance. Sample matches only describe the positions analyzed. Relative L2 and cosine are undefined for zero norms.')));
    else { const descriptions = { added: 'This tensor is present only in the target checkpoint.', removed: 'This tensor is present only in the baseline checkpoint.', 'shape-changed': 'The shape changed. Corresponding values cannot be compared directly.', 'basis-changed': 'These tensors use different numeric units, so a direct numerical delta would be misleading.' }; parent.append(el('p', 'subtle', descriptions[row.status] || 'Compare corresponding weight values to measure the difference.')); }
    if (row.error) parent.append(el('div', 'notice error', row.error)); if (!['added', 'removed', 'shape-changed', 'basis-changed'].includes(row.status)) parent.append(action(m ? 'Analyze again' : 'Analyze this tensor', 'analyzeDiff', { name: row.name }, 'primary full'));
  }
  function renderComparison(content) {
    const pane = el('section', 'compare-pane');
    if (!state.target) { add(pane, heading('See what changed', 'Compare this model with another checkpoint.'), add(el('div', 'compare-start'), add(el('div', 'compare-step'), el('span', 'step-number', '1'), el('div', 'eyebrow', 'BASELINE'), el('h3', '', state.model.name), el('p', 'subtle', 'The model you have open')), el('span', 'step-arrow', '→'), add(el('div', 'compare-step target-step'), el('span', 'step-number', '2'), el('div', 'eyebrow', 'TARGET'), el('h3', '', 'Choose another checkpoint'), el('p', 'subtle', 'Find added, removed, and changed weights'), action('Choose target checkpoint', 'compare', {}, 'primary'), action('Choose Ollama model', 'compareOllama', {}, 'ghost')))); content.append(pane); return; }
    const overview = view.diffOverview(state.rows);
    const analyze = action(overview.analyzed ? 'Analyze next batch' : 'Analyze weight changes', 'analyzeDiff', {}, 'primary'); analyze.id = 'analyze-comparison';
    add(pane, heading('Checkpoint comparison', 'Start with the change map. Open a group for tensor-by-tensor results.', add(el('div', 'actions'), action('Change target', 'compare', {}, 'ghost'), analyze)), add(el('div', 'compare-files'), el('span', '', state.model.name), el('span', 'file-arrow', '→'), el('span', '', state.target.name)));
    const summary = el('div'); summary.id = 'diff-summary'; const map = el('div'); map.id = 'change-map'; add(pane, summary, map);
    const detail = el('details', 'diff-disclosure'); detail.id = 'diff-detail-list'; detail.open = state.diffDetailsOpen; detail.addEventListener('toggle', () => { state.diffDetailsOpen = detail.open; }); detail.append(el('summary', '', 'Tensor-by-tensor results'));
    const toolbar = el('div', 'toolbar'); add(toolbar, input('Search differences…', state.diffQuery, v => { state.diffQuery = v; state.diffPage = 0; renderDiffResults(); }), select([['attention', 'Changes & not analyzed'], ['', 'All tensors'], ...Object.entries(labels)], state.diffFilter, v => { state.diffFilter = v; state.diffPage = 0; renderDiffResults(); }, 'Filter comparison status')); const results = el('div'); results.id = 'diff-results'; add(detail, toolbar, results);
    add(pane, detail, disclosure('How comparison works', el('p', 'subtle', `Names are matched exactly. Each batch compares up to ${integer(state.settings.maxDiffTensors)} compatible tensors, with absolute tolerance ${number(state.settings.absoluteTolerance)}. Changes are target minus baseline. Sample matches do not prove that the whole tensor is unchanged.`))); content.append(pane); renderDiffOverview(); renderDiffResults();
  }
  function renderComparisonTab() { const n = document.getElementById('content'); if (n) { n.replaceChildren(); renderComparison(n); } }
  function revealDiffResults() { const n = document.getElementById('diff-detail-list'); if (!n) return; n.querySelector('summary')?.focus({ preventScroll: true }); n.scrollIntoView({ block: 'start', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' }); }
  function renderDiffOverview() {
    const summary = document.getElementById('diff-summary'), map = document.getElementById('change-map'); if (!summary || !map) return; summary.replaceChildren(); map.replaceChildren(); const d = view.diffOverview(state.rows);
    const analyze = document.getElementById('analyze-comparison');
    if (analyze) { analyze.textContent = d.pending ? d.analyzed ? 'Analyze next batch' : 'Analyze weight changes' : d.unavailable ? 'Retry unavailable' : d.analyzed ? 'Analysis complete' : 'No comparable tensors'; analyze.dataset.inactive = String(!d.pending && !d.unavailable); analyze.disabled = state.busy || analyze.dataset.inactive === 'true'; }
    const title = d.changes ? `${d.changes} tensors have changes` : d.pending ? 'Ready to compare weight values' : d.unavailable ? 'Some tensors could not be analyzed' : d.counts['sample-match'] ? 'No changes found in analyzed values' : 'All tensors match within tolerance';
    summary.append(add(el('div', 'diff-headline'), el('h3', '', title), el('p', 'subtle', `${d.analyzed} numerically analyzed · ${d.pending} not analyzed${d.unavailable ? ` · ${d.unavailable} unavailable` : ''}`)));
    const strip = el('div', 'status-strip'); strip.setAttribute('role', 'img'); strip.setAttribute('aria-label', Object.entries(d.counts).map(([s, count]) => `${count} ${labels[s]}`).join(', ')); const legend = el('div', 'status-legend');
    for (const [status, count] of Object.entries(d.counts)) { const segment = el('span', status); segment.style.flexGrow = count; strip.append(segment); const b = button('', () => { state.diffFilter = status; state.diffPage = 0; state.diffDetailsOpen = true; renderComparisonTab(); revealDiffResults(); }, 'legend-button'); add(b, el('i', `legend-dot ${status}`), document.createTextNode(`${count} ${labels[status]}`)); legend.append(b); } add(summary, strip, legend);
    map.append(el('p', 'map-hint', 'Change map · Select a group to inspect it. Counts include structural and precision changes.')); const grid = el('div', 'change-grid');
    for (const g of d.groups.slice(0, 120)) { const text = g.status === 'added' ? `${g.rows.length} added` : g.status === 'removed' ? `${g.rows.length} removed` : g.changes ? `${g.changes} changed` : g.unavailable ? `${g.unavailable} unavailable` : g.pending ? `${g.pending} not analyzed` : g.sampleMatches ? 'Sample matches' : 'Full match'; const b = button('', () => { state.diffGroup = g.key; state.diffFilter = 'attention'; state.diffPage = 0; state.diffDetailsOpen = true; renderComparisonTab(); revealDiffResults(); }, `change-cell ${g.status}${state.diffGroup === g.key ? ' active' : ''}`); b.setAttribute('aria-label', `Inspect ${g.key}: ${text}`); add(b, brokenName(g.key, 'group-name'), el('div', 'change-count', text), el('div', 'subtle', `${g.rows.length} tensor${g.rows.length === 1 ? '' : 's'}${g.maxAbs != null && g.changes ? ` · max |Δ| ${number(g.maxAbs)}` : ''}`)); grid.append(b); }
    map.append(grid); if (d.groups.length > 120) map.append(el('p', 'subtle', 'Showing the first 120 groups. Search tensor results to inspect every group.'));
  }
  function renderDiffResults() {
    const parent = document.getElementById('diff-results'); if (!parent) return; parent.replaceChildren(); if (state.diffGroup) parent.append(add(el('div', 'breadcrumb'), button('All groups', () => { state.diffGroup = ''; state.diffPage = 0; renderDiffResults(); renderDiffOverview(); }, 'text-button'), el('span', '', '/'), brokenName(state.diffGroup, 'crumb-current')));
    const rows = view.diffRows(state.rows, { query: state.diffQuery, status: state.diffFilter, group: state.diffGroup }); state.diffPage = Math.min(state.diffPage, Math.max(0, Math.ceil(rows.length / PAGE) - 1));
    if (!rows.length) { parent.append(add(el('div', 'empty'), el('p', '', 'No tensors match this view.'), button('Show all tensors', () => { state.diffGroup = ''; state.diffFilter = ''; state.diffQuery = ''; renderComparisonTab(); }, 'ghost'))); return; }
    const { wrap, body } = table(['Tensor', 'Result', 'Mean |Δ|', 'Max |Δ|']);
    for (const row of rows.slice(state.diffPage * PAGE, (state.diffPage + 1) * PAGE)) { const n = el('tr'), result = el('td'); result.append(badge(row.status)); if (row.dtypeChanged) result.append(el('span', 'precision-change', 'Precision changed')); add(n, add(el('td', 'name-cell'), button(row.name, () => showDifference(row.name), 'tensor-link')), result, el('td', 'numeric', number(row.metrics?.meanAbs)), el('td', 'numeric', number(row.metrics?.maxAbs))); body.append(n); }
    add(parent, wrap, el('p', 'subtle', 'Select a tensor for shapes, precision, sampling coverage, RMSE, and other comparison metrics.')); pager(parent, rows.length, state.diffPage, p => { state.diffPage = p; renderDiffResults(); });
  }
  function precisionChart(s) { const n = el('div', 'precision-chart'); for (const [dtype, data] of Object.entries(s.dtypes)) { const share = s.parameters ? data.parameters / s.parameters : 0, bar = el('div', 'size-bar'), fill = el('span'); fill.style.width = `${share * 100}%`; bar.append(fill); add(n, add(el('div', 'precision-row'), el('span', 'dtype', dtype), el('span', 'subtle', `${percent(share)} · ${integer(data.parameters)} values`)), bar); } return n; }
  function ollamaInfo(source) {
    const n = disclosure('Ollama source & prompt settings', pairs([['Model', source.name], ['Registry', source.registry], ['Runner', source.runner || 'Default'], ['Manifest payload size', bytes(source.totalSize)]]), el('p', 'path', source.manifestPath));
    if (source.blobPath) n.append(disclosure('Weight blob path', el('p', 'path', source.blobPath)));
    if (source.params) n.append(disclosure('Default generation parameters', el('pre', '', JSON.stringify(source.params, null, 2))));
    if (source.configs) n.append(disclosure('Model configuration', el('pre', '', JSON.stringify(source.configs, null, 2))));
    for (const [key, title] of [['template', 'Prompt template'], ['system', 'System prompt'], ['messages', 'Messages']]) if (source[key]) n.append(disclosure(title, el('pre', '', source[key].text), source[key].truncated && el('p', 'subtle', 'Preview truncated at 64 KiB.')));
    for (const [i, license] of (source.licenses || []).entries()) n.append(disclosure(`License ${i + 1}`, el('pre', '', license.text), license.truncated && el('p', 'subtle', 'Preview truncated at 64 KiB.')));
    for (const extra of source.extraWeights || []) n.append(disclosure(`${extra.kind} · ${bytes(extra.size)}`, el('p', 'path', extra.path), el('p', 'subtle', 'This additional weight blob is not included in the primary model totals.')));
    n.append(disclosure('Manifest layers', el('pre', '', JSON.stringify(source.layers.slice(0, 100), null, 2)), source.layers.length > 100 && el('p', 'subtle', `Showing 100 of ${integer(source.layers.length)} layers. Export JSON for the full list.`)));
    return n;
  }
  function renderInfo(content) {
    const pane = el('section', 'info-pane'), s = state.model.summary; pane.append(heading('Model details', 'File metadata, precision distribution, and exact stored counts.')); const grid = el('div', 'info-grid');
    add(grid, add(el('div', 'card'), el('h3', '', 'Checkpoint'), pairs([['Stored values', integer(s.parameters)], ['Tensors', integer(s.tensorCount)], [state.model.source ? 'Primary weight files' : 'File size', bytes(s.fileBytes)], ['Tensor storage', `${bytes(s.tensorBytes)}${s.unknownByteSizes ? ' (partial)' : ''}`]]), el('p', 'path', state.model.path)), add(el('div', 'card'), el('h3', '', 'Precision distribution'), precisionChart(s)));
    add(pane, grid, state.model.source?.kind === 'ollama' && ollamaInfo(state.model.source), disclosure('Model metadata', el('pre', '', JSON.stringify(state.model.metadata, null, 2))), disclosure('Interpretation & sampling', el('p', 'subtle', 'Layer groups and the tree follow tensor names. They do not recover forward-pass order. Counts describe stored values, including buffers and aliases, rather than unique trainable parameters. ONNX counts include supported initializers and Constant tensors in parsed graphs.'), el('p', 'subtle', `Statistics sample up to ${integer(state.settings.sampleSize)} values in evenly spaced windows. Small tensors are checked in full. Samples can miss values outside the sampled positions.`)), action('PyTorch setup instructions', 'setup')); content.append(pane);
  }
  function renderOperators(content) { const pane = el('section', 'operators-pane'); pane.append(heading('ONNX operators', 'Operator order and input/output names from the serialized graph. Subgraphs use scoped names.')); const { wrap, body } = table(['Operator', 'Type', 'Inputs', 'Outputs']); for (const n of state.model.nodes.slice(0, 1000)) body.append(add(el('tr'), el('td', 'name-cell', n.name), el('td', 'dtype', n.opType), el('td', 'name-cell', n.inputs.join(', ')), el('td', 'name-cell', n.outputs.join(', ')))); pane.append(wrap); if (state.model.nodes.length > 1000) pane.append(el('p', 'subtle', 'Showing 1,000 operators. Export JSON for the complete list.')); content.append(pane); }
  function updateBusy() {
    const line = document.getElementById('busy'); if (line) line.hidden = !state.busy; const label = document.getElementById('busy-label'); if (label) label.textContent = state.busyLabel || ''; document.querySelectorAll('[data-action]').forEach(b => { b.disabled = (state.busy && b.dataset.action !== 'cancel') || b.dataset.inactive === 'true'; });
    const drawer = document.getElementById('tensor-detail'); if (drawer?.open && state.busy && !document.getElementById('drawer-progress')) { const progress = el('div', 'drawer-progress'); progress.id = 'drawer-progress'; progress.setAttribute('role', 'status'); add(progress, el('span', '', state.busyLabel), action('Cancel', 'cancel')); drawer.append(progress); } else if (!state.busy) document.getElementById('drawer-progress')?.remove();
  }
  window.addEventListener('message', event => {
    const m = event.data;
    if (m.type === 'model') { state.model = m.model; state.groups = view.groups(m.model.tensors); state.settings = m.settings; state.stats.clear(); state.rows = []; state.target = null; state.group = ''; state.page = 0; state.groupPage = 0; state.drawerOpen = false; buildShell(); }
    else if (m.type === 'statistics') { state.stats.set(m.name, m.data); if (state.drawerMode === 'tensor' && state.selected === m.name) renderDetail(); }
    else if (m.type === 'comparison') { document.getElementById('tensor-detail')?.close(); state.drawerOpen = false; state.target = m.target; state.rows = m.rows; state.tab = 'compare'; state.diffPage = 0; state.diffGroup = ''; state.diffFilter = 'attention'; state.diffDetailsOpen = false; persist(); renderTab(); }
    else if (m.type === 'diffRow') { const index = state.rows.findIndex(r => r.name === m.row.name); if (index >= 0) state.rows[index] = m.row; state.busyLabel = `Comparing tensor ${m.completed} / ${m.total}…`; updateBusy(); if (state.drawerMode === 'diff' && state.selectedDiff === m.row.name) renderDetail(); if (!state.diffTimer) state.diffTimer = setTimeout(() => { state.diffTimer = null; renderDiffOverview(); renderDiffResults(); }, 150); }
    else if (m.type === 'diffDone') { renderDiffOverview(); renderDiffResults(); }
    else if (m.type === 'busy') { state.busy = true; state.busyLabel = m.label; for (const id of ['error', 'drawer-error']) { const n = document.getElementById(id); if (n) n.hidden = true; } updateBusy(); }
    else if (m.type === 'idle') { state.busy = false; updateBusy(); }
    else if (m.type === 'error') showError(m.message);
  });
  buildShell(); send('ready');
})();
