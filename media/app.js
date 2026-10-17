/* global acquireVsCodeApi, WeightScopeView */
'use strict';
(() => {
  const vscode = acquireVsCodeApi(), view = WeightScopeView, root = document.getElementById('app'), saved = vscode.getState() || {}, PAGE = 40;
  const state = { tab: saved.tab || 'weights', query: saved.query || '', dtype: '', page: 0, groupPage: 0, group: '', selected: saved.selected || '',
    stats: new Map(), images: new Map(), diffImages: new Map(), imageMode: saved.imageMode === 'peak' ? 'peak' : 'rms', imageRange: saved.imageRange === 'full' ? 'full' : 'clip', rows: [], diffQuery: '', diffFilter: 'attention', diffGroup: '', diffRole: '', diffPage: 0, diffDetailsOpen: false,
    sizeMetric: saved.sizeMetric === 'bytes' ? 'bytes' : 'elements', mapPage: 0, rolePage: 0, mapQuery: '', chartObservers: [],
    mapView: saved.mapView === 'table' ? 'table' : 'fingerprint', mapScale: saved.mapScale === 'linear' ? 'linear' : 'log', gridCursors: {}, gridObservers: new Map(), modelColor: ['size', 'rms', 'peak'].includes(saved.modelColor) ? saved.modelColor : 'precision', modelPage: 0, modelRolePage: 0, scanErrors: new Map(),
    busy: false, showTree: false, moreColumns: false, drawerOpen: false, drawerMode: 'tensor' };
  const send = (type, data = {}) => vscode.postMessage({ type, ...data });
  const persist = () => vscode.setState({ tab: state.tab, query: state.query, selected: state.selected, sizeMetric: state.sizeMetric, mapView: state.mapView, mapScale: state.mapScale, imageMode: state.imageMode, imageRange: state.imageRange, modelColor: state.modelColor });
  const number = n => n == null ? '—' : new Intl.NumberFormat('en', { maximumSignificantDigits: 5 }).format(n);
  const integer = n => new Intl.NumberFormat('en', { maximumFractionDigits: 0 }).format(n);
  const compact = n => new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 2 }).format(n);
  const chartNumber = n => new Intl.NumberFormat('en', { maximumSignificantDigits: 3, notation: Math.abs(n) >= 10000 || (n !== 0 && Math.abs(n) < .001) ? 'scientific' : 'standard' }).format(n);
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
  function pager(parent, total, page, update, size = PAGE, label = '') {
    if (total <= size) return; const last = Math.ceil(total / size) - 1, n = el('div', 'pager');
    n.dataset.pageLabel = label;
    function move(nextPage, direction) {
      update(nextPage);
      const replacement = [...parent.querySelectorAll('.pager')].find(p => p.dataset.pageLabel === label);
      const choices = replacement?.querySelectorAll('button');
      (choices?.[direction]?.disabled ? choices[1 - direction] : choices?.[direction])?.focus({ preventScroll: true });
    }
    const prev = button(`← Previous${label ? ` ${label}` : ''}`, () => move(page - 1, 0)); prev.disabled = !page;
    const next = button(`Next${label ? ` ${label}` : ''} →`, () => move(page + 1, 1)); next.disabled = page >= last;
    parent.append(add(n, prev, el('span', '', `${page + 1} / ${last + 1}`), next));
  }
  function heading(title, description, actions) { return add(el('div', 'section-heading'), add(el('div'), el('h2', '', title), description && el('p', 'subtle', description)), actions); }
  function showError(message) { for (const id of ['error', ...(state.drawerOpen ? ['drawer-error'] : [])]) { const n = document.getElementById(id); if (n) { n.className = 'notice error'; n.setAttribute('role', 'alert'); n.textContent = message; n.hidden = false; } } }
  function buildShell() {
    state.chartObservers.forEach(o => o.disconnect()); state.chartObservers = []; state.gridObservers.forEach(o => o.disconnect()); state.gridObservers.clear();
    root.replaceChildren(); const shell = el('div', 'shell'), top = el('header', 'topbar'), brand = el('div', 'brand'), logo = el('span', 'logo');
    logo.setAttribute('aria-hidden', 'true'); add(brand, logo, el('span', '', 'WeightScope'));
    const tools = el('details', 'tools-menu'); add(tools, el('summary', '', 'Tools'), add(el('div', 'tools-popover'), action('Open Ollama model', 'openOllama'), action('Compare Ollama model', 'compareOllama'), action('Save checkpoint as…', 'saveCheckpoint'), action('Refresh model', 'refresh'), action('Export JSON report', 'export'), action('PyTorch setup', 'setup')));
    add(top, brand, add(el('div', 'actions'), add(el('span', 'local-label'), el('i', 'dot'), document.createTextNode('Local inspection')), tools, action('Compare checkpoint', 'compare', {}, 'primary')));
    const error = el('div', 'notice error'); error.id = 'error'; error.setAttribute('role', 'alert'); error.hidden = true;
    const busy = el('div', 'busy-line'); busy.id = 'busy'; busy.hidden = !state.busy; busy.setAttribute('role', 'status');
    const label = el('span', '', state.busyLabel || ''); label.id = 'busy-label'; add(busy, label, action('Cancel', 'cancel')); shell.append(top);
    if (!state.model) { add(shell, add(el('main', 'loading'), el('h1', '', 'Open a checkpoint. See what’s inside.'), el('p', 'subtle', 'Reading model metadata…'), action('Retry', 'refresh'), action('Open Ollama model', 'openOllama'), action('PyTorch setup', 'setup')), error, busy); root.append(shell); return; }
    const m = state.model, s = m.summary, hero = el('section', 'hero');
    add(hero, add(el('div', 'file-line'), el('h1', '', m.name), el('span', 'badge', m.format.toUpperCase()), m.source?.kind === 'ollama' && el('span', 'badge', 'Ollama'), m.source?.kind === 'sharded' && el('span', 'badge', `${integer(m.source.shards.length)} shards`)), add(el('div', 'model-facts'), el('span', '', `${compact(s.parameters)} stored values`), el('span', '', `${integer(s.tensorCount)} tensors`), el('span', '', `${bytes(s.fileBytes)} ${m.source ? 'weights' : 'file'}`))); shell.append(hero);
    if (m.warnings.length) shell.append(disclosure(`${m.warnings.length} format ${m.warnings.length === 1 ? 'note' : 'notes'}`, el('p', 'subtle', m.warnings.join(' ')))); add(shell, error, busy);
    const tabs = el('nav', 'tabs'); tabs.setAttribute('aria-label', 'Inspector sections');
    const choices = [['weights', 'Explore'], ['compare', 'Compare'], ['info', 'Model details']]; if (m.format === 'onnx') choices.push(['operators', 'Operators']);
    for (const [id, title] of choices) { const b = button(title, () => { state.tab = id; persist(); renderTab(); }); b.dataset.tab = id; b.className = `tab${state.tab === id ? ' active' : ''}`; b.setAttribute('aria-pressed', String(state.tab === id)); tabs.append(b); }
    const content = el('main'); content.id = 'content'; add(shell, tabs, content, add(el('footer', 'footer'), el('span', '', 'Original files stay unchanged · Models stay on your machine'), el('span', '', 'Weight analysis runs when you request it')));
    const drawer = el('dialog', 'drawer'); drawer.id = 'tensor-detail'; drawer.setAttribute('aria-labelledby', 'drawer-title'); drawer.addEventListener('close', () => {
      state.drawerOpen = false; state.chartObservers.forEach(o => o.disconnect()); state.chartObservers = [];
      const name = state.drawerMode === 'diff' ? state.selectedDiff : state.selected;
      const replacement = [...root.querySelectorAll('[data-matrix-key], [data-tensor-row] button, .tensor-link')].find(b => b.dataset.matrixKey === name || b.closest('[data-tensor-row]')?.dataset.tensorRow === name || b.textContent === name);
      (state.returnFocus?.isConnected ? state.returnFocus : replacement)?.focus({ preventScroll: true });
    });
    drawer.addEventListener('click', e => { if (e.target === drawer && e.clientX < drawer.getBoundingClientRect().left) drawer.close(); });
    shell.append(drawer); root.append(shell); renderTab();
  }
  function renderTab() {
    const content = document.getElementById('content'); if (!content) return; content.replaceChildren(); state.gridObservers.forEach(o => o.disconnect()); state.gridObservers.clear();
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
      const storage = state.sizeMetric === 'bytes', value = g => storage && g.unknownBytes ? null : g[state.sizeMetric];
      const max = state.groups.reduce((n, g) => Math.max(n, value(g) || 0), 1), list = el('div', 'layer-list'), map = el('section', 'model-map');
      map.id = 'model-map'; parent.append(map); renderModelMap(map); if (state.groups.length) parent.append(el('h3', 'list-title', 'Layer groups'));
      parent.append(add(el('div', 'list-heading'), el('p', 'map-hint', 'Groups follow tensor names. Bars share one scale across all pages.'), select([['elements', 'Stored values'], ['bytes', 'Storage bytes']], state.sizeMetric, v => { state.sizeMetric = v; persist(); renderWeightResults(); }, 'Layer bar measure')));
      state.groupPage = Math.min(state.groupPage, Math.max(0, Math.ceil(state.groups.length / 24) - 1));
      for (const g of state.groups.slice(state.groupPage * 24, (state.groupPage + 1) * 24)) {
        const row = button('', () => { state.group = g.key; state.page = 0; renderWeightResults(); }, 'layer-row');
        const n = value(g), label = n == null ? 'Unknown storage' : storage ? bytes(n) : `${compact(n)} values`;
        const tensorLabel = `${integer(g.tensors.length)} tensor${g.tensors.length === 1 ? '' : 's'}`;
        row.setAttribute('aria-label', `Explore ${g.key}, ${tensorLabel}, ${label}`);
        const bar = el('div', `size-bar${n == null ? ' unknown-size' : ''}`), fill = el('span'); fill.style.width = `${(n || 0) / max * 100}%`; bar.append(fill);
        add(row, add(el('div', 'layer-label'), brokenName(g.key, 'group-name'), el('span', 'subtle', `${tensorLabel} · ${[...g.dtypes].join(', ')}`)), bar, el('span', 'layer-measure', label)); list.append(row);
      }
      parent.append(list); pager(parent, state.groups.length, state.groupPage, p => { state.groupPage = p; renderWeightResults(); }, 24); if (!state.groups.length) parent.append(el('div', 'empty', 'No supported stored tensors. See Model details for format notes.')); return;
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
  // Tableau 10: distinct categories that hold up on dark and light backgrounds.
  const PRECISION_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'];
  const peakOf = s => s?.min == null ? null : Math.max(Math.abs(s.min), Math.abs(s.max));
  // The model at a glance: every tensor is one cell, colored right away by stored precision or size, or by weight statistics after a scan.
  function renderModelMap(container) {
    state.gridObservers.get('model-map')?.disconnect(); container.replaceChildren();
    const tensors = state.model.tensors, mode = state.modelColor, stats = state.stats, light = lightTheme(); if (!tensors.length) return;
    const matrix = view.comparisonMatrix(tensors), kinds = view.precisionKinds(tensors), colorOf = new Map(kinds.map((k, i) => [k.label, PRECISION_COLORS[i % PRECISION_COLORS.length]]));
    const scannable = tensors.filter(t => !t.analysisUnavailable), scanned = scannable.filter(t => stats.has(t.name) || state.scanErrors.has(t.name)).length, numeric = mode === 'rms' || mode === 'peak';
    const measure = t => mode === 'size' ? t.logicalElements ?? t.elements : mode === 'rms' ? stats.get(t.name)?.rms ?? null : mode === 'peak' ? peakOf(stats.get(t.name)) : null;
    let low = Infinity, high = 0; for (const t of tensors) { const v = measure(t); if (Number.isFinite(v) && v > 0) { low = Math.min(low, v); high = Math.max(high, v); } }
    const scale = view.rangeScale(low === Infinity ? 0 : low, high, 'log'), format = mode === 'size' ? compact : chartNumber;
    const choice = select([['precision', 'Precision'], ['size', 'Stored values'], ['rms', 'Weight RMS'], ['peak', 'Largest |value|']], mode, v => { state.modelColor = v; persist(); refreshModelMap(); }, 'Color the model map by');
    const scan = action(scanned ? 'Scan remaining weights' : 'Scan weights', 'scan', {}, 'ghost'); choice.dataset.matrixKey = 'model-color'; scan.dataset.matrixKey = 'model-scan';
    scan.dataset.inactive = String(scanned >= scannable.length); scan.disabled = state.busy || scanned >= scannable.length;
    container.append(add(el('div', 'map-heading'), add(el('div'), el('h3', '', 'Model map'), el('p', 'map-hint', `${integer(matrix.groups.length)} layer groups × ${integer(matrix.roles.length)} tensor roles · Select a cell to inspect its tensor.`)), add(el('div', 'map-controls'), choice, scan)));
    const { groups, roles, size } = gridPage(matrix, 'modelPage', 'modelRolePage');
    function modelCell(group, role, list) {
      const t = list[0], s = stats.get(t.name), kind = view.precisionLabel(t), count = t.logicalElements ?? t.elements, values = list.map(measure).filter(Number.isFinite);
      const value = values.length ? Math.max(...values) : null, scannedCell = list.every(x => stats.has(x.name)), error = list.map(x => x.analysisUnavailable || state.scanErrors.get(x.name)).find(Boolean);
      const bad = list.reduce((n, x) => n + (stats.get(x.name)?.nan || 0) + (stats.get(x.name)?.inf || 0), 0), bits = t.bytes == null || !count ? null : t.bytes * 8 / count;
      const dims = shape(t.logicalShape != null ? { shape: t.logicalShape } : t), reading = mode === 'precision' ? `${kind}${bits == null ? '' : ` · ${number(bits)} bits/value`} · ${dims}` : mode === 'size' ? `${compact(count)} values · ${bytes(t.bytes)}`
        : scannedCell ? `${mode === 'rms' ? 'RMS' : 'Largest |value|'} ${number(value)} · ${s.exact ? 'full tensor' : 'sampled'}` : error ? 'Analysis unavailable' : 'Not scanned';
      return { kind, value, scanned: scannedCell, error, bad, detail: `${list.length > 1 ? `${list.length} tensors · ` : ''}${reading}${bad ? ` · ${integer(bad)} NaN/Inf` : ''}`,
        description: `${list.length > 1 ? `${group.key}, ${role}, ${list.length} tensors` : t.name}: ${kind}, ${dims}, ${compact(count)} values${s ? `, RMS ${number(s.rms)}, largest magnitude ${number(peakOf(s))}, ${s.exact ? 'full tensor' : 'sampled estimate'}` : ''}${bad ? `, ${integer(bad)} NaN or infinite values` : ''}${error ? `. ${error}` : ''}`,
        open: () => { if (list.length === 1) selectTensor(t.name); else { state.group = group.key; state.page = 0; renderWeightResults(); } } };
    }
    const cells = groups.map(g => roles.map(r => g.cells.has(r) ? modelCell(g, r, g.cells.get(r)) : null));
    gridCanvas(container, { key: 'model-map', label: 'Model map', roledescription: 'model map', groups, roles, cells, paint(g, item, x, y, w, h, t) {
      if (mode === 'precision') { g.fillStyle = colorOf.get(item.kind); g.fillRect(x, y, w, h); }
      else if (item.value > 0) { g.fillStyle = view.heatColor(scale.position(item.value), light); g.fillRect(x, y, w, h); }
      else if (item.value === 0 || (numeric && item.scanned)) { g.globalAlpha = item.value === 0 ? .22 : .5; g.fillStyle = t.muted; g.fillRect(x, y, w, h); }
      else if (item.error) t.outline(x, y, w, h, t.css('--red'), true);
      else { g.fillStyle = t.pattern; g.fillRect(x, y, w, h); }
      if (item.bad) t.corner(x, y, w, h, t.css('--red'), 'bottom');
    } });
    const legend = el('div', 'heatmap-legend'), swatch = (cls, text, color) => { const i = el('i', `swatch ${cls}`); if (color) i.style.background = color; return add(el('span'), i, document.createTextNode(text)); };
    if (mode === 'precision') for (const k of kinds) legend.append(swatch('', `${k.label}${k.bits == null ? '' : ` · ${number(k.bits)} bits`} · ${integer(k.tensors)} tensor${k.tensors === 1 ? '' : 's'}`, colorOf.get(k.label)));
    else {
      if (scale.ticks.length) { const ramp = el('i', 'drift-ramp'); ramp.style.background = `linear-gradient(to right, ${[0, .25, .5, .75, 1].map(f => view.heatColor(f, light)).join(', ')})`; add(legend, el('span', '', `${mode === 'size' ? 'Stored values' : mode === 'rms' ? 'RMS' : 'Largest |value|'}${scale.mode === 'log' ? ' (log)' : ''}`), el('span', '', format(scale.ticks[0])), ramp, el('span', '', format(scale.ticks[2]))); }
      if (numeric) add(legend, swatch('unverified', 'Not scanned'), swatch('undefined-drift', 'No finite values'), swatch('unavailable', 'Unavailable'));
    }
    if (stats.size) legend.append(swatch('non-finite', 'Contains NaN or infinity'));
    container.append(legend, el('p', 'map-hint', `${integer(scanned)} of ${integer(scannable.length)} tensors scanned. ${numeric && !scanned ? 'Scan the weights to color the map by their values. ' : ''}A scan samples up to ${integer(state.settings.sampleSize)} values per tensor, like Analyze weights${state.model.format === 'pytorch' ? ', and starts the PyTorch helper once per tensor' : ''}. Cancel it at any time.`));
    pager(container, matrix.groups.length, state.modelPage, p => { state.modelPage = p; refreshModelMap(); }, size.groups, 'groups');
    pager(container, matrix.roles.length, state.modelRolePage, p => { state.modelRolePage = p; refreshModelMap(); }, size.roles, 'roles');
  }
  function refreshModelMap() {
    const container = document.getElementById('model-map'); if (!container) return;
    const focused = container.contains(document.activeElement) ? document.activeElement.dataset.matrixKey : null;
    renderModelMap(container); if (focused) container.querySelector(`[data-matrix-key="${focused}"]`)?.focus({ preventScroll: true });
  }
  function scheduleModelMap() { if (!state.modelMapTimer) state.modelMapTimer = setTimeout(() => { state.modelMapTimer = null; refreshModelMap(); }, 150); }
  function selectTensor(name) { state.returnFocus = document.activeElement; state.selected = name; state.drawerMode = 'tensor'; state.drawerOpen = true; persist(); renderDetail(); document.querySelectorAll('[data-tensor-row]').forEach(r => r.classList.toggle('selected', r.dataset.tensorRow === name)); }
  function showDifference(name) { state.returnFocus = document.activeElement; state.selectedDiff = name; state.drawerMode = 'diff'; state.drawerOpen = true; renderDetail(); }
  function drawerHeader(parent, title, name) { const h = el('h2', '', title); h.id = 'drawer-title'; h.tabIndex = -1; add(parent, add(el('div', 'drawer-header'), h, button('Close', () => parent.close(), 'ghost')), brokenName(name)); const error = el('div', 'notice error'); error.id = 'drawer-error'; error.setAttribute('role', 'alert'); error.hidden = true; parent.append(error); }
  function tiles(items) { const n = el('div', 'metric-tiles'); for (const [label, value] of items) n.append(add(el('div'), el('div', 'subtle', label), el('div', 'metric-value', value))); return n; }
  const quantile = (sorted, q) => sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] : 0;
  // A tensor as pixels: each pixel is one tile of its matrix view, colored by the tile's signed peak or RMS.
  function imageSection(parent, data, kind) {
    const box = el('div', 'weight-image'), canvas = el('canvas'), readout = el('p', 'image-readout'), legend = el('div', 'image-legend'), facts = el('p', 'subtle'), word = kind === 'diff' ? 'change' : 'value';
    const span = (i, n, total, unit) => { const a = Math.floor(i * total / n), b = Math.max(a, Math.ceil((i + 1) * total / n) - 1); return a === b ? `${unit} ${integer(a)}` : `${unit}s ${integer(a)}–${integer(b)}`; };
    const where = t => `${span(Math.floor(t / data.tileCols), data.tileRows, data.rows, 'row')}, ${span(t % data.tileCols, data.tileCols, data.cols, 'column')}`;
    const hint = 'Hover the image to read a tile.', nonFinite = new Set(data.nonFinite);
    const controls = add(el('div', 'image-controls'), select([['rms', 'RMS'], ['peak', 'Signed peak']], state.imageMode, v => { state.imageMode = v; persist(); draw(); }, 'Image statistic'), select([['clip', 'Clip outer 0.5%'], ['full', 'Full range']], state.imageRange, v => { state.imageRange = v; persist(); draw(); }, 'Image color range'));
    canvas.setAttribute('role', 'img'); readout.textContent = hint; box.append(canvas); add(parent, controls, box, readout, legend, facts);
    function draw() {
      const peak = state.imageMode === 'peak', values = data[peak ? 'peak' : 'rms'], light = lightTheme(), magnitudes = values.filter(v => v != null).map(Math.abs).sort((a, b) => a - b);
      // Signed peaks share one symmetric range around zero. RMS is never negative and clusters well above zero, so it spans its own low-to-high range.
      const level = (state.imageRange === 'clip' ? quantile(magnitudes, .995) : magnitudes.at(-1)) || magnitudes.at(-1) || 1;
      const floor = peak ? 0 : Math.min(state.imageRange === 'clip' ? quantile(magnitudes, .005) : magnitudes[0] ?? 0, level * .999), position = v => (v - floor) / (level - floor);
      const clipped = level < (magnitudes.at(-1) || 0) || floor > (peak ? 0 : magnitudes[0] ?? 0);
      const off = document.createElement('canvas'); off.width = data.tileCols; off.height = data.tileRows;
      const g = off.getContext('2d'); g.fillStyle = getComputedStyle(box).getPropertyValue('--mint'); g.fillRect(0, 0, 1, 1); const invalid = [...g.getImageData(0, 0, 1, 1).data.slice(0, 3)];
      const pixels = g.createImageData(data.tileCols, data.tileRows);
      values.forEach((v, i) => { if (nonFinite.has(i)) pixels.data.set([...invalid, 255], i * 4); else if (v != null) pixels.data.set([...(peak ? view.divergingRgb(v / level, light) : view.heatRgb(position(v), light)), 255], i * 4); });
      g.putImageData(pixels, 0, 0);
      const width = Math.max(120, box.clientWidth), natural = width * data.rows / data.cols, height = Math.round(Math.max(36, Math.min(width * 1.25, natural))), ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); canvas.style.height = `${height}px`;
      const c = canvas.getContext('2d'); c.imageSmoothingEnabled = false; c.drawImage(off, 0, 0, canvas.width, canvas.height);
      const ramp = el('i', 'image-ramp'); ramp.style.background = `linear-gradient(to right, ${[0, .25, .5, .75, 1].map(f => `rgb(${(peak ? view.divergingRgb(f * 2 - 1, light) : view.heatRgb(f, light)).join(', ')})`).join(', ')})`;
      legend.replaceChildren(); add(legend, el('span', '', peak ? `−${chartNumber(level)}` : chartNumber(floor)), ramp, el('span', '', peak ? `+${chartNumber(level)}` : chartNumber(level)), el('span', 'subtle', peak ? `Signed peak ${word} per tile${clipped ? ', clipped' : ''}` : `RMS ${word} per tile${clipped ? ', clipped' : ''}`), nonFinite.size && add(el('span'), el('i', 'swatch non-finite-tile'), document.createTextNode(`${integer(nonFinite.size)} tiles contain NaN or infinity`)));
      const largest = data.largest ? `Largest ${word} magnitude${data.exact ? '' : ' among values read'}: ${number(data.largest.value)} at ${where(data.largest.tile)}.` : `No finite ${word}s.`;
      facts.textContent = `${data.exact ? 'Full tensor' : `${integer(data.rowsRead)} of ${integer(data.rows)} rows read${data.columnsSampled ? ' in sampled column windows' : ', every column'}`} · ${integer(data.rows)} × ${integer(data.cols)} shown as ${integer(data.tileRows)} × ${integer(data.tileCols)} tiles${data.shape.length > 2 ? ` (first ${data.split} of ${data.shape.length} dimensions as rows)` : ''}.${Math.abs(natural - height) > 1 ? ' Aspect ratio compressed to fit.' : ''}${clipped ? ` Colors saturate beyond ${peak ? `±${chartNumber(level)}` : `${chartNumber(floor)} and ${chartNumber(level)}`}.` : ''} ${largest}`;
      canvas.setAttribute('aria-label', `${kind === 'diff' ? 'Change' : 'Weight'} image, ${data.tileRows} by ${data.tileCols} tiles. ${peak ? `Blue is negative and red is positive, saturating at ±${chartNumber(level)}` : `Brighter is larger RMS, from ${chartNumber(floor)} to ${chartNumber(level)}`}. ${largest}`);
    }
    canvas.addEventListener('pointermove', e => {
      const r = Math.min(data.tileRows - 1, Math.floor(e.offsetY / canvas.clientHeight * data.tileRows)), col = Math.min(data.tileCols - 1, Math.floor(e.offsetX / canvas.clientWidth * data.tileCols)), t = r * data.tileCols + col;
      readout.textContent = `${where(t)} · ${nonFinite.has(t) ? 'contains NaN or infinity' : data.peak[t] == null ? 'not read' : `peak ${number(data.peak[t])} · RMS ${number(data.rms[t])}`}`;
    });
    canvas.addEventListener('pointerleave', () => { readout.textContent = hint; });
    parent.append(disclosure('How the image is made', el('p', 'subtle', `Each pixel is one tile of the ${word}s. Signed peak colors a tile by its ${word} with the largest magnitude, so a single outlier stays visible; RMS colors it by root mean square. Leading dimensions form rows and trailing dimensions form columns, in row-major order. Tensors with more than ${integer(4194304)} values (${integer(262144)} for PyTorch) read evenly spaced runs of whole rows; tiles that were not read stay hatched.`)));
    const observer = new ResizeObserver(draw); observer.observe(box); state.chartObservers.push(observer); draw();
  }
  function imagePrompt(parent, text, label, type, name) { parent.append(add(el('div', 'image-prompt'), el('p', 'subtle', text), action(label, type, { name }, 'ghost'))); }
  function renderDetail() { const parent = document.getElementById('tensor-detail'); if (!parent || !state.drawerOpen) return; state.chartObservers.forEach(o => o.disconnect()); state.chartObservers = []; const wasOpen = parent.open, hadFocus = parent.contains(document.activeElement); parent.replaceChildren(); if (state.drawerMode === 'diff') renderDiffDetail(parent); else renderTensorDetail(parent); if (!wasOpen && state.drawerOpen) parent.showModal(); else if (wasOpen && hadFocus) parent.querySelector('#drawer-title')?.focus({ preventScroll: true }); updateBusy(); }
  function distributionChart(parent, domain, series) {
    if (domain?.min == null || domain.max == null || !series.some(s => s.counts.some(Boolean))) { parent.append(el('p', 'subtle', 'No finite values available for a distribution.')); return; }
    const chart = el('div', 'distribution-chart'), svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `${series.map(s => s.label).join(' and ')} distributions, using the same ${series[0].counts.length} bins from ${number(domain.min)} to ${number(domain.max)}. Height is percentage of finite analyzed values.`);
    chart.append(svg); parent.append(chart);
    const legend = el('div', 'distribution-legend');
    for (const s of series) legend.append(add(el('span'), el('i', s.className), document.createTextNode(s.label)));
    parent.append(legend);
    const data = disclosure('Distribution bin counts'), { wrap, body } = table(['Value interval', ...series.map(s => s.label)]);
    const boundary = i => domain.min * (1 - i / series[0].counts.length) + domain.max * (i / series[0].counts.length);
    series[0].counts.forEach((_, i) => body.append(add(el('tr'), el('td', '', `${number(boundary(i))} to ${number(boundary(i + 1))}${i === series[0].counts.length - 1 ? ' (inclusive)' : ''}`), ...series.map(s => el('td', 'numeric', integer(s.counts[i]))))));
    data.append(el('p', 'subtle', 'Intervals include the lower boundary; only the final interval includes its upper boundary. NaN and infinite values are excluded.'), wrap); parent.append(data);
    function draw() {
      const width = Math.max(220, chart.clientWidth), height = 205, left = 68, right = width - 12, top = 12, bottom = 157;
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.replaceChildren();
      const node = (tag, attrs, text) => { const n = document.createElementNS(svg.namespaceURI, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); if (text != null) n.textContent = text; svg.append(n); return n; };
      const normalized = series.map(s => { const total = s.counts.reduce((a, b) => a + b, 0); return s.counts.map(n => total ? n / total * 100 : 0); });
      const ceiling = Math.min(100, Math.max(1, ...normalized.flat()) * 1.1), y = n => bottom - n / ceiling * (bottom - top), binWidth = (right - left) / series[0].counts.length;
      for (const tick of [0, ceiling / 2, ceiling]) { node('line', { x1: left, x2: right, y1: y(tick), y2: y(tick), class: 'chart-grid' }); node('text', { x: left - 7, y: y(tick) + 4, 'text-anchor': 'end' }, `${new Intl.NumberFormat('en', { maximumSignificantDigits: 3 }).format(tick)}%`); }
      series.forEach((s, index) => {
        if (index === 1) {
          let path = '';
          normalized[index].forEach((n, i) => { const x = left + i * binWidth; path += `${i ? 'L' : 'M'}${x},${y(n)}L${x + binWidth},${y(n)}`; });
          node('path', { d: path, class: 'histogram-target' });
        } else normalized[index].forEach((n, i) => {
          const bar = node('rect', { x: left + i * binWidth + .5, y: y(n), width: Math.max(.5, binWidth - 1), height: bottom - y(n), class: s.className });
          const title = document.createElementNS(svg.namespaceURI, 'title'); title.textContent = `${number(boundary(i))} to ${number(boundary(i + 1))}: ${integer(s.counts[i])} ${s.label.toLowerCase()} values`; bar.append(title);
        });
      });
      const ticks = width > 340 ? [0, .5, 1] : [0, 1];
      for (const fraction of ticks) node('text', { x: left + (right - left) * fraction, y: bottom + 20, 'text-anchor': fraction === 0 ? 'start' : fraction === 1 ? 'end' : 'middle' }, chartNumber(domain.min * (1 - fraction) + domain.max * fraction));
      node('text', { x: (left + right) / 2, y: 200, 'text-anchor': 'middle' }, 'Value');
      node('text', { transform: `translate(12 ${(top + bottom) / 2}) rotate(-90)`, 'text-anchor': 'middle' }, 'Finite values (%)');
    }
    const observer = new ResizeObserver(draw); observer.observe(chart); state.chartObservers.push(observer); draw();
  }
  function renderTensorDetail(parent) {
    const t = state.model.tensors.find(t => t.name === state.selected); if (!t) { state.drawerOpen = false; if (parent.open) parent.close(); return; }
    const decoded = t.logicalShape != null;
    drawerHeader(parent, 'Tensor inspector', t.name); add(parent, add(el('div', 'tensor-facts'), el('span', 'badge', decoded ? t.quantization.scheme.toUpperCase() : t.dtype), el('span', 'shape', shape(decoded ? { shape: t.logicalShape } : t))), el('p', 'subtle', `${integer(t.logicalElements ?? t.elements)} ${decoded ? 'weight' : 'stored'} values · ${bytes(t.bytes)}${t.shard ? ` · ${t.shard}` : ''}`)); const data = state.stats.get(t.name);
    if (!t.analysisUnavailable) { parent.append(el('h3', '', 'Weight image')); const picture = state.images.get(t.name); if (picture) imageSection(parent, picture, 'tensor'); else imagePrompt(parent, 'See the whole tensor as pixels: outlier channels, head structure and empty regions.', 'Draw weight image', 'image', t.name); }
    if (t.analysisUnavailable) parent.append(add(el('div', 'analyze-prompt'), el('h3', '', 'Storage metadata available'), el('p', 'subtle', t.analysisUnavailable)));
    else if (!data) parent.append(add(el('div', 'analyze-prompt'), el('div', 'eyebrow', 'WEIGHT HEALTH'), el('h3', '', 'Check the values inside this tensor.'), el('p', 'subtle', 'See the distribution, range, and whether any values are NaN or infinite.'), action('Analyze weights', 'stats', { name: t.name }, 'primary full'), el('p', 'subtle', `Up to ${integer(state.settings.sampleSize)} values will be checked. Small tensors are checked in full.`)));
    else {
      add(parent, el('h3', '', 'Weight distribution'), el('p', 'subtle', `${data.exact ? 'Full tensor' : 'Sampled estimate'} · ${integer(data.sampled)} / ${integer(data.total)} values`));
      distributionChart(parent, data.histogramRange || data, [{ label: 'Weights', counts: data.histogram, className: 'histogram-baseline' }]);
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
    drawerHeader(parent, 'Tensor difference', row.name); add(parent, badge(row.status), pairs([['Baseline', row.left ? `${row.left.dtype} · ${shape(row.left)}` : 'Not present'], ['Target', row.right ? `${row.right.dtype} · ${shape(row.right)}` : 'Not present']])); if (row.baseName) parent.append(pairs([['Baseline tensor', row.baseName], ['Target tensor', row.targetName]])); for (const note of row.layout || []) parent.append(el('p', 'subtle', note)); if (row.dtypeChanged) parent.append(el('p', 'subtle', 'The stored precision changed between checkpoints.')); const m = row.metrics;
    if (m) add(parent, el('h3', '', 'How much did it change?'), el('p', 'subtle', `${m.exact ? 'Full comparison' : 'Sampled comparison'} · ${integer(m.sampled)} / ${integer(m.total)} values`), tiles([['Mean absolute change', number(m.meanAbs)], ['Largest absolute change', number(m.maxAbs)], ['RMSE', number(m.rmse)], ['Cosine similarity', number(m.cosine)]]), disclosure('More comparison details', pairs([['Relative L2', number(m.relativeL2)], ['Changed values', integer(m.changedValues)], ['Non-finite pairs', integer(m.nonFinite)], ['Non-finite mismatches', integer(m.nonFiniteMismatch)], ['Absolute tolerance', number(m.tolerance)]]), el('p', 'subtle', 'Full match means all values agree within tolerance. Sample matches only describe the positions analyzed. Relative L2 and cosine are undefined for zero norms.')));
    else { const descriptions = { added: 'This tensor is present only in the target checkpoint.', removed: 'This tensor is present only in the baseline checkpoint.', 'shape-changed': 'The shape changed. Corresponding values cannot be compared directly.', 'basis-changed': 'These tensors use different numeric units, so a direct numerical delta would be misleading.' }; parent.append(el('p', 'subtle', descriptions[row.status] || 'Compare corresponding weight values to measure the difference.')); }
    if (!['added', 'removed', 'shape-changed', 'basis-changed'].includes(row.status)) { parent.append(el('h3', '', 'Change image')); const picture = state.diffImages.get(row.name); if (picture) imageSection(parent, picture, 'diff'); else imagePrompt(parent, 'See where the weights changed: target minus baseline, tile by tile.', 'Draw change image', 'diffImage', row.name); }
    if (m?.distributions) { parent.append(el('h3', '', 'Baseline & target distributions'), el('p', 'subtle', 'Shared bin boundaries and scale. Baseline is filled; target is a line.')); distributionChart(parent, m.distributions, [{ label: 'Baseline', counts: m.distributions.baseline, className: 'histogram-baseline' }, { label: 'Target', counts: m.distributions.target, className: 'histogram-target' }]); }
    if (row.error) parent.append(el('div', 'notice error', row.error)); if (!['added', 'removed', 'shape-changed', 'basis-changed'].includes(row.status)) parent.append(action(m ? 'Analyze again' : 'Analyze this tensor', 'analyzeDiff', { name: row.name }, 'primary full'));
  }
  function renderComparison(content) {
    const pane = el('section', 'compare-pane');
    if (!state.target) { add(pane, heading('See what changed', 'Compare this model with another checkpoint.'), add(el('div', 'compare-start'), add(el('div', 'compare-step'), el('span', 'step-number', '1'), el('div', 'eyebrow', 'BASELINE'), el('h3', '', state.model.name), el('p', 'subtle', 'The model you have open')), el('span', 'step-arrow', '→'), add(el('div', 'compare-step target-step'), el('span', 'step-number', '2'), el('div', 'eyebrow', 'TARGET'), el('h3', '', 'Choose another checkpoint'), el('p', 'subtle', 'Find added, removed, and changed weights'), action('Choose target checkpoint', 'compare', {}, 'primary'), action('Choose Ollama model', 'compareOllama', {}, 'ghost')))); content.append(pane); return; }
    const overview = view.diffOverview(state.rows);
    const analyze = action(overview.analyzed ? 'Analyze next batch' : 'Analyze weight changes', 'analyzeDiff', {}, 'primary'); analyze.id = 'analyze-comparison';
    add(pane, heading('Checkpoint comparison', 'Compare drift across layer groups and tensor roles. Select a cell to inspect it.', add(el('div', 'actions'), action('Change target', 'compare', {}, 'ghost'), analyze)), add(el('div', 'compare-files'), el('span', '', state.model.name), el('span', 'file-arrow', '→'), el('span', '', state.target.name)), state.matching && state.matching.kind !== 'exact' && el('p', 'subtle', state.matching.description));
    const summary = el('div'); summary.id = 'diff-summary'; const map = el('div'); map.id = 'change-map';
    add(pane, summary, add(el('div', 'toolbar map-toolbar'), input('Find a layer or tensor role…', state.mapQuery, v => { state.mapQuery = v; state.mapPage = 0; state.rolePage = 0; renderDiffOverview(); })), map);
    const detail = el('details', 'diff-disclosure'); detail.id = 'diff-detail-list'; detail.open = state.diffDetailsOpen; detail.addEventListener('toggle', () => { state.diffDetailsOpen = detail.open; }); detail.append(el('summary', '', 'Tensor-by-tensor results'));
    const toolbar = el('div', 'toolbar'); add(toolbar, input('Search differences…', state.diffQuery, v => { state.diffQuery = v; state.diffPage = 0; renderDiffResults(); }), select([['attention', 'Changes & not analyzed'], ['', 'All tensors'], ...Object.entries(labels)], state.diffFilter, v => { state.diffFilter = v; state.diffPage = 0; renderDiffResults(); }, 'Filter comparison status')); const results = el('div'); results.id = 'diff-results'; add(detail, toolbar, results);
    add(pane, detail, disclosure('How comparison works', el('p', 'subtle', `${state.matching?.description || 'Tensor names are matched exactly.'} Each batch compares up to ${integer(state.settings.maxDiffTensors)} compatible tensors, with absolute tolerance ${number(state.settings.absoluteTolerance)}. Changes are target minus baseline. Sample matches do not prove that the whole tensor is unchanged.`))); content.append(pane); renderDiffOverview(); renderDiffResults();
  }
  function renderComparisonTab() { const n = document.getElementById('content'); if (n) { n.replaceChildren(); renderComparison(n); } }
  function revealDiffResults() { const n = document.getElementById('diff-detail-list'); if (!n) return; n.querySelector('summary')?.focus({ preventScroll: true }); n.scrollIntoView({ block: 'start', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' }); }
  function renderDiffOverview() {
    const summary = document.getElementById('diff-summary'), map = document.getElementById('change-map'); if (!summary || !map) return;
    const focusedKey = map.contains(document.activeElement) ? document.activeElement.dataset.matrixKey : null;
    summary.replaceChildren(); map.replaceChildren(); const d = view.diffOverview(state.rows);
    const analyze = document.getElementById('analyze-comparison');
    if (analyze) { analyze.textContent = d.pending ? d.analyzed ? 'Analyze next batch' : 'Analyze weight changes' : d.unavailable ? 'Retry unavailable' : d.analyzed ? 'Analysis complete' : 'No comparable tensors'; analyze.dataset.inactive = String(!d.pending && !d.unavailable); analyze.disabled = state.busy || analyze.dataset.inactive === 'true'; }
    const title = d.changes ? `${d.changes} tensors have changes` : d.pending ? 'Ready to compare weight values' : d.unavailable ? 'Some tensors could not be analyzed' : d.counts['sample-match'] ? 'No changes found in analyzed values' : 'All tensors match within tolerance';
    summary.append(add(el('div', 'diff-headline'), el('h3', '', title), el('p', 'subtle', `${d.analyzed} numerically analyzed · ${d.pending} not analyzed${d.unavailable ? ` · ${d.unavailable} unavailable` : ''}`)));
    const strip = el('div', 'status-strip'); strip.setAttribute('role', 'img'); strip.setAttribute('aria-label', Object.entries(d.counts).map(([s, count]) => `${count} ${labels[s]}`).join(', ')); const legend = el('div', 'status-legend');
    for (const [status, count] of Object.entries(d.counts)) { const segment = el('span', status); segment.style.flexGrow = count; strip.append(segment); const b = button('', () => { state.diffGroup = ''; state.diffRole = ''; state.diffQuery = ''; state.diffFilter = status; state.diffPage = 0; state.diffDetailsOpen = true; renderComparisonTab(); revealDiffResults(); }, 'legend-button'); add(b, el('i', `legend-dot ${status}`), document.createTextNode(`${count} ${labels[status]}`)); legend.append(b); } add(summary, strip, legend);
    renderChangeMap(map);
    if (focusedKey) [...map.querySelectorAll('[data-matrix-key]')].find(b => b.dataset.matrixKey === focusedKey)?.focus({ preventScroll: true });
  }
  const codes = { unverified: '…', unavailable: '×', added: '+', removed: '−', 'shape-changed': 'S', 'basis-changed': 'U' };
  const driftLabel = v => v > 0 && v * 100 < .01 ? `${chartNumber(v * 100)}%` : percent(v);
  function lightTheme() { const c = getComputedStyle(document.body).backgroundColor.match(/[\d.]+/g)?.map(Number); return Boolean(c) && (c[3] ?? 1) > 0 && (.2126 * c[0] + .7152 * c[1] + .0722 * c[2]) / 255 > .5; }
  function openMapRows(group, role = '') { state.diffGroup = group; state.diffRole = role; state.diffFilter = ''; state.diffQuery = ''; state.diffPage = 0; state.diffDetailsOpen = true; renderComparisonTab(); revealDiffResults(); }
  // One drift map cell. A name collision keeps every tensor and is colored by its largest finite drift.
  function mapCell(group, role, rows) {
    const row = rows[0], m = rows.length === 1 ? row.metrics : null, finite = rows.map(r => r.metrics?.relativeL2).filter(Number.isFinite);
    const relative = finite.length ? Math.max(...finite) : null, status = rows.length > 1 ? 'mixed' : row.status;
    const main = rows.length > 1 ? `${rows.length} tensors` : m ? relative == null ? 'n/a' : percent(relative) : codes[status] || '—';
    const note = rows.length > 1 ? 'Select to list' : m ? `${m.exact ? 'Full' : 'Sample'}${row.dtypeChanged ? ' · P' : ''}${m.nonFiniteMismatch ? ' · !' : ''}` : `${labels[status]}${row.dtypeChanged ? ' · P' : ''}`;
    const description = rows.length > 1 ? `${group.key}, ${role}, ${rows.length} tensors${relative == null ? '' : `, largest relative L2 ${percent(relative)}`}` : `${row.name}: ${labels[row.status]}${row.dtypeChanged ? ', precision changed' : ''}${m ? `, ${relative == null ? 'relative L2 undefined for zero baseline norm or non-finite arithmetic' : `${percent(relative)} relative L2`}, ${m.exact ? 'full comparison' : 'sampled estimate'}, ${integer(m.sampled)} of ${integer(m.total)} values${m.nonFiniteMismatch ? `, ${integer(m.nonFiniteMismatch)} non-finite mismatches` : ''}` : ''}${row.error ? `. ${row.error}` : ''}`;
    return { rows, status, relative, main, note, detail: `${main} · ${note}`, description, analyzed: rows.some(r => r.metrics), precision: rows.some(r => r.dtypeChanged), nonFinite: rows.some(r => r.metrics?.nonFiniteMismatch),
      open: () => rows.length === 1 ? showDifference(row.name) : openMapRows(group.key, role) };
  }
  function renderChangeMap(parent) {
    state.gridObservers.get('fingerprint')?.disconnect();
    const all = view.comparisonMatrix(state.rows), query = state.mapQuery.toLowerCase(), scale = view.driftScale(all, state.mapScale), light = lightTheme();
    const matrix = query ? view.comparisonMatrix(state.rows.filter(r => view.matchesQuery(r, query))) : all, fingerprint = state.mapView === 'fingerprint';
    const viewChoice = select([['fingerprint', 'Fingerprint'], ['table', 'Table']], state.mapView, v => { state.mapView = v; state.mapPage = 0; state.rolePage = 0; persist(); renderDiffOverview(); }, 'Drift map view');
    const scaleChoice = select([['log', 'Log scale'], ['linear', 'Linear scale']], state.mapScale, v => { state.mapScale = v; persist(); renderDiffOverview(); }, 'Drift map scale');
    viewChoice.dataset.matrixKey = 'map-view'; scaleChoice.dataset.matrixKey = 'map-scale';
    parent.append(add(el('div', 'map-heading'), add(el('div'), el('h3', '', 'Layer drift map'), el('p', 'map-hint', `${integer(matrix.groups.length)} layer groups × ${integer(matrix.roles.length)} tensor roles · ${!scale.ticks.length ? 'No drift measured yet.' : `One ${scale.mode} relative L2 scale across all pages and filters.${state.mapScale === 'log' && scale.mode === 'linear' ? ' A log scale needs at least two different drift values.' : ''}`}`)), add(el('div', 'map-controls'), viewChoice, scaleChoice)));
    if (!matrix.groups.length) { parent.append(el('p', 'empty', 'No tensors match the map search.')); return; }
    const pages = fingerprint ? renderFingerprint(parent, matrix, scale, light) : renderHeatmapTable(parent, matrix, scale, light);
    parent.append(driftLegend(matrix, scale, light, fingerprint));
    parent.append(el('p', 'subtle', `Color shows the amount of drift, not a health score.${scale.mode === 'log' ? ' On the log scale, equal color steps are equal ratios of drift.' : ''} Relative L2 is undefined for a zero baseline norm. Full matches use the configured tolerance; sample matches do not prove whole-tensor equality.`));
    pager(parent, matrix.groups.length, state.mapPage, p => { state.mapPage = p; renderDiffOverview(); }, pages.groups, 'groups');
    pager(parent, matrix.roles.length, state.rolePage, p => { state.rolePage = p; renderDiffOverview(); }, pages.roles, 'roles');
  }
  function driftLegend(matrix, scale, light, fingerprint) {
    const legend = el('div', 'heatmap-legend');
    if (scale.ticks.length) { const ramp = el('i', 'drift-ramp'); ramp.style.background = `linear-gradient(to right, ${[0, .25, .5, .75, 1].map(t => view.heatColor(t, light)).join(', ')})`; add(legend, el('span', '', `Relative L2${scale.mode === 'log' ? ' (log)' : ''}`), el('span', '', driftLabel(scale.ticks[0])), ramp, el('span', '', driftLabel(scale.ticks[2]))); }
    if (!fingerprint) { add(legend, el('span', '', '… Not analyzed'), el('span', '', '× Unavailable'), el('span', '', '+ Added'), el('span', '', '− Removed'), el('span', '', 'S Shape changed'), el('span', '', 'U Units differ'), el('span', '', 'P Precision changed'), el('span', '', '! Non-finite mismatch'), el('span', '', '— No tensor')); return legend; }
    const swatch = (cls, text) => add(el('span'), el('i', `swatch ${cls}`), document.createTextNode(text));
    add(legend, swatch('match', 'No drift'), swatch('undefined-drift', 'n/a'), swatch('unverified', 'Not analyzed'), swatch('unavailable', 'Unavailable'), swatch('added', 'Added'), swatch('removed', 'Removed'), swatch('structural', 'Shape or units changed'), swatch('precision', 'Precision changed'), swatch('non-finite', 'Non-finite mismatch'), swatch('no-tensor', 'No tensor'));
    let full = 0, sampled = 0; for (const g of matrix.groups) for (const rows of g.cells.values()) for (const r of rows) if (r.metrics) r.metrics.exact ? full++ : sampled++;
    return add(el('div'), legend, (full || sampled) && el('p', 'map-hint', `${integer(full)} full comparisons · ${integer(sampled)} sampled estimates. Hover or focus a cell for its coverage.`));
  }
  function renderHeatmapTable(parent, matrix, scale, light) {
    const rowSize = 24, columnSize = 10;
    state.mapPage = Math.min(state.mapPage, Math.max(0, Math.ceil(matrix.groups.length / rowSize) - 1));
    state.rolePage = Math.min(state.rolePage, Math.max(0, Math.ceil(matrix.roles.length / columnSize) - 1));
    const roles = matrix.roles.slice(state.rolePage * columnSize, (state.rolePage + 1) * columnSize);
    const wrap = el('div', 'table-wrap heatmap-wrap'), t = el('table', 'heatmap'), head = el('tr'), body = el('tbody');
    t.style.setProperty('--roles', roles.length);
    t.setAttribute('aria-label', 'Relative L2 checkpoint drift by tensor name group and role');
    const corner = el('th', '', 'Layer group / tensor role'); corner.scope = 'col'; head.append(corner);
    for (const role of roles) { const th = el('th'); th.append(brokenName(role.replace(/\.weight$/, ''), 'role-name')); th.scope = 'col'; th.title = role; th.setAttribute('aria-label', role); head.append(th); }
    add(t, add(el('thead'), head), body); wrap.append(t); parent.append(wrap);
    for (const group of matrix.groups.slice(state.mapPage * rowSize, (state.mapPage + 1) * rowSize)) {
      const tr = el('tr'), label = el('th'); label.scope = 'row'; label.append(button(group.key, () => openMapRows(group.key), 'tensor-link')); tr.append(label);
      for (const role of roles) {
        const td = el('td'), rows = group.cells.get(role);
        if (!rows) td.append(el('span', 'missing-cell', '—'));
        else {
          const c = mapCell(group, role, rows), cell = button('', c.open, `heatmap-cell ${c.status}`);
          cell.dataset.matrixKey = rows.map(r => r.name).join('\n');
          add(cell, el('span', 'heatmap-value', c.main), el('span', 'heatmap-note', c.note));
          if (c.relative > 0) { const position = scale.position(c.relative); cell.style.setProperty('--heat-color', view.heatColor(position, light)); cell.classList.add('heated', `ink-${view.heatInk(position, light)}`); }
          cell.title = c.description; cell.setAttribute('aria-label', c.description); td.append(cell);
        }
        tr.append(td);
      }
      body.append(tr);
    }
    return { groups: rowSize, roles: columnSize };
  }
  // A canvas grid of layer groups (columns, left to right) by tensor roles (rows, top to bottom), so a whole model reads as one
  // image. Each item brings its tooltip lines, screen-reader description and action; paint draws its cell.
  function gridCanvas(parent, { key, label, roledescription, groups, roles, cells, paint }) {
    state.gridObservers.get(key)?.disconnect();
    const wrap = el('div', 'fingerprint'), canvas = el('canvas'), focus = el('div', 'fingerprint-focus'), tip = el('div', 'fingerprint-tip'), live = el('div', 'visually-hidden');
    wrap.tabIndex = 0; wrap.dataset.matrixKey = key; wrap.setAttribute('role', 'group'); wrap.setAttribute('aria-roledescription', roledescription);
    wrap.setAttribute('aria-label', `${label}: ${integer(groups.length)} layer groups as columns, ${integer(roles.length)} tensor roles as rows. Arrow keys move between cells; Enter opens the selected cell.`);
    canvas.setAttribute('aria-hidden', 'true'); live.setAttribute('aria-live', 'polite'); focus.hidden = true; tip.hidden = true;
    add(wrap, canvas, focus, tip, live); parent.append(wrap);
    const saved = state.gridCursors[key];
    let layout = null, cursor = (() => { const c = groups.findIndex(g => g.key === saved?.group), r = roles.indexOf(saved?.role); return c >= 0 && r >= 0 ? { c, r } : null; })();
    const at = (x, y) => { if (!layout) return null; const c = Math.floor((x - layout.left) / layout.cell), r = Math.floor((y - layout.top) / layout.row); return x >= layout.left && y >= layout.top && c < groups.length && r < roles.length ? { c, r } : null; };
    function mark(p) {
      if (!p || !layout) { focus.hidden = true; tip.hidden = true; return; }
      const cell = cells[p.c][p.r], x = layout.left + p.c * layout.cell, y = layout.top + p.r * layout.row;
      Object.assign(focus.style, { left: `${x - 2}px`, top: `${y - 2}px`, width: `${layout.cell + 3}px`, height: `${layout.row + 3}px` }); focus.hidden = false;
      tip.replaceChildren(el('strong', '', `${groups[p.c].key} · ${roles[p.r]}`), el('span', '', cell ? cell.detail : 'No tensor in this group and role')); tip.hidden = false;
      const right = x + layout.cell + 10, left = right + tip.offsetWidth > wrap.clientWidth ? Math.max(0, x - tip.offsetWidth - 10) : right;
      Object.assign(tip.style, { left: `${left}px`, top: `${Math.max(0, Math.min(y - 6, wrap.clientHeight - tip.offsetHeight))}px` });
    }
    function move(p) { cursor = p; state.gridCursors[key] = { group: groups[p.c].key, role: roles[p.r] }; mark(p); live.textContent = cells[p.c][p.r]?.description || `${groups[p.c].key}, ${roles[p.r]}: no tensor`; }
    function draw() {
      const style = getComputedStyle(wrap), css = name => style.getPropertyValue(name).trim(), mono = css('--vscode-editor-font-family') || 'monospace', ratio = window.devicePixelRatio || 1;
      const width = Math.max(240, wrap.clientWidth), gutter = Math.round(Math.min(200, Math.max(96, width * .2))), top = 2;
      const cell = Math.max(3, Math.min(56, Math.floor((width - gutter - 4) / groups.length))), row = Math.max(14, Math.min(28, Math.round(cell * .7))), gap = cell >= 6 ? 1 : 0;
      const g = canvas.getContext('2d'); g.font = `11px ${mono}`;
      const fit = (text, max) => { if (g.measureText(text).width <= max) return text; let s = text; while (s.length > 1 && g.measureText(`…${s}`).width > max) s = s.slice(1); return `…${s}`; };
      const short = groups.map(group => { const parts = group.key.split('.'); return parts.findLast(p => /^\d+$/.test(p)) ?? parts.at(-1); });
      const rotated = cell >= 13, axis = rotated ? Math.min(96, 14 + Math.max(...short.map(s => g.measureText(s).width))) : 24, height = top + roles.length * row + axis;
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); canvas.style.height = `${height}px`;
      g.setTransform(ratio, 0, 0, ratio, 0, 0); g.clearRect(0, 0, width, height); g.font = `11px ${mono}`; g.textBaseline = 'middle';
      const muted = css('--muted'), hatch = document.createElement('canvas');
      hatch.width = hatch.height = 6; const h = hatch.getContext('2d'); h.strokeStyle = muted; h.globalAlpha = .55; h.beginPath(); h.moveTo(0, 6); h.lineTo(6, 0); h.stroke();
      const tools = { css, muted, pattern: g.createPattern(hatch, 'repeat'), light: lightTheme(),
        // A tinted, outlined cell for structural states; a cross marks unavailable analysis.
        outline(x, y, w, hgt, color, cross) { g.globalAlpha = .25; g.fillStyle = color; g.fillRect(x, y, w, hgt); g.globalAlpha = 1; g.strokeStyle = color; g.lineWidth = 1; g.strokeRect(x + .5, y + .5, w - 1, hgt - 1); if (cross && w >= 7) { g.beginPath(); g.moveTo(x + 2, y + 2); g.lineTo(x + w - 2, y + hgt - 2); g.moveTo(x + w - 2, y + 2); g.lineTo(x + 2, y + hgt - 2); g.stroke(); } },
        corner(x, y, w, hgt, color, where) { if (w < 7) return; const n = Math.min(6, w / 2, hgt / 2); g.globalAlpha = 1; g.fillStyle = color; g.beginPath(); if (where === 'top') { g.moveTo(x + w - n, y); g.lineTo(x + w, y); g.lineTo(x + w, y + n); } else { g.moveTo(x, y + hgt - n); g.lineTo(x, y + hgt); g.lineTo(x + n, y + hgt); } g.fill(); } };
      g.fillStyle = muted; g.textAlign = 'right';
      roles.forEach((role, r) => g.fillText(fit(role.replace(/\.weight$/, ''), gutter - 14), gutter - 8, top + r * row + row / 2));
      cells.forEach((column, c) => column.forEach((item, r) => { if (item) { paint(g, item, gutter + c * cell, top + r * row, cell - gap, row - gap, tools); g.globalAlpha = 1; } }));
      g.fillStyle = muted; const base = top + roles.length * row + 6;
      if (rotated) short.forEach((text, c) => { g.save(); g.translate(gutter + c * cell + cell / 2, base); g.rotate(-Math.PI / 2); g.textAlign = 'right'; g.fillText(fit(text, axis - 10), 0, 0); g.restore(); });
      else {
        const step = [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 50, 64, 100, 128, 200, 256].find(n => n * cell >= 28) || 512; g.textAlign = 'center';
        short.forEach((text, c) => { if (/^\d+$/.test(text) && Number(text) % step === 0) g.fillText(text, gutter + c * cell + cell / 2, base + 6); });
      }
      layout = { left: gutter, top, cell, row };
      if (cursor && document.activeElement === wrap) mark(cursor);
    }
    wrap.addEventListener('keydown', e => {
      const step = { ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowDown: [0, 1], ArrowUp: [0, -1] }[e.key];
      if (step) { e.preventDefault(); move(cursor ? { c: Math.max(0, Math.min(groups.length - 1, cursor.c + step[0])), r: Math.max(0, Math.min(roles.length - 1, cursor.r + step[1])) } : { c: 0, r: 0 }); }
      else if ((e.key === 'Enter' || e.key === ' ') && cursor && cells[cursor.c][cursor.r]) { e.preventDefault(); cells[cursor.c][cursor.r].open(); }
      else if (e.key === 'Escape') tip.hidden = true;
    });
    wrap.addEventListener('focus', () => { if (cursor) mark(cursor); else { const c = cells.findIndex(column => column.some(Boolean)); move({ c: Math.max(0, c), r: Math.max(0, c < 0 ? 0 : cells[c].findIndex(Boolean)) }); } });
    wrap.addEventListener('blur', () => mark(null));
    canvas.addEventListener('pointermove', e => { const p = at(e.offsetX, e.offsetY); canvas.style.cursor = p && cells[p.c][p.r] ? 'pointer' : 'default'; mark(p || (document.activeElement === wrap ? cursor : null)); });
    canvas.addEventListener('pointerleave', () => mark(document.activeElement === wrap ? cursor : null));
    canvas.addEventListener('click', e => { const p = at(e.offsetX, e.offsetY); if (p && cells[p.c][p.r]) { move(p); cells[p.c][p.r].open(); } });
    const observer = new ResizeObserver(draw); state.gridObservers.set(key, observer); observer.observe(wrap); draw();
  }
  // Grids page up to 160 layer groups and 48 tensor roles at a time.
  function gridPage(matrix, pageKey, roleKey) {
    const columnSize = 160, rowSize = 48;
    state[pageKey] = Math.min(state[pageKey], Math.max(0, Math.ceil(matrix.groups.length / columnSize) - 1));
    state[roleKey] = Math.min(state[roleKey], Math.max(0, Math.ceil(matrix.roles.length / rowSize) - 1));
    return { groups: matrix.groups.slice(state[pageKey] * columnSize, (state[pageKey] + 1) * columnSize), roles: matrix.roles.slice(state[roleKey] * rowSize, (state[roleKey] + 1) * rowSize), size: { groups: columnSize, roles: rowSize } };
  }
  function renderFingerprint(parent, matrix, scale, light) {
    const { groups, roles, size } = gridPage(matrix, 'mapPage', 'rolePage');
    const cells = groups.map(g => roles.map(r => g.cells.has(r) ? mapCell(g, r, g.cells.get(r)) : null));
    gridCanvas(parent, { key: 'fingerprint', label: 'Drift fingerprint', roledescription: 'drift fingerprint', groups, roles, cells, paint(g, item, x, y, w, h, t) {
      if (item.relative != null) { g.globalAlpha = item.relative > 0 ? 1 : .22; g.fillStyle = item.relative > 0 ? view.heatColor(scale.position(item.relative), light) : t.muted; g.fillRect(x, y, w, h); }
      else if (item.analyzed) { g.globalAlpha = .5; g.fillStyle = t.muted; g.fillRect(x, y, w, h); }
      else if (['unverified', 'mixed'].includes(item.status)) { g.fillStyle = t.pattern; g.fillRect(x, y, w, h); }
      else t.outline(x, y, w, h, item.status === 'added' ? t.css('--mint') : ['removed', 'unavailable'].includes(item.status) ? t.css('--red') : t.css('--accent'), item.status === 'unavailable');
      if (item.precision) t.corner(x, y, w, h, t.css('--accent'), 'top');
      if (item.nonFinite) t.corner(x, y, w, h, t.css('--red'), 'bottom');
    } });
    return size;
  }
  function renderDiffResults() {
    const parent = document.getElementById('diff-results'); if (!parent) return; parent.replaceChildren(); if (state.diffGroup) parent.append(add(el('div', 'breadcrumb'), button('All groups', () => { state.diffGroup = ''; state.diffRole = ''; state.diffPage = 0; renderDiffResults(); renderDiffOverview(); }, 'text-button'), el('span', '', '/'), brokenName(`${state.diffGroup}${state.diffRole ? ` / ${state.diffRole}` : ''}`, 'crumb-current')));
    const rows = view.diffRows(state.rows, { query: state.diffQuery, status: state.diffFilter, group: state.diffGroup, role: state.diffRole }); state.diffPage = Math.min(state.diffPage, Math.max(0, Math.ceil(rows.length / PAGE) - 1));
    if (!rows.length) { parent.append(add(el('div', 'empty'), el('p', '', 'No tensors match this view.'), button('Show all tensors', () => { state.diffGroup = ''; state.diffRole = ''; state.diffFilter = ''; state.diffQuery = ''; renderComparisonTab(); }, 'ghost'))); return; }
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
  function shardInfo(source, format) {
    const { wrap, body } = table(['Shard', 'Tensors', 'Size']);
    for (const shard of source.shards.slice(0, 200)) body.append(add(el('tr'), el('td', 'name-cell', shard.file), el('td', 'numeric', integer(shard.tensors)), el('td', 'numeric', bytes(shard.bytes))));
    return disclosure(`${format === 'pytorch' ? 'PyTorch' : 'Safetensors'} shards (${integer(source.shards.length)})`, source.indexPath ? el('p', 'path', source.indexPath) : el('p', 'subtle', 'No index file was found; shards were grouped by file name.'), wrap, source.shards.length > 200 && el('p', 'subtle', `Showing 200 of ${integer(source.shards.length)} shards. Export JSON for the full list.`));
  }
  function renderInfo(content) {
    const pane = el('section', 'info-pane'), s = state.model.summary; pane.append(heading('Model details', 'File metadata, precision distribution, and exact stored counts.')); const grid = el('div', 'info-grid');
    add(grid, add(el('div', 'card'), el('h3', '', 'Checkpoint'), pairs([['Stored values', integer(s.parameters)], ['Tensors', integer(s.tensorCount)], [state.model.source?.kind === 'sharded' ? 'Shard files' : state.model.source ? 'Primary weight files' : 'File size', bytes(s.fileBytes)], ['Tensor storage', `${bytes(s.tensorBytes)}${s.unknownByteSizes ? ' (partial)' : ''}`]]), el('p', 'path', state.model.path)), add(el('div', 'card'), el('h3', '', 'Precision distribution'), precisionChart(s)));
    add(pane, grid, state.model.source?.kind === 'ollama' && ollamaInfo(state.model.source), state.model.source?.kind === 'sharded' && shardInfo(state.model.source, state.model.format), disclosure('Model metadata', el('pre', '', JSON.stringify(state.model.metadata, null, 2))), disclosure('Interpretation & sampling', el('p', 'subtle', 'Layer groups and the tree follow tensor names. They do not recover forward-pass order. Counts describe stored values, including buffers and aliases, rather than unique trainable parameters. ONNX counts include supported initializers and Constant tensors in parsed graphs.'), el('p', 'subtle', `Statistics sample up to ${integer(state.settings.sampleSize)} values in evenly spaced windows. Small tensors are checked in full. Samples can miss values outside the sampled positions.`)), action('PyTorch setup instructions', 'setup')); content.append(pane);
  }
  function renderOperators(content) { const pane = el('section', 'operators-pane'); pane.append(heading('ONNX operators', 'Operator order and input/output names from the serialized graph. Subgraphs use scoped names.')); const { wrap, body } = table(['Operator', 'Type', 'Inputs', 'Outputs']); for (const n of state.model.nodes.slice(0, 1000)) body.append(add(el('tr'), el('td', 'name-cell', n.name), el('td', 'dtype', n.opType), el('td', 'name-cell', n.inputs.join(', ')), el('td', 'name-cell', n.outputs.join(', ')))); pane.append(wrap); if (state.model.nodes.length > 1000) pane.append(el('p', 'subtle', 'Showing 1,000 operators. Export JSON for the complete list.')); content.append(pane); }
  function updateBusy() {
    const line = document.getElementById('busy'); if (line) line.hidden = !state.busy; const label = document.getElementById('busy-label'); if (label) label.textContent = state.busyLabel || ''; document.querySelectorAll('[data-action]').forEach(b => { b.disabled = (state.busy && b.dataset.action !== 'cancel') || b.dataset.inactive === 'true'; });
    const drawer = document.getElementById('tensor-detail'); if (drawer?.open && state.busy && !document.getElementById('drawer-progress')) { const progress = el('div', 'drawer-progress'); progress.id = 'drawer-progress'; progress.setAttribute('role', 'status'); add(progress, el('span', '', state.busyLabel), action('Cancel', 'cancel')); drawer.append(progress); } else if (!state.busy) document.getElementById('drawer-progress')?.remove();
  }
  window.addEventListener('message', event => {
    const m = event.data;
    if (m.type === 'model') { clearTimeout(state.diffTimer); state.diffTimer = null; state.model = m.model; state.groups = view.groups(m.model.tensors); state.settings = m.settings; state.stats.clear(); state.scanErrors.clear(); state.modelPage = 0; state.modelRolePage = 0; state.images.clear(); state.diffImages.clear(); state.rows = []; state.target = null; state.group = ''; state.page = 0; state.groupPage = 0; state.drawerOpen = false; buildShell(); }
    else if (m.type === 'statistics' || m.type === 'scanError') {
      if (m.type === 'statistics') state.stats.set(m.name, m.data); else state.scanErrors.set(m.name, m.message);
      if (m.total) { state.busyLabel = `Scanning weights ${integer(m.completed)} / ${integer(m.total)}…`; updateBusy(); }
      if (state.drawerMode === 'tensor' && state.selected === m.name) renderDetail(); scheduleModelMap();
    }
    else if (m.type === 'scanDone') { clearTimeout(state.modelMapTimer); state.modelMapTimer = null; refreshModelMap(); }
    else if (m.type === 'tensorImage') { state.images.set(m.name, m.data); if (state.drawerMode === 'tensor' && state.selected === m.name) renderDetail(); }
    else if (m.type === 'diffImage') { state.diffImages.set(m.name, m.data); if (state.drawerMode === 'diff' && state.selectedDiff === m.name) renderDetail(); }
    else if (m.type === 'comparison') { clearTimeout(state.diffTimer); state.diffTimer = null; state.diffImages.clear(); document.getElementById('tensor-detail')?.close(); state.drawerOpen = false; state.target = m.target; state.rows = m.rows; state.matching = m.matching; state.tab = 'compare'; state.diffPage = 0; state.diffGroup = ''; state.diffRole = ''; state.mapQuery = ''; state.mapPage = 0; state.rolePage = 0; state.diffFilter = 'attention'; state.diffDetailsOpen = false; persist(); renderTab(); }
    else if (m.type === 'diffRow') { const index = state.rows.findIndex(r => r.name === m.row.name); if (index >= 0) state.rows[index] = m.row; state.busyLabel = `Comparing tensor ${m.completed} / ${m.total}…`; updateBusy(); if (state.drawerMode === 'diff' && state.selectedDiff === m.row.name) renderDetail(); if (!state.diffTimer) state.diffTimer = setTimeout(() => { state.diffTimer = null; renderDiffOverview(); renderDiffResults(); }, 150); }
    else if (m.type === 'diffDone') { clearTimeout(state.diffTimer); state.diffTimer = null; renderDiffOverview(); renderDiffResults(); }
    else if (m.type === 'busy') { state.busy = true; state.busyLabel = m.label; for (const id of ['error', 'drawer-error']) { const n = document.getElementById(id); if (n) n.hidden = true; } updateBusy(); }
    else if (m.type === 'idle') { state.busy = false; updateBusy(); }
    else if (m.type === 'error') showError(m.message);
    else if (m.type === 'checkpointSaved') { state.savedPath = m.path; const n = document.getElementById('error'); if (n) { n.className = 'notice'; n.setAttribute('role', 'status'); n.textContent = `Checkpoint copy saved: ${m.path}`; n.hidden = false; } }
  });
  // VS Code swaps the body theme class without reloading; the drift map's colors depend on it.
  new MutationObserver(() => { if (state.tab === 'compare' && state.target) renderDiffOverview(); }).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  buildShell(); send('ready');
})();
