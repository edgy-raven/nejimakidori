const $ = (selector) => document.querySelector(selector);
const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const shape = (value) => Array.isArray(value) ? (value.length ? value.join(' × ') : 'scalar') : String(value ?? '—');
const text = (value) => Array.isArray(value) ? value.join(' · ') : (typeof value === 'object' && value !== null ? Object.entries(value).map(([key, entry]) => `${key}: ${text(entry)}`).join(' · ') : String(value ?? ''));
const sections = {
  architecture: { title: 'Computational graph', description: 'Shared encoder, policy factorization, opponent marginals and action-conditioned settlement distributions.', search: 'Search subgraphs, modules or operators…' },
  features: { title: 'Input tensor contract', description: 'Native dtypes, per-position shapes, encodings and consumer references. Seat axes are actor-relative.', search: 'Search tensor names, encodings or consumers…' },
  outputs: { title: 'Inference tensor outputs', description: 'Policy logits, marginal beliefs, conditional completion heads and settlement diagnostics.', search: 'Search output tensors or signatures…' },
  training: { title: 'Optimization schedule', description: 'Alternating payment-likelihood critic fitting and MMD actor updates over match and root-rollout records.', search: 'Search objectives, update phases or data sources…' },
  math: { title: 'Objectives and constraints', description: 'Implemented likelihoods, policy objectives, deterministic projections and continuation equations.', search: 'Search equations, variables or objectives…' },
  gradients: { title: 'Gradient paths', description: 'Phase-specific differentiation paths, detached tensors and optimizer variable ownership.', search: 'Search phases, parameter groups or detach boundaries…' },
};
let catalog;
let section = 'architecture';
let selected = '';
let query = '';
let group = '';
let dtype = '';
let gradientPhase = 'all';
let selectedOperation = '';
let architectureZoom = 1;
const architecturePan = { left: 0, top: 0 };
const mapPadding = 600;
const toolbar = $('.toolbar');
const mapPointers = new Map();
let mapInitialized = false;
let mapIndexOpen = false;
let mapJump = '';
let mapDrag = null;
let mapPinch = null;
let mapClickBlocked = 0;
let mapPhase = 'critic';

function currentOverlay() {
  return catalog.architecture.gradient_overlays.find((overlay) => overlay.id === mapPhase);
}
function items() {
  if (section === 'outputs') return [...catalog.outputs, ...catalog.auxiliaries.map((item) => ({ ...item, auxiliary: true }))];
  if (section === 'math') return catalog.mathematics;
  if (section === 'gradients') return catalog.architecture.gradient_items;
  if (section === 'architecture') return [...catalog.architecture.nodes, ...catalog.architecture.blocks.map((block) => ({ ...block, block: true })), ...(currentOverlay() ? [{ ...currentOverlay().objective, objective: true, kind: 'objective' }] : []), ...catalog.architecture.gradient_items];
  return catalog[section];
}
function itemId(item) { return item.id ?? item.name; }
function title(item) { return item.title ?? item.name ?? item.id; }
function groupTitle(id) { return catalog.groups.find((entry) => entry.id === id)?.title ?? id; }
function matches(item) {
  const references = item.block || item.objective ? item.equations.map((reference) => catalog.mathematics.find((topic) => topic.id === reference.topic)) : [];
  const statuses = item.block ? catalog.architecture.gradient_overlays.map((overlay) => overlay.statuses.find((entry) => entry.block === item.id)) : [];
  return (!query || JSON.stringify([item, references, statuses]).toLowerCase().includes(query.toLowerCase())) &&
    (section !== 'gradients' || gradientPhase === 'all' || item.phase === gradientPhase) &&
    (section !== 'features' || ((!group || item.group === group) && (!dtype || item.dtype === dtype)));
}
function go(nextSection, id = '') {
  location.hash = `${nextSection}${id ? '/' + encodeURIComponent(id) : ''}${nextSection === 'architecture' ? '?phase=' + encodeURIComponent(mapPhase) : ''}`;
}
function route() {
  const [path, parameters] = location.hash.slice(1).split('?');
  const [nextSection, ...parts] = path.split('/');
  const phase = new URLSearchParams(parameters).get('phase');
  if (nextSection === 'architecture') mapPhase = ['forward', 'actor', 'critic'].includes(phase) ? phase : 'critic';
  const previous = section;
  section = Object.hasOwn(sections, nextSection) ? nextSection : 'architecture';
  try {
    selected = decodeURIComponent(parts[0] ?? '');
    if (section === 'architecture') mapPhase = catalog.architecture.gradient_items.find((item) => item.id === selected)?.phase ?? mapPhase;
    selectedOperation = section === 'architecture' ? decodeURIComponent(parts.slice(1).join('/')) : '';
  } catch { selected = ''; selectedOperation = ''; }
  if (previous !== section) {
    query = ''; group = ''; dtype = '';
    $('#search').value = ''; $('#group').value = ''; $('#dtype').value = '';
  }
  render();
  if (parts.length && window.innerWidth <= 1000 && section === 'math') {
    requestAnimationFrame(() => $('.mathematical-topic')?.scrollIntoView({ block: 'start' }));
  } else if (parts.length && window.innerWidth <= 1000 && section !== 'architecture') {
    requestAnimationFrame(() => $('#inspector').scrollIntoView({ block: 'start' }));
  }
  if (selectedOperation) requestAnimationFrame(() => document.querySelector(`[data-operation-id="${CSS.escape(selectedOperation)}"]`)?.scrollIntoView({ block: 'center' }));
}
function sourceButton(source) {
  if (!source || !source.path) return '';
  return `<button class="source-link" type="button" data-source="${escape(source.path)}" data-line="${Number(source.line) || 1}">${escape(source.path)}${source.line ? ':' + source.line : ''}${source.role ? `<span class="source-role">${escape(source.role)}</span>` : ''}</button>`;
}
function sourceList(sources) {
  return (Array.isArray(sources) ? sources : [sources]).filter(Boolean).map(sourceButton).join('');
}
function paragraphs(value) {
  return text(value).split(/\n\n+/).filter(Boolean).map((part) => `<p>${escape(part)}</p>`).join('');
}
function detailList(value) {
  if (!value) return '';
  return Array.isArray(value) ? `<ul>${value.map((part) => `<li>${escape(text(part))}</li>`).join('')}</ul>` : paragraphs(value);
}
function inspector(item) {
  if (!item) {
    $('#inspector').innerHTML = '<div class="inspect-head"><p class="eyebrow">DETAIL INSPECTOR</p><h2>Select an item</h2></div><div class="inspect-body"><p>Select a graph node or tensor to inspect its contract, dependencies and source definition.</p></div>';
    return;
  }
  let body = '';
  if (item.gradient) {
    body += `<span class="edge-kind ${escape(item.kind)}">${escape(item.phase)} · ${item.kind === 'stop' ? 'Stopped path' : 'Reverse-mode path'}</span>${paragraphs(item.description)}<div class="math-display equation-render">${item.mathml}</div>${paragraphs(item.kind === 'stop' ? 'This local stopped path contributes zero. Other paths may still update the same variable.' : 'For this local forward dependency y = f(x), the upstream adjoint of y is multiplied by the transposed Jacobian. This is one path contribution, not a measured gradient magnitude.')}<button type="button" class="related-link" data-open-gradient-map="${escape(item.id)}">Show path in map</button><h3>Endpoints</h3>${[item.from, item.to].map((id) => `<button type="button" class="related-link" data-gradient-endpoint="${escape(id)}" data-phase="${escape(item.phase)}">${escape(catalog.architecture.blocks.find((block) => block.id === id)?.title ?? catalog.architecture.gradient_overlays.find((overlay) => overlay.objective.id === id).objective.title)}</button>`).join('')}`;
  } else if (section === 'features') {
    body += `<dl class="detail-facts"><div><dt>Shape</dt><dd>${escape(shape(item.shape))}</dd></div><div><dt>Data type</dt><dd>${escape(item.dtype)}</dd></div><div><dt>Elements</dt><dd>${Number(item.elements).toLocaleString()}</dd></div><div><dt>Bytes / position</dt><dd>${Number(item.bytes).toLocaleString()}</dd></div></dl>`;
    body += paragraphs(item.description);
    if (item.encoding) body += `<h3>Encoding</h3>${paragraphs(item.encoding)}`;
    body += `<h3>Native layout</h3><p>Byte offset <code>${item.offset.toLocaleString()}</code> in the native per-position feature record.</p>`;
    body += `<h3>Group</h3><p>${escape(groupTitle(item.group))}</p>`;
    if (item.producers?.length) body += `<h3>Producer references</h3>${sourceList(item.producers)}`;
    body += `<h3>Consumer references</h3>${item.consumers?.length ? sourceList(item.consumers) : '<p>No current learned consumer is identified by the source catalog.</p>'}`;
  } else if (section === 'architecture' && item.block) {
    const parent = catalog.architecture.nodes.find((node) => node.id === item.parent);
    body += `<h3>Parent component</h3><button class="related-link" data-select="${escape(item.parent)}">${escape(parent.title)}</button>`;
    body += paragraphs(item.summary);
    if (currentOverlay()) {
      const status = currentOverlay().statuses.find((entry) => entry.block === item.id);
      body += `<h3>${escape(currentOverlay().title)}</h3><span class="gradient-status ${escape(status.status)}">${escape(status.status)}</span>${paragraphs(status.description)}`;
    }
    body += equationDescriptions(item.equations);
    if (currentOverlay()) {
      const edges = [...new Map(currentOverlay().edges.filter((edge) => edge.from === item.id || edge.to === item.id).map((edge) => [edge.kind + ':' + edge.label, edge])).values()];
      if (edges.length) body += `<details class="reference-item"><summary>Autodiff paths</summary><ul class="edge-details">${edges.map((edge) => `<li><span class="edge-kind ${escape(edge.kind)}">${escape(edge.kind === 'stop' ? 'stop gradient' : 'reverse mode')}</span>${escape(edge.label)}</li>`).join('')}</ul></details>`;
    }
    if (item.dimensions.length) body += `<h3>Tensor dimensions</h3>${detailList(item.dimensions)}`;
    if (item.sharing) body += `<h3>Parameter sharing</h3>${paragraphs(item.sharing)}`;
    if (item.residual) body += `<h3>Residual path</h3>${paragraphs(item.residual)}`;
    for (const [label, endpoint, other] of [['Input dependencies', 'to', 'from'], ['Downstream consumers', 'from', 'to']]) {
      const edges = catalog.architecture.block_edges.filter((edge) => edge[endpoint] === item.id);
      if (edges.length) body += `<h3>${label}</h3><div class="related-links">${edges.map((edge) => {
        const block = catalog.architecture.blocks.find((entry) => entry.id === edge[other]);
        return `<button class="related-link" data-select="${escape(block.id)}" title="${escape(edge.label)}">${escape(block.title)}</button>`;
      }).join('')}</div>`;
    }
    body += `<h3>Operator sequence</h3><ol class="block-operations">${item.operations.map((operation, index) => `<li><details class="block-operation" data-operation-id="${escape(operation.id)}" ${operation.id === selectedOperation ? 'open' : ''}><summary><span class="operator-index">${String(index + 1).padStart(2, '0')}</span>${escape(operation.title)}</summary>${paragraphs(operation.description)}${operation.dimensions?.length ? `<h4>Tensor dimensions</h4>${detailList(operation.dimensions)}` : ''}${operation.sharing ? `<h4>Parameter sharing</h4>${paragraphs(operation.sharing)}` : ''}${operation.residual ? `<h4>Residual path</h4>${paragraphs(operation.residual)}` : ''}${sourceList(operation.source)}<a class="operation-link" href="#architecture/${encodeURIComponent(item.id)}/${encodeURIComponent(operation.id)}?phase=${encodeURIComponent(mapPhase)}">Operator permalink</a></details></li>`).join('')}</ol>`;
  } else if (section === 'architecture' && item.objective) {
    body += paragraphs(item.description);
    body += equationDescriptions(item.equations);
    body += `<h3>Gradient scope</h3>${detailList(currentOverlay().notes)}`;
  } else if (section === 'architecture') {
    body += paragraphs(item.summary);
    if (item.dimensions?.length) body += `<h3>Dimensions</h3>${detailList(item.dimensions)}`;
    body += detailList(item.details);
    const blocks = catalog.architecture.blocks.filter((block) => block.parent === item.id);
    if (blocks.length) body += `<h3>Individual blocks</h3><div class="related-links">${blocks.map((block) => `<button class="related-link block-link" data-select="${escape(block.id)}">${escape(block.title)}</button>`).join('')}</div>`;
    for (const [label, direction] of [['Input dependencies', 'to'], ['Downstream consumers', 'from']]) {
      const edges = catalog.architecture.edges.filter((edge) => edge[direction] === item.id);
      if (edges.length) {
        body += `<h3>${label}</h3><div class="related-links">${edges.map((edge) => {
          const id = direction === 'to' ? edge.from : edge.to;
          const target = catalog.architecture.nodes.find((node) => node.id === id);
          return `<button class="related-link" data-select="${escape(id)}" title="${escape(edge.label)}">${escape(target ? title(target) : id)}</button>`;
        }).join('')}</div>`;
      }
    }
    const consumerPath = item.source?.path;
    const implementation = catalog.architecture.modules.find((module) => module.path === consumerPath);
    if (implementation) body += `<h3>Implementation</h3><button class="related-link" data-module="${escape(implementation.path)}">Browse ${implementation.symbols.length} classes and functions</button>`;
    const count = catalog.features.filter((feature) => feature.consumers?.some((consumer) => consumer.path === consumerPath)).length;
    if (count) body += `<h3>Input references</h3><button class="related-link" data-feature-source="${escape(consumerPath)}">${count} tensors referenced by this module</button>`;
  } else if (section === 'math') {
    body += '<h3>Symbols</h3><dl class="symbol-definitions">';
    body += item.symbols.map((symbol) => `<div><dt>${symbol.mathml}</dt><dd>${escape(symbol.meaning)}</dd></div>`).join('');
    body += '</dl><h3>Implementation correspondence</h3><p>Rendered expressions are indexed to exact source definitions in this snapshot.</p>';
  } else {
    if (item.shape !== undefined) body += `<dl class="detail-facts"><div><dt>Shape</dt><dd>${escape(shape(item.shape))}</dd></div>${item.dtype ? `<div><dt>Data type</dt><dd>${escape(item.dtype)}</dd></div>` : ''}</dl>`;
    if (item.row_axis) body += `<h3>Row axis</h3><p>${escape(item.row_axis)}</p>`;
    if (item.signatures) body += `<h3>Export signatures</h3><p>${item.signatures.length ? escape(item.signatures.join(' · ')) : 'Internal tensor · not exported'}</p>`;
    body += paragraphs(item.description);
    if (item.formula) body += `<div class="formula">${escape(text(item.formula))}</div>`;
    if (item.details) body += detailList(item.details);
  }
  if (section === 'training' && ['actor', 'critic'].includes(item.id)) body += `<h3>Autodiff map</h3><button type="button" class="related-link" data-open-map-phase="${escape(item.id)}">Inspect ${escape(item.id)} gradients in the map</button>`;
  if (item.source) body += `<h3>Source definition</h3>${sourceList(item.source)}`;
  const inspectorTitle = title(item);
  $('#inspector').innerHTML = `<div class="inspect-head">${section === 'architecture' ? '<button class="map-close" type="button" data-map-close aria-label="Close block details">×</button>' : ''}<button class="back-results" type="button" data-results>Back to results</button><p class="eyebrow">${escape(section === 'features' ? groupTitle(item.group) : section === 'architecture' ? item.kind : sections[section].title)}</p>${item.status ? `<span class="badge ${escape(item.status)}">${escape(item.status)}</span>` : ''}<h2>${escape(inspectorTitle)}</h2>${item.name && item.name !== title(item) ? `<span class="inspect-name">${escape(item.name)}</span>` : ''}</div><div class="inspect-body">${body}</div>`;
}
function diagram() {
  const { blocks, block_edges: dependencies } = catalog.architecture;
  const grid = {
    "native-input-contract": [0, 3],
    "hand-tokenization": [1, 0],
    "hand-suit-0": [2, 0],
    "hand-suit-1": [3, 0],
    "hand-suit-2": [4, 0],
    "hand-honors": [2, 1],
    "hand-bank-output": [4, 1],
    "river-tokenization": [1, 3],
    "river-conv-0": [1, 4],
    "river-conv-1": [2, 4],
    "river-conv-2": [3, 4],
    "river-summaries": [4, 4],
    "board-input-projection": [5, 0],
    "board-block-0": [6, 0],
    "board-block-1": [7, 0],
    "board-block-2": [7, 1],
    "board-block-3": [6, 1],
    "board-output": [5, 1],
    "candidate-projection": [6, 3],
    "belief-readouts": [5, 5],
    "tile-tokenization": [4, 5],
    "tile-attention-0": [3, 5],
    "tile-attention-1": [3, 6],
    "tile-correction": [4, 6],
    "belief-constraints": [5, 6],
    "actor-candidate-aux": [7, 3],
    "actor-completion": [7, 4],
    "actor-final-score": [8, 3],
    "actor-win": [8, 2],
    "player-descriptors": [5, 8],
    "critic-common-projection": [6, 7],
    "critic-player-projection": [6, 8],
    "critic-event": [7, 7],
    "critic-claim-state": [7, 9],
    "critic-bonus-head": [8, 9],
    "critic-score-head": [8, 10],
    "critic-deposits": [8, 7],
    "settlement-decoder": [9, 8],
    "continuation-payoffs": [9, 9],
  };
  const overlay = currentOverlay();
  const positions = new Map(Object.entries(grid).map(([id, [column, row]]) => [id, [40 + column * (overlay ? 310 : 250), 80 + row * (overlay ? 180 : 130)]]));
  const width = overlay ? 3160 : 2560;
  const height = overlay ? 2080 : 1530;
  if (overlay) positions.set(overlay.objective.id, overlay.id === 'actor' ? [2890, 620] : [2890, 1930]);
  const gradientPositions = new Map();
  const occupied = [...positions.values()].map(([x, y]) => ({x: x - 8, y: y - 8, width: 232, height: 92}));
  for (const edge of overlay?.edges ?? []) {
    const start = positions.get(edge.from), end = positions.get(edge.to);
    const midpoint = {x: (start[0] + end[0]) / 2 + 108, y: (start[1] + end[1]) / 2 + 38};
    const candidates = [];
    for (let y = 20; y < height - 64; y += 60) {
      for (let x = 20; x < width - 180; x += 170) {
        if (occupied.every((box) => x + 168 < box.x || x > box.x + box.width || y + 56 < box.y || y > box.y + box.height)) candidates.push({x, y, distance: (x + 80 - midpoint.x) ** 2 + (y + 24 - midpoint.y) ** 2});
      }
    }
    candidates.sort((a, b) => a.distance - b.distance);
    const spot = candidates[0];
    gradientPositions.set(edge.id, [spot.x, spot.y]);
    occupied.push({x: spot.x - 4, y: spot.y - 4, width: 168, height: 56});
  }
  const displayBlocks = overlay ? [...blocks, { ...overlay.objective, kind: 'objective', summary: overlay.objective.description }] : blocks;
  const statuses = new Map(overlay ? overlay.statuses.map((entry) => [entry.block, entry]) : []);
  const mapEdges = [...dependencies.map((edge) => ({ ...edge, isGradient: false })), ...(overlay ? overlay.edges.map((edge) => ({ ...edge, isGradient: true })) : [])];
  const explicit = location.hash.startsWith('#architecture/');
  const selectedPath = overlay?.edges.find((edge) => edge.id === selected);
  const focus = new Set(selectedPath ? [selectedPath.from, selectedPath.to] : explicit ? displayBlocks.filter((block) => block.id === selected || block.parent === selected).map((block) => block.id) : []);
  const connected = new Set(focus);
  mapEdges.forEach((edge) => {
    if (focus.has(edge.from) || focus.has(edge.to)) { connected.add(edge.from); connected.add(edge.to); }
  });
  const regions = [
    ['Native input contract', ['inputs']],
    ['Shared hand bank', ['hand-bank']],
    ['River convolutions', ['rivers']],
    ['Residual board trunk', ['board']],
    ['Candidate and policy branches', ['candidates', 'actor']],
    ['Belief and tile-attention refinement', ['beliefs', 'tile-attention']],
    ['Player-conditioned critic', ['player-context', 'critic']],
    ['Settlement and continuation', ['settlement', 'continuation']],
  ];
  const lanes = regions.map(([label, parents]) => {
    const children = blocks.filter((block) => parents.includes(block.parent));
    const points = children.map((block) => positions.get(block.id));
    const left = Math.min(...points.map(([x]) => x)) - 18;
    const top = Math.min(...points.map(([, y]) => y)) - 48;
    const right = Math.max(...points.map(([x]) => x)) + 234;
    const bottom = Math.max(...points.map(([, y]) => y)) + 94;
    return `<g class="block-lane map-region"><rect x="${left}" y="${top}" width="${right - left}" height="${bottom - top}" rx="14"/><text x="${left + 18}" y="${top + 27}" class="block-lane-title">${escape(label)}</text></g>`;
  }).join('');
  const edges = mapEdges.sort((a, b) => Number(a.id === selected) - Number(b.id === selected)).map((edge) => {
    let [x1, y1] = positions.get(edge.from);
    let [x2, y2] = positions.get(edge.to);
    let path;
    if (x1 === x2) {
      if (y2 - y1 === 130) {
        x1 += 108; x2 += 108; y1 += 76;
        path = `M${x1},${y1} L${x2},${y2}`;
      } else {
        x1 += 216; x2 += 216; y1 += 38; y2 += 38;
        path = `M${x1},${y1} C${x1 + 22},${y1} ${x2 + 22},${y2} ${x2},${y2}`;
      }
    } else {
      const right = x2 > x1;
      x1 += right ? 216 : 0; x2 += right ? 0 : 216;
      y1 += edge.isGradient ? 48 : 38; y2 += edge.isGradient ? 48 : 38;
      const bend = (x1 + x2) / 2;
      path = `M${x1},${y1} C${bend},${y1} ${bend},${y2} ${x2},${y2}`;
    }
    if (edge.isGradient) {
      const [x, y] = gradientPositions.get(edge.id);
      path = `M${x1},${y1} L${x + 80},${y + 24} L${x2},${y2}`;
    }
    const active = focus.has(edge.from) || focus.has(edge.to);
    const marker = edge.isGradient ? edge.kind : edge.kind === 'residual' ? 'residual' : active ? 'active' : 'normal';
    const visible = `<path class="graph-edge ${edge.isGradient ? 'map-gradient-edge' : 'block-edge'} ${escape(edge.kind)} ${edge.id === selected ? 'selected-gradient' : ''} ${active ? 'connected' : focus.size ? 'dimmed' : ''}" d="${path}" marker-end="url(#block-arrow-${marker})" data-from="${escape(edge.from)}" data-to="${escape(edge.to)}"><title>${escape(edge.label)}</title></path>`;
    return edge.isGradient ? `<g class="gradient-path-item" role="button" tabindex="0" data-gradient-path="${escape(edge.id)}" aria-label="${escape(overlay.id + ' gradient: ' + edge.label)}" aria-pressed="${edge.id === selected}">${visible}<path class="gradient-edge-hit" d="${path}"/></g>` : visible;
  }).join('');
  const gradientMarkup = (overlay?.edges ?? []).map((edge) => {
    const [x, y] = gradientPositions.get(edge.id);
    const from = displayBlocks.find((block) => block.id === edge.from).title;
    const to = displayBlocks.find((block) => block.id === edge.to).title;
    return `<g class="map-gradient-node ${escape(edge.kind)} ${edge.id === selected ? 'selected' : ''}" role="button" tabindex="0" data-gradient-path="${escape(edge.id)}" data-map-x="${x}" data-map-y="${y}" aria-pressed="${edge.id === selected}" aria-label="${escape((edge.kind === 'stop' ? 'Stopped gradient' : 'Gradient VJP') + ': ' + from + ' to ' + to)}" transform="translate(${x} ${y})"><rect width="160" height="48" rx="8"/><text x="9" y="15" class="gradient-item-kind">${edge.kind === 'stop' ? '⊥ STOP GRADIENT' : '∇ VJP'}</text><text x="9" y="33" class="gradient-item-target">→ ${escape(to.length > 21 ? to.slice(0, 20) + '…' : to)}</text><title>${escape(from + ' → ' + to + ': ' + edge.label)}</title></g>`;
  }).join('');
  const nodeMarkup = displayBlocks.map((block) => {
    const [x, y] = positions.get(block.id);
    const status = overlay ? (block.id === overlay.objective.id ? { status: 'objective', description: overlay.objective.description } : statuses.get(block.id)) : null;
    let first = '', second = '';
    for (const word of block.title.split(' ')) {
      if (!second && (first + ' ' + word).trim().length <= 25) first = (first + ' ' + word).trim();
      else second = (second + ' ' + word).trim();
    }
    return `<g class="graph-node block-graph-node ${status ? 'phase-status-' + escape(status.status) : ''} ${overlay && block.id === overlay.objective.id ? 'map-objective' : ''} ${focus.has(block.id) ? 'selected' : ''} ${focus.size && !connected.has(block.id) ? 'dimmed' : ''}" transform="translate(${x} ${y})" role="button" tabindex="0" aria-label="${escape(block.title + (status ? ', ' + status.status : ''))}" data-gradient-status="${status ? escape(status.status) : ''}" aria-pressed="${focus.has(block.id)}" data-select="${escape(block.id)}" data-map-x="${x}" data-map-y="${y}"><rect width="216" height="76" rx="6"/><text x="12" y="19" class="graph-kind">${escape((status ? status.status : block.kind).toUpperCase().slice(0, 26))}</text><text x="12" y="${second ? 40 : 48}" class="graph-title">${escape(first)}</text>${second ? `<text x="12" y="59" class="graph-title">${escape(second.slice(0, 27))}</text>` : ''}<title>${escape(block.title + ': ' + block.summary + (status ? ' ' + status.description : ''))}</title></g>`;
  }).join('');
  return `<div class="diagram-panel block-diagram map-panel ${overlay ? 'map-gradient-active' : ''}" data-map-mode="${escape(mapPhase)}"><div class="map-toolbar-slot"></div><div class="map-phase-controls" role="group" aria-label="Map layer">${[['forward', 'Forward'], ['critic', 'Critic gradients'], ['actor', 'Actor gradients']].map(([id, label]) => `<button type="button" data-map-phase="${id}" aria-pressed="${mapPhase === id}">${label}</button>`).join('')}</div>${overlay ? '<div class="map-layer-legend"><span><i class="gradient-line backward"></i>Gradient</span><span><i class="gradient-line stop"></i>Stopped</span><span>Partial = selected subheads</span></div>' : ''}<div class="diagram-tools" role="group" aria-label="Map zoom"><button type="button" data-diagram-zoom="in" aria-label="Zoom in">+</button><button type="button" data-diagram-zoom="out" aria-label="Zoom out">−</button><button type="button" data-diagram-zoom="fit">Fit</button><button type="button" data-diagram-zoom="actual">100%</button><output aria-label="Current zoom">${Math.round(architectureZoom * 100)}%</output></div><div class="map-caption"><button type="button" data-map-index>Source index</button><span>${blocks.length} blocks · ${overlay ? overlay.edges.length + ' gradient items' : dependencies.length + ' dependencies'}</span>${overlay ? `<button type="button" data-select="${escape(overlay.objective.id)}" data-map-result>Objective</button>` : ''}</div><div class="block-diagram-viewport" tabindex="0" aria-label="Computational graph map. Drag to pan, scroll to zoom; plus and minus change zoom."><div class="map-world"><svg class="block-diagram-svg" data-base-width="${width}" data-base-height="${height}" width="${width * architectureZoom}" height="${height * architectureZoom}" viewBox="0 0 ${width} ${height}" role="group" aria-label="Individual model blocks and tensor dependencies"><defs><marker id="block-arrow-normal" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#78949c"/></marker><marker id="block-arrow-active" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#007f78"/></marker><marker id="block-arrow-residual" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#b86e28"/></marker><marker id="block-arrow-backward" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#007f78"/></marker><marker id="block-arrow-stop" markerWidth="8" markerHeight="10" refX="6" refY="5" orient="auto"><path d="M6 0L6 10" stroke="#bb5a24" stroke-width="2.5"/></marker></defs>${lanes}${edges}${nodeMarkup}${gradientMarkup}</svg></div></div></div>`;
}

function vocabulary() {
  return `<div class="section-caption"><h2>Encoding reference</h2><span>Axis indexing and encoding conventions</span></div><div class="reference-list">${catalog.vocabulary.map((entry) => `<details class="reference-item"><summary>${escape(entry.title)}</summary>${paragraphs(entry.description)}</details>`).join('')}</div>`;
}
function blockInventory(blocks) {
  if (!blocks.length) return '';
  return `<div class="section-caption"><h2>Individual computational blocks</h2><span>${blocks.length} blocks</span></div><div class="block-inventory">${catalog.architecture.nodes.map((component) => {
    const children = blocks.filter((block) => block.parent === component.id);
    if (!children.length) return '';
    return `<section class="block-group" aria-label="${escape(component.title)} blocks"><header><button type="button" data-select="${escape(component.id)}">${escape(component.title)}</button><span>${children.length} blocks</span></header><div class="block-grid">${children.map((block) => `<button type="button" class="block-card ${block.id === selected ? 'selected' : ''}" data-select="${escape(block.id)}" aria-pressed="${block.id === selected}"><span class="block-kind">${escape(block.kind)}</span><strong>${escape(block.title)}</strong><span class="block-summary">${escape(block.summary)}</span><span class="block-meta">${block.operations.length} operators${block.residual ? ' · residual path' : ''}</span></button>`).join('')}</div></section>`;
  }).join('')}</div>`;
}
function modules() {
  const filtered = catalog.architecture.modules.filter((module) => !query || JSON.stringify(module).toLowerCase().includes(query.toLowerCase()));
  return `<div class="section-caption"><h2>Source modules</h2><span>${filtered.length} modules · classes, functions and layers</span></div><div class="reference-list">${filtered.map((module) => `<details class="module reference-item" data-module-path="${escape(module.path)}" ${query ? 'open' : ''}><summary><span>${escape(module.path)}</span><span class="module-count">${module.symbols.length} symbols</span></summary>${paragraphs(module.description)}${sourceButton({path: module.path, line: 1})}${module.symbols.map((symbol) => `<details class="symbol"><summary><span class="symbol-kind">${escape(symbol.kind)}</span><code>${escape(symbol.name)}</code></summary>${paragraphs(symbol.description)}${sourceButton(symbol.source)}${symbol.layers.length ? `<h3>Layer declarations</h3>${symbol.layers.map((layer) => `<pre class="layer-code">${escape(layer)}</pre>`).join('')}` : ''}</details>`).join('')}</details>`).join('')}</div>`;
}
function predictionRows(entries) {
  return `<div class="list-panel">${entries.map((entry) => `<button class="entry ${itemId(entry) === selected ? 'selected' : ''}" data-select="${escape(itemId(entry))}" aria-pressed="${itemId(entry) === selected}"><span class="entry-head"><span class="entry-title">${escape(title(entry))}</span>${entry.gradient ? `<span class="edge-kind ${escape(entry.kind)}">${escape(entry.phase)} · ${entry.kind === 'stop' ? 'stopped' : 'backward'}</span>` : ''}${entry.shape !== undefined ? `<span class="entry-shape">${escape(shape(entry.shape))}</span>` : ''}</span><p class="entry-description">${escape(entry.description)}</p></button>`).join('')}</div>`;
}
function topicPicker(entries) {
  return `<nav class="topic-picker" aria-label="${section === 'math' ? 'Mathematical topics' : 'Optimization phases'}">${entries.map((item) => `<button type="button" class="topic-button ${item.id === selected ? 'selected' : ''}" data-select="${escape(item.id)}" aria-pressed="${item.id === selected}">${escape(item.title)}</button>`).join('')}</nav>`;
}
function equationDescriptions(references) {
  return references.map((reference) => {
    const topic = catalog.mathematics.find((entry) => entry.id === reference.topic);
    return `<section class="description-equations"><h3>${escape(topic.title)}</h3>${paragraphs(topic.description)}${reference.equations.map((index) => `<div class="description-equation"><h4>${escape(topic.equations[index].label)}</h4><div class="math-display" tabindex="0" aria-label="${escape(topic.equations[index].label)}">${topic.equations[index].mathml}</div></div>`).join('')}<details class="equation-symbols"><summary>Notation</summary><dl class="symbol-definitions">${topic.symbols.map((symbol) => `<div><dt>${symbol.mathml}</dt><dd>${escape(symbol.meaning)}</dd></div>`).join('')}</dl></details>${sourceList(topic.source)}</section>`;
  }).join('');
}
function mathematicalTopic(item) {
  return `<article class="mathematical-topic"><header class="topic-header"><button class="back-topics" type="button" data-topics>Back to topics</button><p class="eyebrow">SOURCE-ALIGNED EQUATIONS</p><h2>${escape(item.title)}</h2>${paragraphs(item.description)}</header>${item.equations.map((equation) => `<section class="equation"><h3>${escape(equation.label)}</h3><div class="math-display" tabindex="0" aria-label="${escape(equation.label)}">${equation.mathml}</div></section>`).join('')}</article>`;
}
function render() {
  const viewport = $('.block-diagram-viewport');
  if (viewport) { architecturePan.left = viewport.scrollLeft; architecturePan.top = viewport.scrollTop; }
  if (toolbar.parentElement?.classList.contains('map-toolbar-slot')) $('#main').insertBefore(toolbar, $('#loading'));
  $('#main').classList.toggle('map-page', section === 'architecture');
  $('#workspace').classList.toggle('architecture-workspace', section === 'architecture');
  $('#workspace').classList.remove('gradient-workspace');
  const filtered = items().filter(matches);
  const item = items().find((entry) => itemId(entry) === selected) ?? filtered[0];
  if (item) selected = itemId(item);
  document.title = `${sections[section].title} · Nejimakidori`;
  $('#page-title').textContent = sections[section].title;
  $('#page-description').textContent = sections[section].description;
  $('#search').placeholder = sections[section].search;
  $('#group-filter').hidden = section !== 'features';
  $('#dtype-filter').hidden = section !== 'features';
  document.querySelectorAll('[data-section]').forEach((link) => {
    if (link.dataset.section === section) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  });
  $('#result-count').textContent = section === 'architecture' ? `${filtered.filter((item) => !item.block && !item.objective && !item.gradient).length} nodes · ${filtered.filter((item) => item.block).length} blocks` : `${filtered.length} of ${items().length}`;
  $('#overview').innerHTML = section === 'features' ? `<div class="overview-strip"><div><strong>${catalog.features.length}</strong><span>input tensors</span></div><div><strong>${catalog.groups.length}</strong><span>groups</span></div><div><strong>${catalog.features.reduce((sum, feature) => sum + Number(feature.elements), 0).toLocaleString()}</strong><span>scalars / position</span></div></div>` : section === 'architecture' ? diagram() : '';
  if (!filtered.length) {
    $('#results').innerHTML = `<div class="empty"><h2>${section === 'architecture' ? 'No matching components' : 'No matching items'}</h2><p>Try another name, concept or source file.</p><button type="button" data-reset>Clear filters</button></div>`;
  } else if (section === 'features') {
    $('#results').innerHTML = `<div class="table-scroll"><table class="feature-table"><thead><tr><th scope="col">Feature</th><th scope="col">Shape</th><th scope="col">Type</th><th scope="col">Group</th></tr></thead><tbody>${filtered.map((feature) => `<tr class="${itemId(feature) === selected ? 'selected' : ''}"><td><button class="feature-button" data-select="${escape(itemId(feature))}" aria-pressed="${itemId(feature) === selected}">${escape(feature.name)}<span class="feature-title">${escape(feature.title)}</span></button></td><td class="shape">${escape(shape(feature.shape))}</td><td class="dtype">${escape(feature.dtype)}</td><td class="group-label">${escape(groupTitle(feature.group))}</td></tr>`).join('')}</tbody></table></div>`;
  } else if (section === 'architecture') {
    const nodes = filtered.filter((item) => !item.block && !item.objective && !item.gradient);
    const component = item.block ? item.parent : selected;
    $('#results').innerHTML = `${nodes.length ? `<details class="architecture-index reference-item"><summary>Component index · ${nodes.length} subgraphs</summary><div class="component-list">${nodes.map((node) => `<button class="component ${node.id === component ? 'selected' : ''}" data-select="${escape(node.id)}" aria-pressed="${node.id === component}"><span class="kind">${escape(node.kind)}</span><strong>${escape(title(node))}</strong><p>${escape(node.summary)}</p></button>`).join('')}</div></details>` : ''}<details class="architecture-index reference-item" ${query ? 'open' : ''}><summary>Block index · ${filtered.filter((entry) => entry.block).length} blocks</summary>${blockInventory(filtered.filter((entry) => entry.block))}</details>`;
  } else if (section === 'gradients') {
    $('#overview').innerHTML = `<div class="gradient-controls" role="group" aria-label="Gradient phase">${['all', 'actor', 'critic'].map((phase) => `<button type="button" data-gradient-phase="${phase}" aria-pressed="${gradientPhase === phase}">${phase === 'all' ? 'All paths' : phase + ' pass'}</button>`).join('')}</div>`;
    $('#results').innerHTML = predictionRows(filtered);
  } else if (section === 'math') {
    $('#overview').innerHTML = topicPicker(filtered);
    $('#results').innerHTML = mathematicalTopic(item);
  } else {
    $('#results').innerHTML = section === 'outputs' ? [
      ['Exported inference tensors', filtered.filter((entry) => !entry.auxiliary)],
      ['Internal prediction features', filtered.filter((entry) => entry.auxiliary && entry.id !== 'constraint_loss')],
      ['Training loss terms', filtered.filter((entry) => entry.id === 'constraint_loss')],
    ].filter(([, entries]) => entries.length).map(([label, entries]) => `<div class="section-caption"><h2>${label}</h2><span>${entries.length} tensors</span></div>${predictionRows(entries)}`).join('') : predictionRows(filtered);
  }
  if (section === 'architecture') {
    if (query) $('#results').innerHTML = `<div class="map-result-heading">${filtered.filter((entry) => entry.block || entry.gradient).length} items</div>${filtered.filter((entry) => entry.block || entry.gradient).map((block) => `<button type="button" class="map-search-result" data-select="${escape(block.id)}" data-map-result><strong>${escape(block.title)}</strong><span>${escape(block.summary ?? block.description)}</span></button>`).join('') || '<p class="map-no-results">No matching blocks.</p>'}`;
    else $('#results').insertAdjacentHTML('beforeend', modules());
    $('.map-toolbar-slot').append(toolbar);
    $('#results').insertAdjacentHTML('afterbegin', '<button class="map-close" type="button" data-map-close aria-label="Close index">×</button>');
    $('#results').hidden = !query && !mapIndexOpen;
  } else $('#results').hidden = false;
  if (section === 'features') $('#results').insertAdjacentHTML('beforeend', vocabulary());
  inspector(item);
  $('#inspector').hidden = section === 'architecture' && (!location.hash.startsWith('#architecture/') || Boolean(query) || mapIndexOpen);
  const restoredViewport = $('.block-diagram-viewport');
  if (restoredViewport) {
    if (mapInitialized) { restoredViewport.scrollLeft = architecturePan.left; restoredViewport.scrollTop = architecturePan.top; }
    else {
      restoredViewport.scrollLeft = mapPadding + 1280 * architectureZoom - restoredViewport.clientWidth / 2;
      restoredViewport.scrollTop = mapPadding + 765 * architectureZoom - restoredViewport.clientHeight / 2;
      mapInitialized = true;
      if (location.hash.startsWith('#architecture/')) centerMapBlock(selected);
    }
    if (mapJump) { centerMapBlock(mapJump); mapJump = ''; }
  }
}
function scaleMap(next, point, world) {
  const viewport = $('.block-diagram-viewport');
  const svg = $('.block-diagram-svg');
  const anchor = point ?? { x: viewport.clientWidth / 2, y: viewport.clientHeight / 2 };
  const position = world ?? { x: (viewport.scrollLeft + anchor.x - mapPadding) / architectureZoom, y: (viewport.scrollTop + anchor.y - mapPadding) / architectureZoom };
  architectureZoom = Math.max(.05, Math.min(2, next));
  svg.setAttribute('width', Number(svg.dataset.baseWidth) * architectureZoom);
  svg.setAttribute('height', Number(svg.dataset.baseHeight) * architectureZoom);
  viewport.scrollLeft = mapPadding + position.x * architectureZoom - anchor.x;
  viewport.scrollTop = mapPadding + position.y * architectureZoom - anchor.y;
  architecturePan.left = viewport.scrollLeft;
  architecturePan.top = viewport.scrollTop;
  $('.diagram-tools output').textContent = `${Math.round(architectureZoom * 100)}%`;
}
function fitMap() {
  const viewport = $('.block-diagram-viewport');
  const svg = $('.block-diagram-svg');
  scaleMap(Math.min(viewport.clientWidth / Number(svg.dataset.baseWidth), viewport.clientHeight / Number(svg.dataset.baseHeight)) * .94, null, { x: Number(svg.dataset.baseWidth) / 2, y: Number(svg.dataset.baseHeight) / 2 });
}
function centerMapBlock(id) {
  const gradient = catalog.architecture.gradient_items.find((item) => item.id === id);
  id = gradient?.to ?? id;
  const target = currentOverlay()?.objective.id === id ? id : catalog.architecture.blocks.find((entry) => entry.id === id || entry.parent === id).id;
  const node = document.querySelector(`.block-graph-node[data-select="${CSS.escape(target)}"]`);
  const bounds = $('.block-diagram-viewport').getBoundingClientRect();
  const anchor = { x: bounds.width / 2, y: bounds.height / 2 };
  let top = bounds.top;
  let bottom = bounds.bottom;
  if (!$('#inspector').hidden) {
    const drawer = $('#inspector').getBoundingClientRect();
    if (window.matchMedia('(max-width: 700px)').matches) bottom = drawer.top;
    else anchor.x = (drawer.right + bounds.right) / 2 - bounds.left;
  }
  let zoom = Math.max(.8, architectureZoom);
  const centerX = bounds.left + anchor.x;
  for (const control of document.querySelectorAll('.map-toolbar-slot, .map-phase-controls, .map-layer-legend')) {
    const controlBounds = control.getBoundingClientRect();
    if (controlBounds.right > centerX - 108 * zoom && controlBounds.left < centerX + 108 * zoom) top = Math.max(top, controlBounds.bottom);
  }
  anchor.y = (top + bottom) / 2 - bounds.top;
  if (gradient) {
    const origin = document.querySelector(`.block-graph-node[data-select="${CSS.escape(gradient.from)}"]`);
    const card = document.querySelector(`.map-gradient-node[data-gradient-path="${CSS.escape(gradient.id)}"]`);
    const points = [node, origin, card];
    const left = Math.min(...points.map((point) => Number(point.dataset.mapX)));
    const right = Math.max(...points.map((point) => Number(point.dataset.mapX))) + 216;
    const upper = Math.min(...points.map((point) => Number(point.dataset.mapY)));
    const lower = Math.max(...points.map((point) => Number(point.dataset.mapY))) + 76;
    zoom = Math.min(1, (2 * (bounds.width - anchor.x) - 40) / (right - left), (bottom - top - 32) / (lower - upper));
    scaleMap(zoom, anchor, { x: (left + right) / 2, y: (upper + lower) / 2 });
    return;
  }
  scaleMap(zoom, anchor, { x: Number(node.dataset.mapX) + 108, y: Number(node.dataset.mapY) + 38 });
}
function mapPoint(event, viewport) {
  const bounds = viewport.getBoundingClientRect();
  return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
}
document.addEventListener('wheel', (event) => {
  const viewport = event.target.closest('.block-diagram-viewport');
  if (!viewport) return;
  event.preventDefault();
  scaleMap(architectureZoom * Math.exp(-event.deltaY * .0015), mapPoint(event, viewport));
}, { passive: false });
document.addEventListener('pointerdown', (event) => {
  const viewport = event.target.closest('.block-diagram-viewport');
  if (!viewport || (event.pointerType === 'mouse' && event.button !== 0)) return;
  mapPointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  if (mapPointers.size === 1) mapDrag = { id: event.pointerId, x: event.clientX, y: event.clientY, left: viewport.scrollLeft, top: viewport.scrollTop, moved: false };
  if (mapPointers.size === 2) {
    const [a, b] = [...mapPointers.values()];
    const bounds = viewport.getBoundingClientRect();
    const center = { x: (a.x + b.x) / 2 - bounds.left, y: (a.y + b.y) / 2 - bounds.top };
    mapPinch = { distance: Math.hypot(a.x - b.x, a.y - b.y), zoom: architectureZoom, world: { x: (viewport.scrollLeft + center.x - mapPadding) / architectureZoom, y: (viewport.scrollTop + center.y - mapPadding) / architectureZoom } };
    viewport.setPointerCapture(event.pointerId);
    mapClickBlocked = performance.now() + 500;
  }
});
document.addEventListener('pointermove', (event) => {
  if (!mapPointers.has(event.pointerId)) return;
  const viewport = $('.block-diagram-viewport');
  mapPointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
  if (mapPointers.size === 2 && mapPinch) {
    const [a, b] = [...mapPointers.values()];
    const bounds = viewport.getBoundingClientRect();
    scaleMap(mapPinch.zoom * Math.hypot(a.x - b.x, a.y - b.y) / mapPinch.distance, { x: (a.x + b.x) / 2 - bounds.left, y: (a.y + b.y) / 2 - bounds.top }, mapPinch.world);
  } else if (mapDrag?.id === event.pointerId) {
    const dx = event.clientX - mapDrag.x, dy = event.clientY - mapDrag.y;
    if (Math.hypot(dx, dy) > 4) {
      mapDrag.moved = true;
      viewport.classList.add('dragging');
      viewport.setPointerCapture(event.pointerId);
      viewport.scrollLeft = mapDrag.left - dx;
      viewport.scrollTop = mapDrag.top - dy;
    }
  }
});
function finishMapPointer(event) {
  if (!mapPointers.has(event.pointerId)) return;
  const viewport = $('.block-diagram-viewport');
  if (mapDrag?.moved || mapPinch) mapClickBlocked = performance.now() + 300;
  mapPointers.delete(event.pointerId);
  mapPinch = null;
  mapDrag = null;
  viewport.classList.remove('dragging');
  if (mapPointers.size === 1) {
    const [id, point] = [...mapPointers.entries()][0];
    mapDrag = { id, x: point.x, y: point.y, left: viewport.scrollLeft, top: viewport.scrollTop, moved: false };
  }
}
document.addEventListener('pointerup', finishMapPointer);
document.addEventListener('pointercancel', finishMapPointer);

function openSource(path, line) {
  const source = catalog.source_files?.find((entry) => entry.path === path);
  $('#source-title').textContent = path;
  $('#source-digest').textContent = source ? `SHA-256 ${source.sha256}` : 'This file is referenced by the catalog.';
  $('#source-code').innerHTML = source ? source.text.split('\n').map((value, index) => `<span class="code-line ${index + 1 === line ? 'highlight' : ''}" id="source-line-${index + 1}">${escape(value) || ' '}</span>`).join('') : '<p class="source-unavailable">Source text is not included in this catalog snapshot.</p>';
  $('#source-dialog').showModal();
  if (source) $(`#source-line-${line}`)?.scrollIntoView({ block: 'center' });
}
$('#search').addEventListener('input', (event) => {
  query = event.target.value; selected = '';
  if (section === 'architecture') { mapIndexOpen = false; history.replaceState(null, '', '#architecture' + ('?phase=' + encodeURIComponent(mapPhase))); }
  render();
});
$('#group').addEventListener('change', (event) => { group = event.target.value; selected = ''; render(); });
$('#dtype').addEventListener('change', (event) => { dtype = event.target.value; selected = ''; render(); });
$('#close-source').addEventListener('click', () => $('#source-dialog').close());
document.addEventListener('click', (event) => {
  if (performance.now() < mapClickBlocked && event.target.closest('.block-diagram-viewport')) { event.preventDefault(); return; }
  if (event.target.closest('.skip-link')) {
    event.preventDefault();
    $('#main').focus();
    return;
  }
  const select = event.target.closest('[data-select]');
  const source = event.target.closest('[data-source]');
  const featureSource = event.target.closest('[data-feature-source]');
  const module = event.target.closest('[data-module]');
  const gradientPath = event.target.closest('[data-gradient-path]');
  const gradientLink = event.target.closest('[data-open-gradient-map]');
  const gradientEndpoint = event.target.closest('[data-gradient-endpoint]');
  const gradientFilter = event.target.closest('[data-gradient-phase]');
  const zoom = event.target.closest('[data-diagram-zoom]');
  const phase = event.target.closest('[data-map-phase]');
  const phaseLink = event.target.closest('[data-open-map-phase]');
  if (gradientPath || gradientLink) {
    const id = gradientPath?.dataset.gradientPath ?? gradientLink.dataset.openGradientMap;
    const item = catalog.architecture.gradient_items.find((entry) => entry.id === id);
    mapPhase = item.phase;
    if (gradientLink) mapJump = item.id;
    const previousHash = location.hash;
    go('architecture', id);
    if (location.hash === previousHash) render();
  } else if (gradientEndpoint) {
    mapPhase = gradientEndpoint.dataset.phase; mapJump = gradientEndpoint.dataset.gradientEndpoint;
    go('architecture', mapJump);
  } else if (gradientFilter) {
    gradientPhase = gradientFilter.dataset.gradientPhase; selected = ''; render();
  } else if (phase) {
    const wasObjective = catalog.architecture.gradient_overlays.some((overlay) => overlay.objective.id === selected);
    mapPhase = phase.dataset.mapPhase;
    if (catalog.architecture.gradient_items.some((item) => item.id === selected)) go('architecture');
    else if (wasObjective) go('architecture', currentOverlay()?.objective.id ?? '');
    else history.replaceState(null, '', location.hash.split('?')[0] + ('?phase=' + encodeURIComponent(mapPhase)));
    render();
  } else if (phaseLink) {
    mapPhase = phaseLink.dataset.openMapPhase;
    mapJump = currentOverlay().objective.id;
    go('architecture', currentOverlay().objective.id);
  } else if (event.target.closest('[data-map-close]')) {
    mapIndexOpen = false; query = ''; $('#search').value = '';
    go('architecture'); render();
  } else if (event.target.closest('[data-map-index]')) {
    mapIndexOpen = !mapIndexOpen;
    query = ''; $('#search').value = '';
    render();
  } else if (zoom) {
    if (zoom.dataset.diagramZoom === 'fit') fitMap();
    else if (zoom.dataset.diagramZoom === 'actual') scaleMap(1);
    else scaleMap(architectureZoom * (zoom.dataset.diagramZoom === 'in' ? 1.25 : .8));
  } else if (select) {
    const previousHash = location.hash;
    if (section === 'architecture') {
      mapPhase = catalog.architecture.gradient_items.find((item) => item.id === select.dataset.select)?.phase ?? mapPhase;
      if (!select.closest('.block-diagram-viewport')) mapJump = select.dataset.select;
      query = ''; $('#search').value = ''; mapIndexOpen = false;
    }
    go(section, select.dataset.select);
    if (section === 'architecture' && location.hash === previousHash) render();
    if (window.innerWidth <= 1000 && !['architecture', 'math', 'gradients'].includes(section)) requestAnimationFrame(() => $('#inspector').scrollIntoView({ block: 'start', behavior: 'smooth' }));
  } else if (event.target.closest('[data-topics]')) {
    go('math');
    $('#overview').scrollIntoView({ block: 'start' });
  } else if (event.target.closest('[data-results]')) {
    go(section);
    $('#results').scrollIntoView({ block: 'start' });
  } else if (module) {
    query = '';
    $('#search').value = '';
    render();
    const target = document.querySelector(`[data-module-path="${CSS.escape(module.dataset.module)}"]`);
    target.open = true;
    target.scrollIntoView({ block: 'start', behavior: 'smooth' });
  } else if (source) openSource(source.dataset.source, Number(source.dataset.line));
  else if (featureSource) {
    section = 'features'; query = featureSource.dataset.featureSource; group = ''; dtype = ''; selected = '';
    $('#search').value = query; $('#group').value = ''; $('#dtype').value = '';
    go('features'); render();
  } else if (event.target.closest('[data-reset]')) {
    query = ''; group = ''; dtype = ''; selected = '';
    $('#search').value = ''; $('#group').value = ''; $('#dtype').value = ''; render();
  }
});
document.addEventListener('keydown', (event) => {
  if (event.target.closest('.block-diagram-viewport') && ['+', '=', '-'].includes(event.key)) {
    event.preventDefault(); scaleMap(architectureZoom * (event.key === '-' ? .8 : 1.25));
  }
  if (event.target.matches('.graph-node') && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); go(section, event.target.dataset.select); }
  if (event.target.matches('.gradient-path-item, .map-gradient-node') && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); go('architecture', event.target.dataset.gradientPath); }
  if (event.key === '/' && !event.target.matches('input,textarea,select') && !$('#source-dialog').open) { event.preventDefault(); $('#search').focus(); }
  if (event.key === 'Escape' && event.target === $('#search')) { query = ''; selected = ''; $('#search').value = ''; render(); }
});
try {
  const response = await fetch('catalog.json');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  catalog = await response.json();
  $('#feature-count').textContent = catalog.features.length;
  $('#architecture-count').textContent = catalog.architecture.blocks.length;
  $('#output-count').textContent = catalog.outputs.length + catalog.auxiliaries.length;
  $('#training-count').textContent = catalog.training.length;
  $('#source-version').textContent = `Source ${catalog.meta.source_digest.slice(0, 12)}`;
  $('#source-version').title = catalog.meta.source_digest;
  $('#group').innerHTML += catalog.groups.map((entry) => `<option value="${escape(entry.id)}">${escape(entry.title)}</option>`).join('');
  $('#dtype').innerHTML += [...new Set(catalog.features.map((entry) => entry.dtype))].sort().map((entry) => `<option value="${escape(entry)}">${escape(entry)}</option>`).join('');
  for (const selector of ['#search', '#group', '#dtype']) $(selector).disabled = false;
  $('#loading').hidden = true; $('#workspace').hidden = false;
  window.addEventListener('hashchange', route);
  route();
} catch (error) {
  $('#loading').innerHTML = `<div class="empty"><h2>The catalog could not be loaded</h2><p>${escape(error.message)}</p><p>Reload this page to try again.</p></div>`;
}
