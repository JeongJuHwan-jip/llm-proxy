'use strict';

const FALLBACK_INTERVAL = 30;
let countdown = FALLBACK_INTERVAL;
let fallbackTimer = null;
let sseConnected = false;

// ── Helpers ─────────────────────────────────────────────────────────────────

function fmtTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour12: false });
}
function fmtLatency(ms) {
  if (ms == null) return '\u2014';
  return ms >= 1000 ? (ms / 1000).toFixed(2) + 's' : Math.round(ms) + 'ms';
}
function fmtRate(r) { return (r * 100).toFixed(1) + '%'; }

function escHtml(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function mk(tag, attrs, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'className') n.className = v;
    else if (k === 'textContent') n.textContent = v;
    else n.setAttribute(k, v);
  }
  for (const c of children) {
    if (typeof c === 'string') n.appendChild(document.createTextNode(c));
    else if (c) n.appendChild(c);
  }
  return n;
}

// ══════════════════════════════════════════════════════════════════════════════
//  TAB SWITCHING
// ══════════════════════════════════════════════════════════════════════════════

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });
}

// ══════════════════════════════════════════════════════════════════════════════
//  MONITOR TAB — endpoint cards + request log
// ══════════════════════════════════════════════════════════════════════════════

async function fetchStatus() {
  const r = await fetch('/api/status');
  if (!r.ok) throw new Error('status fetch failed');
  return r.json();
}

function renderEndpoints(statuses) {
  const c = document.getElementById('endpoints-container');
  if (!statuses.length) { c.innerHTML = '<p class="muted">No endpoints.</p>'; return; }
  c.innerHTML = '';
  for (const s of statuses) {
    const st = s.circuit_state;
    c.appendChild(mk('div', { className: `card ${st}` },
      mk('div', { className: 'card-name', textContent: s.name }),
      mk('div', { className: 'card-url', textContent: s.url }),
      mk('span', { className: `card-state state-${st}`, textContent: st.replace('_',' ') }),
      mk('div', { className: 'card-stats' },
        mk('div',{},'Requests'), mk('div',{},mk('span',{textContent:String(s.total_requests)})),
        mk('div',{},'Failures'), mk('div',{},mk('span',{textContent:String(s.total_failures)})),
        mk('div',{},'Timeouts'), mk('div',{},mk('span',{textContent:String(s.total_timeouts)})),
        mk('div',{},'Timeout rate'), mk('div',{},mk('span',{textContent:fmtRate(s.timeout_rate)})),
        mk('div',{},'Avg latency'), mk('div',{},mk('span',{textContent:fmtLatency(s.avg_latency_ms)})),
        mk('div',{},'CB failures'), mk('div',{},mk('span',{textContent:String(s.consecutive_failures)})),
      )
    ));
  }
}

async function fetchRequests() {
  const r = await fetch('/api/requests?limit=50');
  if (!r.ok) throw new Error('requests fetch failed');
  return r.json();
}

function renderRequests({ total, rows }) {
  document.getElementById('total-count').textContent = total.toLocaleString();
  const tbody = document.getElementById('requests-body');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="muted">No requests yet.</td></tr>';
    return;
  }
  tbody.innerHTML = '';
  for (const r of rows) {
    const aHtml = r.attempts.map(a => {
      const cls = a.success ? 'attempt-ok' : (a.is_timeout ? 'attempt-to' : 'attempt-err');
      const icon = a.success ? '\u2713' : (a.is_timeout ? '\u23F1' : '\u2717');
      return `<span class="${cls}" title="${a.endpoint} \u2014 ${Math.round(a.latency_ms)}ms${a.error ? ': '+a.error : ''}">${icon} ${a.endpoint}</span>`;
    }).join('<br>');
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${fmtTime(r.timestamp)}</td>
      <td>${escHtml(r.model)}</td>
      <td>${escHtml(r.selected_endpoint||'\u2014')}</td>
      <td class="status-${r.status}">${r.status}</td>
      <td>${fmtLatency(r.total_latency_ms)}</td>
      <td>${r.is_stream ? '\u26A1 stream' : 'sync'}</td>
      <td class="attempt-list">${aHtml}</td>`;
    tbody.appendChild(tr);
  }
}

async function refreshMonitor() {
  try {
    const [statuses, reqData] = await Promise.all([fetchStatus(), fetchRequests()]);
    renderEndpoints(statuses);
    renderRequests(reqData);
  } catch (e) { console.error('Refresh error:', e); }
}

// ══════════════════════════════════════════════════════════════════════════════
//  SSE
// ══════════════════════════════════════════════════════════════════════════════

function connectSSE() {
  const src = new EventSource('/api/events');
  src.onopen = () => {
    sseConnected = true; stopFallback(); updateConn();
  };
  src.onmessage = () => refreshMonitor();
  src.onerror = () => {
    sseConnected = false; src.close(); updateConn(); startFallback();
    setTimeout(connectSSE, 3000);
  };
}

function startFallback() {
  if (fallbackTimer) return;
  countdown = FALLBACK_INTERVAL;
  fallbackTimer = setInterval(() => {
    if (--countdown <= 0) { refreshMonitor(); countdown = FALLBACK_INTERVAL; }
  }, 1000);
}
function stopFallback() { if (fallbackTimer) { clearInterval(fallbackTimer); fallbackTimer = null; } }

function updateConn() {
  const el = document.getElementById('connection-status');
  if (sseConnected) { el.textContent = 'Live'; el.className = 'live'; }
  else { el.textContent = 'Offline'; el.className = ''; }
}

// ══════════════════════════════════════════════════════════════════════════════
//  SETTINGS — Unified (failover + routing)
// ══════════════════════════════════════════════════════════════════════════════

// Discovery data: { endpoint_models: { ep: [model, ...] } }
let discoveryData = null;

async function loadSettings() {
  try {
    const [settingsRes, discoveryRes] = await Promise.all([
      fetch('/api/config/settings'),
      fetch('/api/discovery'),
    ]);

    // Failover + routes
    if (settingsRes.ok) {
      const d = await settingsRes.json();
      document.getElementById('fo-max-retries').value = d.failover.max_retries;
      document.getElementById('fo-cb-threshold').value = d.failover.circuit_breaker_threshold;
      document.getElementById('fo-cb-cooldown').value = d.failover.circuit_breaker_cooldown;
      document.getElementById('fo-strategy').value = d.failover.routing_strategy;
      renderRoutes(d.routes || []);
    }

    // Discovery palette
    if (discoveryRes.ok) {
      discoveryData = await discoveryRes.json();
      renderPalette();
    }
  } catch (e) { console.error('settings load:', e); }
}

// ── Palette (model blocks source) ──────────────────────────────────────────

function renderPalette() {
  const container = document.getElementById('model-palette');
  container.innerHTML = '';
  if (!discoveryData || !discoveryData.endpoint_models) {
    container.innerHTML = '<p class="muted">No discovery data.</p>';
    return;
  }
  const epModels = discoveryData.endpoint_models;
  for (const ep of Object.keys(epModels)) {
    for (const model of epModels[ep]) {
      const block = mk('div', { className: 'model-block', draggable: 'true' },
        mk('span', { className: 'ep-tag', textContent: ep }),
        mk('span', { className: 'model-name', textContent: model }),
      );
      block.dataset.endpoint = ep;
      block.dataset.model = model;

      block.addEventListener('dragstart', onPaletteDragStart);
      block.addEventListener('dragend', onDragEnd);
      container.appendChild(block);
    }
  }
  if (!container.children.length) {
    container.innerHTML = '<p class="muted">No models discovered.</p>';
  }
}

// ── Route cards ────────────────────────────────────────────────────────────

function renderRoutes(routes) {
  const container = document.getElementById('routes-container');
  container.innerHTML = '';
  if (!routes.length) {
    container.innerHTML = '<p class="muted">No routes. Click "+ New Route" to create one.</p>';
    return;
  }
  routes.forEach(route => container.appendChild(createRouteCard(route)));
}

function createRouteCard(route) {
  const card = mk('div', { className: 'route-card' });

  // Header
  const header = mk('div', { className: 'route-card-header' },
    mk('label', { textContent: 'Route' }),
    mk('input', { type: 'text', className: 'route-name-input', value: route.name || '', placeholder: 'route-name' }),
    mk('button', { type: 'button', className: 'btn btn-danger btn-sm route-delete', textContent: 'Delete' }),
  );
  header.querySelector('.route-delete').addEventListener('click', () => card.remove());
  card.appendChild(header);

  // Chain (drop zone)
  const chain = mk('div', { className: 'route-chain' });
  setupDropZone(chain);
  card.appendChild(chain);

  // Populate existing steps
  (route.chain || []).forEach((step, i) => {
    const ep = step.server || step.endpoint || '';
    const model = step.model || '';
    chain.appendChild(createChainItem(ep, model, i + 1));
  });
  renumberChain(chain);

  return card;
}

function createChainItem(endpoint, model, order) {
  const item = mk('div', { className: 'chain-item', draggable: 'true' },
    mk('span', { className: 'chain-order', textContent: String(order || '') }),
    mk('span', { className: 'ep-tag', textContent: endpoint }),
    mk('span', { className: 'model-name', textContent: model }),
  );
  item.dataset.endpoint = endpoint;
  item.dataset.model = model;

  const removeBtn = mk('button', { className: 'chain-remove', textContent: '\u00D7', title: 'Remove from chain' });
  removeBtn.addEventListener('click', () => {
    const chain = item.parentElement;
    item.remove();
    renumberChain(chain);
  });
  item.appendChild(removeBtn);

  item.addEventListener('dragstart', onChainDragStart);
  item.addEventListener('dragend', onDragEnd);

  return item;
}

function renumberChain(chain) {
  chain.querySelectorAll('.chain-item').forEach((item, i) => {
    item.querySelector('.chain-order').textContent = i + 1;
  });
}

// ── Drag & Drop ────────────────────────────────────────────────────────────

let dragSource = null;    // the DOM element being dragged
let dragOrigin = null;    // 'palette' | 'chain'
let dragData = null;      // { endpoint, model }

function onPaletteDragStart(e) {
  dragSource = e.currentTarget;
  dragOrigin = 'palette';
  dragData = { endpoint: dragSource.dataset.endpoint, model: dragSource.dataset.model };
  dragSource.classList.add('dragging');
  e.dataTransfer.effectAllowed = 'copy';
  e.dataTransfer.setData('text/plain', `${dragData.endpoint}/${dragData.model}`);
}

function onChainDragStart(e) {
  dragSource = e.currentTarget;
  dragOrigin = 'chain';
  dragData = { endpoint: dragSource.dataset.endpoint, model: dragSource.dataset.model };
  dragSource.classList.add('dragging');
  e.dataTransfer.effectAllowed = 'move';
  e.dataTransfer.setData('text/plain', `${dragData.endpoint}/${dragData.model}`);
}

function onDragEnd() {
  if (dragSource) dragSource.classList.remove('dragging');
  clearMarkers();
  document.querySelectorAll('.route-chain.drag-over').forEach(z => z.classList.remove('drag-over'));
  dragSource = null;
  dragOrigin = null;
  dragData = null;
}

function clearMarkers() {
  document.querySelectorAll('.chain-insert-marker').forEach(m => m.remove());
}

function setupDropZone(chain) {
  chain.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = dragOrigin === 'palette' ? 'copy' : 'move';
    chain.classList.add('drag-over');
    showInsertMarker(chain, e.clientY);
  });

  chain.addEventListener('dragleave', (e) => {
    // Only remove if truly leaving the chain
    if (!chain.contains(e.relatedTarget)) {
      chain.classList.remove('drag-over');
      clearMarkers();
    }
  });

  chain.addEventListener('drop', (e) => {
    e.preventDefault();
    chain.classList.remove('drag-over');
    clearMarkers();

    if (!dragData) return;

    const insertIdx = getInsertIndex(chain, e.clientY);

    // If moving within chains, remove from old position
    if (dragOrigin === 'chain' && dragSource && dragSource.parentElement) {
      dragSource.remove();
    }

    const newItem = createChainItem(dragData.endpoint, dragData.model, 0);
    const items = [...chain.querySelectorAll('.chain-item')];
    if (insertIdx >= items.length) {
      chain.appendChild(newItem);
    } else {
      chain.insertBefore(newItem, items[insertIdx]);
    }
    renumberChain(chain);
  });
}

function getInsertIndex(chain, clientY) {
  const items = [...chain.querySelectorAll('.chain-item')];
  for (let i = 0; i < items.length; i++) {
    const rect = items[i].getBoundingClientRect();
    const midY = rect.top + rect.height / 2;
    if (clientY < midY) return i;
  }
  return items.length;
}

function showInsertMarker(chain, clientY) {
  clearMarkers();
  const items = [...chain.querySelectorAll('.chain-item:not(.dragging)')];
  const marker = mk('div', { className: 'chain-insert-marker' });

  const insertIdx = getInsertIndex(chain, clientY);
  if (insertIdx >= items.length) {
    chain.appendChild(marker);
  } else {
    chain.insertBefore(marker, items[insertIdx]);
  }
}

// ── Collect & Save ──────────────────────────────────────────────────────────

function collectRoutingData() {
  const routes = [];
  document.querySelectorAll('.route-card').forEach(card => {
    const name = card.querySelector('.route-name-input').value.trim();
    if (!name) return;
    const chain = [];
    card.querySelectorAll('.chain-item').forEach(item => {
      const ep = item.dataset.endpoint;
      const mdl = item.dataset.model;
      if (ep && mdl) chain.push({ endpoint: ep, model: mdl });
    });
    if (chain.length) routes.push({ name, chain });
  });
  return routes;
}

function initSettingsControls() {
  document.getElementById('btn-add-route').addEventListener('click', () => {
    const container = document.getElementById('routes-container');
    const hint = container.querySelector('.muted');
    if (hint) hint.remove();
    container.appendChild(createRouteCard({ name: '', chain: [] }));
  });

  document.getElementById('btn-apply-settings').addEventListener('click', async () => {
    const st = document.getElementById('settings-status');
    st.textContent = 'Saving...'; st.className = 'save-status';

    const payload = {
      failover: {
        max_retries:              +document.getElementById('fo-max-retries').value,
        circuit_breaker_threshold:+document.getElementById('fo-cb-threshold').value,
        circuit_breaker_cooldown: +document.getElementById('fo-cb-cooldown').value,
        routing_strategy:          document.getElementById('fo-strategy').value,
      },
      routes: collectRoutingData(),
    };

    try {
      const r = await fetch('/api/config/settings', {
        method: 'PUT', headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload),
      });
      const d = await r.json();
      if (r.ok) {
        st.textContent = 'Applied'; st.className = 'save-status save-ok';
        await loadSettings();
      } else {
        const msg = d.errors ? d.errors.join(', ') : (d.error || 'Failed');
        st.textContent = msg; st.className = 'save-status save-err';
      }
    } catch (err) {
      st.textContent = err.message; st.className = 'save-status save-err';
    }
    setTimeout(() => { st.textContent = ''; }, 4000);
  });
}

// ══════════════════════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════════════════════

initTabs();
refreshMonitor();
loadSettings();
initSettingsControls();
connectSSE();
document.getElementById('btn-refresh').addEventListener('click', () => refreshMonitor());
