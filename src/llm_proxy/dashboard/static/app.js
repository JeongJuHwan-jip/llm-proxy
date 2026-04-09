'use strict';

const FALLBACK_INTERVAL = 30; // seconds — used only when SSE is disconnected

let countdown = FALLBACK_INTERVAL;
let fallbackTimer = null;
let sseConnected = false;

// ── Helpers ─────────────────────────────────────────────────────────────────

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour12: false });
}

function fmtLatency(ms) {
  if (ms == null) return '—';
  return ms >= 1000 ? (ms / 1000).toFixed(2) + 's' : Math.round(ms) + 'ms';
}

function fmtRate(r) {
  return (r * 100).toFixed(1) + '%';
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'className') node.className = v;
    else if (k === 'textContent') node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) {
    if (typeof child === 'string') node.appendChild(document.createTextNode(child));
    else if (child) node.appendChild(child);
  }
  return node;
}

// ── Endpoint cards ───────────────────────────────────────────────────────────

async function fetchStatus() {
  const res = await fetch('/api/status');
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
}

function renderEndpoints(statuses) {
  const container = document.getElementById('endpoints-container');
  if (!statuses.length) {
    container.innerHTML = '<p class="loading">No endpoints configured.</p>';
    return;
  }

  container.innerHTML = '';
  for (const s of statuses) {
    const state = s.circuit_state;
    const card = el('div', { className: `card ${state}` },
      el('div', { className: 'card-name', textContent: s.name }),
      el('div', { className: 'card-url', textContent: s.url }),
      el('span', { className: `card-state state-${state}`, textContent: state.replace('_', ' ') }),
      el('div', { className: 'card-stats' },
        el('div', {}, 'Requests'),
        el('div', {}, el('span', { textContent: String(s.total_requests) })),
        el('div', {}, 'Failures'),
        el('div', {}, el('span', { textContent: String(s.total_failures) })),
        el('div', {}, 'Timeouts'),
        el('div', {}, el('span', { textContent: String(s.total_timeouts) })),
        el('div', {}, 'Timeout rate'),
        el('div', {}, el('span', { textContent: fmtRate(s.timeout_rate) })),
        el('div', {}, 'Avg latency'),
        el('div', {}, el('span', { textContent: fmtLatency(s.avg_latency_ms) })),
        el('div', {}, 'CB failures'),
        el('div', {}, el('span', { textContent: String(s.consecutive_failures) })),
      )
    );
    container.appendChild(card);
  }
}

// ── Request log table ────────────────────────────────────────────────────────

async function fetchRequests() {
  const res = await fetch('/api/requests?limit=50');
  if (!res.ok) throw new Error('Failed to fetch requests');
  return res.json();
}

function renderRequests({ total, rows }) {
  document.getElementById('total-count').textContent = total.toLocaleString();
  const tbody = document.getElementById('requests-body');

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="loading">No requests yet.</td></tr>';
    return;
  }

  tbody.innerHTML = '';
  for (const r of rows) {
    const attemptsHtml = r.attempts.map(a => {
      const cls = a.success ? 'attempt-ok' : (a.is_timeout ? 'attempt-to' : 'attempt-err');
      const icon = a.success ? '✓' : (a.is_timeout ? '⏱' : '✗');
      return `<span class="${cls}" title="${a.endpoint} — ${Math.round(a.latency_ms)}ms${a.error ? ': ' + a.error : ''}">${icon} ${a.endpoint}</span>`;
    }).join('<br>');

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${fmtTime(r.timestamp)}</td>
      <td>${escHtml(r.model)}</td>
      <td>${escHtml(r.selected_endpoint || '—')}</td>
      <td class="status-${r.status}">${r.status}</td>
      <td>${fmtLatency(r.total_latency_ms)}</td>
      <td>${r.is_stream ? '⚡ stream' : 'sync'}</td>
      <td class="attempt-list">${attemptsHtml}</td>
    `;
    tbody.appendChild(tr);
  }
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Refresh ─────────────────────────────────────────────────────────────────

async function refresh() {
  try {
    const [statuses, requestData] = await Promise.all([
      fetchStatus(),
      fetchRequests(),
    ]);
    renderEndpoints(statuses);
    renderRequests(requestData);
  } catch (err) {
    console.error('Refresh error:', err);
  }
}

// ── SSE (real-time updates) ─────────────────────────────────────────────────

function connectSSE() {
  const source = new EventSource('/api/events');

  source.onopen = () => {
    sseConnected = true;
    stopFallbackTimer();
    updateConnectionStatus();
  };

  source.onmessage = () => {
    refresh();
  };

  source.onerror = () => {
    sseConnected = false;
    source.close();
    updateConnectionStatus();
    startFallbackTimer();
    // Reconnect after 3 seconds
    setTimeout(connectSSE, 3000);
  };
}

// ── Fallback polling (when SSE is disconnected) ─────────────────────────────

function startFallbackTimer() {
  if (fallbackTimer) return;
  countdown = FALLBACK_INTERVAL;
  fallbackTimer = setInterval(() => {
    countdown -= 1;
    document.getElementById('countdown').textContent = countdown;
    if (countdown <= 0) {
      refresh();
      countdown = FALLBACK_INTERVAL;
    }
  }, 1000);
}

function stopFallbackTimer() {
  if (fallbackTimer) {
    clearInterval(fallbackTimer);
    fallbackTimer = null;
  }
}

function updateConnectionStatus() {
  const el = document.getElementById('refresh-info');
  if (sseConnected) {
    el.innerHTML = 'Live <button id="btn-refresh">Refresh now</button>';
  } else {
    el.innerHTML = 'Reconnecting... fallback poll: <span id="countdown">30</span>s <button id="btn-refresh">Refresh now</button>';
  }
  document.getElementById('btn-refresh').addEventListener('click', () => refresh());
}

// ── Init ────────────────────────────────────────────────────────────────────

refresh();
connectSSE();
