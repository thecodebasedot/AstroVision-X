"""The vetting page: one file, no network, no framework.

It is served by :mod:`server` and talks to it over a handful of JSON routes.
Everything the astronomer needs is on one screen: the cutout large, the
background-subtracted cutout beside it, the pipeline's numbers and reasons,
the object's history when there is one, and the keys. A verdict is one
keystroke and is written before the next item loads.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AstroVision-X vetting</title>
<style>
  :root { --bg:#0f1117; --panel:#171a23; --line:#262a36; --ink:#e6e8ee; --muted:#8b91a3;
          --real:#2a9d6f; --bogus:#d9534f; --unsure:#e0a82c; --accent:#4c8dff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { display:flex; align-items:center; gap:16px; padding:10px 18px;
           border-bottom:1px solid var(--line); background:var(--panel); }
  header h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.2px; }
  header .field { color:var(--muted); }
  header label { margin-left:auto; color:var(--muted); }
  header input { background:var(--bg); color:var(--ink); border:1px solid var(--line);
                 border-radius:6px; padding:5px 8px; width:180px; }
  main { display:grid; grid-template-columns: 1fr 380px; gap:18px; padding:18px; }
  .stamps { display:flex; gap:14px; flex-wrap:wrap; }
  .stamp { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }
  .stamp img { display:block; width:288px; height:288px; image-rendering:pixelated; border-radius:4px; }
  .stamp .cap { color:var(--muted); font-size:12px; margin-top:6px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin-bottom:12px; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:.6px; color:var(--muted); margin:0 0 8px; }
  .kv { display:grid; grid-template-columns: 130px 1fr; gap:3px 10px; font-size:13px; }
  .kv dt { color:var(--muted); } .kv dd { margin:0; font-variant-numeric: tabular-nums; }
  ul { margin:0; padding-left:18px; } li { margin:2px 0; }
  .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid var(--line); }
  .badge.transient { color:#ffb86b; } .badge.lens { color:#9ad1ff; } .badge.anomaly { color:#e6a5ff; } .badge.source { color:var(--muted); }
  .keys { display:flex; gap:10px; flex-wrap:wrap; }
  .key { border:1px solid var(--line); border-radius:8px; padding:8px 12px; cursor:pointer; background:var(--bg); color:var(--ink); }
  .key b { display:inline-block; min-width:18px; text-align:center; border:1px solid var(--line); border-radius:4px; margin-right:6px; padding:0 4px; }
  .key.real:hover { border-color:var(--real); } .key.bogus:hover { border-color:var(--bogus); } .key.unsure:hover { border-color:var(--unsure); }
  textarea { width:100%; background:var(--bg); color:var(--ink); border:1px solid var(--line); border-radius:6px; padding:6px; min-height:48px; }
  .progress { height:6px; background:var(--line); border-radius:3px; overflow:hidden; margin-top:8px; }
  .progress i { display:block; height:100%; background:var(--accent); width:0; }
  .toast { position:fixed; bottom:18px; left:50%; transform:translateX(-50%); background:var(--panel);
           border:1px solid var(--line); padding:8px 14px; border-radius:8px; opacity:0; transition:opacity .2s; }
  .toast.show { opacity:1; }
  canvas { width:100%; height:110px; background:var(--bg); border-radius:4px; }
  .done { padding:40px; text-align:center; color:var(--muted); }
  .disclaimer { color:var(--muted); font-size:12px; padding:0 18px 18px; }
</style>
</head>
<body>
<header>
  <h1>AstroVision-X vetting</h1>
  <span class="field" id="field"></span>
  <span id="counter" class="field"></span>
  <label>Reviewer <input id="reviewer" placeholder="your name (required)"></label>
</header>
<main>
  <section>
    <div class="stamps">
      <div class="stamp"><img id="stamp" alt="cutout"><div class="cap">image, asinh stretch</div></div>
      <div class="stamp" id="subwrap"><img id="stamp_sub" alt="background-subtracted cutout"><div class="cap">background subtracted</div></div>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>Verdict</h2>
      <div class="keys">
        <button class="key real" data-label="real"><b>R</b>real</button>
        <button class="key bogus" data-label="bogus"><b>B</b>bogus</button>
        <button class="key unsure" data-label="unsure"><b>U</b>unsure</button>
        <button class="key" data-action="skip"><b>S</b>skip</button>
        <button class="key" data-action="prev"><b>&larr;</b>previous</button>
      </div>
      <div style="margin-top:10px"><textarea id="note" placeholder="note (optional)"></textarea></div>
      <div class="progress"><i id="bar"></i></div>
      <div id="progress" style="color:var(--muted); font-size:12px; margin-top:6px"></div>
    </div>
    <div class="card" id="historycard" hidden>
      <h2>History across epochs</h2>
      <canvas id="lc" width="600" height="110"></canvas>
      <div id="historytext" style="color:var(--muted); font-size:12px"></div>
    </div>
  </section>
  <aside>
    <div class="card">
      <h2>Candidate</h2>
      <dl class="kv" id="meta"></dl>
    </div>
    <div class="card"><h2>Why the pipeline flagged it</h2><ul id="reasons"></ul></div>
    <div class="card"><h2>Caveats</h2><ul id="caveats"></ul></div>
    <div class="card"><h2>Evidence</h2><dl class="kv" id="evidence"></dl></div>
    <div class="card"><h2>Measurements</h2><dl class="kv" id="measurements"></dl></div>
  </aside>
</main>
<p class="disclaimer">The pipeline ranks candidates and shows its evidence. Nothing here is a
confirmed detection; the verdict recorded is yours, under your name, and is kept alongside
what the model said so the two can be compared later.</p>
<div class="toast" id="toast"></div>
<script>
(function () {
  const $ = id => document.getElementById(id);
  const reviewer = $('reviewer');
  try { reviewer.value = localStorage.getItem('avx_reviewer') || ''; } catch (e) {}
  reviewer.addEventListener('change', () => { try { localStorage.setItem('avx_reviewer', reviewer.value); } catch (e) {} });

  let current = null, order = [], cursor = -1;

  function toast(text, colour) {
    const t = $('toast'); t.textContent = text; t.style.borderColor = colour || 'var(--line)';
    t.classList.add('show'); setTimeout(() => t.classList.remove('show'), 900);
  }
  function fmt(v) {
    if (v === null || v === undefined) return '—';
    if (typeof v === 'number') return Math.abs(v) >= 1000 || Math.abs(v) < 0.01 && v !== 0 ? v.toPrecision(4) : v.toFixed(3).replace(/\.?0+$/, '');
    if (Array.isArray(v)) return v.length ? v.join(', ') : '—';
    return String(v);
  }
  function fill(dl, obj) {
    dl.innerHTML = '';
    for (const [k, v] of Object.entries(obj || {})) {
      const dt = document.createElement('dt'); dt.textContent = k.replace(/_/g, ' ');
      const dd = document.createElement('dd'); dd.textContent = fmt(v);
      dl.appendChild(dt); dl.appendChild(dd);
    }
    if (!dl.children.length) dl.innerHTML = '<dt>—</dt><dd></dd>';
  }
  function list(ul, items) {
    ul.innerHTML = '';
    (items && items.length ? items : ['—']).forEach(t => { const li = document.createElement('li'); li.textContent = t; ul.appendChild(li); });
  }
  function drawHistory(rows) {
    const card = $('historycard');
    const pts = (rows || []).filter(r => r.mjd !== null && r.flux !== null);
    if (pts.length < 1) { card.hidden = true; return; }
    card.hidden = false;
    const c = $('lc'), ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height);
    const xs = pts.map(p => p.mjd), ys = pts.map(p => p.flux);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const sx = v => 30 + (x1 > x0 ? (v - x0) / (x1 - x0) : 0.5) * (c.width - 50);
    const sy = v => c.height - 20 - (y1 > y0 ? (v - y0) / (y1 - y0) : 0.5) * (c.height - 35);
    ctx.strokeStyle = '#262a36'; ctx.beginPath(); ctx.moveTo(30, c.height - 20); ctx.lineTo(c.width - 20, c.height - 20); ctx.stroke();
    ctx.strokeStyle = '#4c8dff'; ctx.lineWidth = 2; ctx.beginPath();
    pts.forEach((p, i) => { const x = sx(p.mjd), y = sy(p.flux); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke();
    ctx.fillStyle = '#4c8dff';
    pts.forEach(p => { ctx.beginPath(); ctx.arc(sx(p.mjd), sy(p.flux), 4, 0, 6.283); ctx.fill();
      if (p.flux_err) { ctx.strokeStyle = '#4c8dff'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(sx(p.mjd), sy(p.flux - p.flux_err)); ctx.lineTo(sx(p.mjd), sy(p.flux + p.flux_err)); ctx.stroke(); } });
    ctx.fillStyle = '#8b91a3'; ctx.font = '11px system-ui';
    ctx.fillText('MJD ' + x0.toFixed(1), 30, c.height - 6); ctx.fillText(x1.toFixed(1), c.width - 70, c.height - 6);
    ctx.fillText('flux', 2, 12);
    $('historytext').textContent = pts.length + ' detection' + (pts.length === 1 ? '' : 's') + ' in the catalog database' +
      (pts.length > 1 ? ', ' + [...new Set(pts.map(p => p.band))].join('/') + ' band' : '');
  }
  function show(item) {
    current = item;
    if (!item) {
      document.querySelector('main').innerHTML = '<div class="done">Every item has a verdict. Thank you.<br><br>' +
        '<span id="progress"></span></div>';
      refreshProgress(); return;
    }
    $('stamp').src = '/api/cutout/' + item.item_id + '.png?t=' + Date.now();
    $('subwrap').hidden = !item.has_subtracted;
    if (item.has_subtracted) $('stamp_sub').src = '/api/cutout/' + item.item_id + '.png?kind=subtracted&t=' + Date.now();
    const meta = {
      kind: item.kind, rank: item.rank, 'pipeline verdict': item.model_verdict.replace(/_/g, ' '),
      'model label': item.model_label, 'model confidence': item.model_confidence,
      score: item.score, 'x, y': item.x.toFixed(1) + ', ' + item.y.toFixed(1),
      'ra, dec': item.ra === null ? '—' : item.ra.toFixed(6) + ', ' + item.dec.toFixed(6),
      'source id': item.source_id, 'previous verdicts': item.previous && item.previous.length ? item.previous.map(v => v.reviewer + ': ' + v.label).join('; ') : 'none'
    };
    fill($('meta'), meta);
    const dt = $('meta').firstChild; if (dt) { const dd = dt.nextSibling; dd.innerHTML = '<span class="badge ' + item.kind + '">' + item.kind + '</span>'; }
    list($('reasons'), item.reasons); list($('caveats'), item.caveats);
    fill($('evidence'), item.evidence); fill($('measurements'), item.measurements);
    drawHistory(item.history);
    $('note').value = '';
    $('counter').textContent = 'item ' + item.item_id + ' of ' + item.n_items;
    refreshProgress();
  }
  async function refreshProgress() {
    const p = await (await fetch('/api/progress')).json();
    const el = $('progress'); if (!el) return;
    el.textContent = p.n_done + ' of ' + p.n_items + ' decided · real ' + p.counts.real + ' · bogus ' + p.counts.bogus +
      ' · unsure ' + p.counts.unsure + (p.disagreements ? ' · ' + p.disagreements + ' with disagreeing reviewers' : '');
    const bar = $('bar'); if (bar) bar.style.width = (100 * p.n_done / Math.max(p.n_items, 1)) + '%';
  }
  async function load(direction) {
    const r = await fetch('/api/next?after=' + (current ? current.item_id : 0) + '&direction=' + (direction || 'next') +
                          '&reviewer=' + encodeURIComponent(reviewer.value));
    const item = await r.json();
    show(item.item_id ? item : null);
  }
  async function verdict(label) {
    if (!current) return;
    if (!reviewer.value.trim()) { toast('Enter your name first: a verdict needs a reviewer', 'var(--bogus)'); reviewer.focus(); return; }
    const r = await fetch('/api/verdict', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ item_id: current.item_id, label: label, reviewer: reviewer.value.trim(), note: $('note').value }) });
    if (!r.ok) { toast('Not recorded: ' + (await r.text()), 'var(--bogus)'); return; }
    toast(label + ' recorded', label === 'real' ? 'var(--real)' : label === 'bogus' ? 'var(--bogus)' : 'var(--unsure)');
    load('next');
  }
  document.querySelectorAll('.key').forEach(b => b.addEventListener('click', () => {
    if (b.dataset.label) verdict(b.dataset.label); else if (b.dataset.action === 'skip') load('next'); else load('prev');
  }));
  document.addEventListener('keydown', e => {
    if (e.target === reviewer || e.target === $('note')) { if (e.key === 'Escape') e.target.blur(); return; }
    const k = e.key.toLowerCase();
    if (k === 'r') verdict('real'); else if (k === 'b') verdict('bogus'); else if (k === 'u') verdict('unsure');
    else if (k === 's' || k === 'arrowright' || k === 'n') load('next'); else if (k === 'arrowleft' || k === 'p') load('prev');
    else if (k === '/') { $('note').focus(); e.preventDefault(); }
  });
  fetch('/api/queue').then(r => r.json()).then(q => { $('field').textContent = q.field || ''; });
  load('next');
})();
</script>
</body>
</html>
"""
