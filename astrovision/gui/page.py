"""The desktop application's page: one HTML file, no build step, no framework.

Everything the page shows comes from the JSON API in :mod:`app`; everything
it draws is drawn by the browser. The style is the vetting page's.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AstroVision-X</title>
<style>
  :root { --bg:#0f1117; --panel:#171a23; --line:#262a36; --ink:#e6e8ee; --muted:#8b91a3;
          --accent:#4c8dff; --ok:#2a9d6f; --warn:#e0a82c; --bad:#d9534f; --dim:#1d2130; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; height:100vh; display:flex; flex-direction:column; }
  header { display:flex; align-items:center; gap:14px; padding:10px 18px; border-bottom:1px solid var(--line); background:var(--panel); }
  header h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.2px; }
  header .sub { color:var(--muted); font-size:12px; }
  header .right { margin-left:auto; color:var(--muted); font-size:12px; }
  .shell { display:flex; flex:1; min-height:0; }
  nav { width:190px; background:var(--panel); border-right:1px solid var(--line); padding:12px 8px; display:flex; flex-direction:column; gap:2px; }
  nav button { text-align:left; background:none; border:0; color:var(--ink); padding:9px 12px; border-radius:6px; cursor:pointer; font-size:14px; }
  nav button:hover { background:var(--dim); } nav button.on { background:var(--dim); color:#fff; border-left:3px solid var(--accent); padding-left:9px; }
  nav .spacer { flex:1; } nav .boundary { color:var(--muted); font-size:11px; padding:8px 10px; line-height:1.4; }
  main { flex:1; overflow:auto; padding:16px 18px; min-width:0; }
  .row { display:flex; gap:14px; align-items:flex-start; } .row > * { min-width:0; }
  .col { flex:1; } .col.narrow { flex:0 0 380px; } .col.wide { flex:2; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin-bottom:12px; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:.6px; color:var(--muted); margin:0 0 8px; }
  .card h2 .r { float:right; text-transform:none; letter-spacing:0; font-weight:normal; }
  label.f { display:block; color:var(--muted); font-size:12px; margin:8px 0 3px; }
  input[type=text], input[type=number], select, textarea { width:100%; background:var(--bg); color:var(--ink); border:1px solid var(--line); border-radius:6px; padding:6px 8px; font:inherit; }
  input[type=number] { width:110px; }
  .inline { display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end; } .inline > div { flex:0 0 auto; }
  .btn { border:1px solid var(--line); border-radius:6px; padding:7px 12px; cursor:pointer; background:var(--bg); color:var(--ink); font:inherit; }
  .btn:hover { border-color:var(--accent); } .btn.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  .btn.primary:disabled { opacity:.5; cursor:default; } .btn.small { padding:3px 8px; font-size:12px; }
  .files { max-height:340px; overflow:auto; border:1px solid var(--line); border-radius:6px; background:var(--bg); }
  .files div { padding:5px 8px; cursor:pointer; display:flex; gap:8px; align-items:center; border-bottom:1px solid var(--dim); font-size:13px; }
  .files div:hover { background:var(--dim); } .files div.sel { background:#1f2a45; }
  .files .dir { color:#9ad1ff; } .files .sz { margin-left:auto; color:var(--muted); font-size:11px; }
  .crumb { display:flex; gap:6px; align-items:center; margin-bottom:6px; } .crumb input { flex:1; }
  .stages { display:grid; grid-template-columns: 1fr 70px; gap:2px 10px; font-size:13px; font-variant-numeric:tabular-nums; }
  .stages .n { color:var(--muted); } .stages .n.running { color:var(--accent); } .stages .n.ok { color:var(--ink); }
  .stages .n.failed { color:var(--bad); } .stages .n.skipped { color:var(--muted); opacity:.55; }
  .stages .t { text-align:right; color:var(--muted); }
  .bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; margin:8px 0; } .bar i { display:block; height:100%; background:var(--accent); width:0; transition:width .3s; }
  pre.log { background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:8px; font-size:11.5px; max-height:220px; overflow:auto; margin:0; white-space:pre-wrap; color:var(--muted); }
  .tabs { display:flex; gap:4px; border-bottom:1px solid var(--line); margin-bottom:10px; }
  .tabs button { background:none; border:0; color:var(--muted); padding:7px 12px; cursor:pointer; font:inherit; border-bottom:2px solid transparent; }
  .tabs button.on { color:var(--ink); border-bottom-color:var(--accent); }
  iframe { width:100%; height:calc(100vh - 210px); border:1px solid var(--line); border-radius:6px; background:#fff; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; font-variant-numeric:tabular-nums; }
  th, td { text-align:left; padding:4px 6px; border-bottom:1px solid var(--dim); white-space:nowrap; }
  th { color:var(--muted); font-weight:500; cursor:pointer; position:sticky; top:0; background:var(--panel); }
  tr:hover td { background:var(--dim); } tr.sel td { background:#1f2a45; }
  .tablewrap { max-height:calc(100vh - 300px); overflow:auto; border:1px solid var(--line); border-radius:6px; }
  .kv { display:grid; grid-template-columns: 150px 1fr; gap:3px 10px; font-size:13px; } .kv dt { color:var(--muted); } .kv dd { margin:0; font-variant-numeric:tabular-nums; word-break:break-all; }
  .nums { display:flex; gap:10px; flex-wrap:wrap; } .num { background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:8px 12px; min-width:110px; }
  .num b { display:block; font-size:20px; font-weight:600; } .num span { color:var(--muted); font-size:12px; }
  .warn { color:var(--warn); } .bad { color:var(--bad); } .ok { color:var(--ok); } .muted { color:var(--muted); }
  img.stamp { image-rendering:pixelated; border:1px solid var(--line); border-radius:6px; background:#000; width:256px; height:256px; }
  img.preview { max-width:100%; border:1px solid var(--line); border-radius:6px; background:#000; }
  .cand { display:flex; gap:12px; padding:10px 0; border-bottom:1px solid var(--dim); } .cand img { width:128px; height:128px; }
  .cand .body { flex:1; } .cand .body b { display:block; }
  .badge { display:inline-block; padding:1px 8px; border-radius:999px; font-size:11px; border:1px solid var(--line); margin-right:6px; }
  .badge.transient { color:#ffb86b; } .badge.lens { color:#9ad1ff; } .badge.anomaly { color:#e6a5ff; } .badge.variable { color:#a5ffd6; }
  .toast { position:fixed; bottom:18px; left:50%; transform:translateX(-50%); background:var(--panel); border:1px solid var(--line); padding:8px 14px; border-radius:8px; opacity:0; transition:opacity .2s; z-index:9; }
  .toast.show { opacity:1; } .toast.bad { border-color:var(--bad); }
  ul.plain { margin:4px 0; padding-left:18px; } ul.plain li { margin:2px 0; }
  .jobs div.j { display:flex; gap:10px; padding:7px 4px; border-bottom:1px solid var(--dim); cursor:pointer; align-items:center; }
  .jobs div.j:hover { background:var(--dim); } .jobs .st { min-width:60px; }
  a { color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>AstroVision-X</h1><span class="sub" id="ver"></span>
  <span class="right" id="hdrjob"></span>
</header>
<div class="shell">
  <nav>
    <button data-v="analyze" class="on">Analyse an image</button>
    <button data-v="series">Series &amp; transients</button>
    <button data-v="simulate">Simulate a field</button>
    <button data-v="alerts">Alerts</button>
    <button data-v="jobs">Runs</button>
    <button data-v="about">About</button>
    <div class="spacer"></div>
    <div class="boundary" id="boundary"></div>
  </nav>
  <main id="main"></main>
</div>
<div class="toast" id="toast"></div>

<script>
const $ = id => document.getElementById(id);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({error: 'bad response'}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
};
const post = (path, body) => api(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function toast(msg, bad) { const t = $('toast'); t.textContent = msg; t.className = 'toast show' + (bad ? ' bad' : ''); setTimeout(() => t.className = 'toast', bad ? 5000 : 2200); }
function fmt(v) {
  if (v === null || v === undefined || v === '') return '—';
  if (typeof v === 'number') { if (Number.isInteger(v)) return String(v); if (Math.abs(v) >= 1000) return v.toFixed(2); return Math.abs(v) < 0.01 && v !== 0 ? v.toPrecision(3) : v.toFixed(3).replace(/\.?0+$/, ''); }
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  return String(v);
}
function human(bytes) { if (bytes < 1024) return bytes + ' B'; if (bytes < 1048576) return (bytes/1024).toFixed(0) + ' KB'; return (bytes/1048576).toFixed(1) + ' MB'; }

let status = null, view = 'analyze', currentJob = null, poller = null;
const state = { analyze: {path: '', dir: ''}, series: {paths: [], dir: ''}, alerts: {path: '', dir: ''} };

// ---- file browser --------------------------------------------------------------
function browser(id, kinds, onPick, multi) {
  const el = $(id);
  el.innerHTML = '<div class="crumb"><button class="btn small" id="' + id + '_up">↑</button><input type="text" id="' + id + '_path"><button class="btn small" id="' + id + '_go">Go</button></div><div class="files" id="' + id + '_list"></div>';
  const list = $(id + '_list'), pathBox = $(id + '_path');
  const b = { dir: '', selected: new Set() };
  async function load(path) {
    try {
      const d = await api('/api/browse?path=' + encodeURIComponent(path || ''));
      b.dir = d.path; pathBox.value = d.path; list.innerHTML = '';
      $(id + '_up').onclick = () => d.parent && load(d.parent);
      d.dirs.forEach(x => { const r = document.createElement('div'); r.innerHTML = '<span class="dir">▸ ' + esc(x.name) + '</span>'; r.onclick = () => load(x.path); list.appendChild(r); });
      d.files.filter(f => kinds.includes(f.kind)).forEach(f => {
        const r = document.createElement('div'); r.innerHTML = '<span>' + esc(f.name) + '</span><span class="sz">' + human(f.size) + '</span>';
        if (b.selected.has(f.path)) r.classList.add('sel');
        r.onclick = () => {
          if (multi) { if (b.selected.has(f.path)) b.selected.delete(f.path); else b.selected.add(f.path); r.classList.toggle('sel'); onPick([...b.selected].sort()); }
          else { [...list.children].forEach(c => c.classList.remove('sel')); r.classList.add('sel'); b.selected = new Set([f.path]); onPick(f.path); }
        };
        list.appendChild(r);
      });
      if (!d.dirs.length && !d.files.length) list.innerHTML = '<div class="muted">nothing here this page can open</div>';
    } catch (e) { toast(e.message, true); }
  }
  $(id + '_go').onclick = () => load(pathBox.value);
  pathBox.onkeydown = e => { if (e.key === 'Enter') load(pathBox.value); };
  b.load = load; b.selectAll = () => { b.selected = new Set([...list.querySelectorAll('div')].filter(r => r.querySelector('.sz')).map((r, i) => r)); };
  return b;
}

// ---- views ----------------------------------------------------------------------
function optionsCard(prefix, withRedshift) {
  return '<div class="card"><h2>Options</h2><div class="inline">' +
    '<div><label class="f">Preset</label><select id="' + prefix + '_preset"><option value="">default</option>' + status.presets.map(p => '<option>' + p + '</option>').join('') + '</select></div>' +
    '<div><label class="f">Detection threshold (σ)</label><input type="number" id="' + prefix + '_thr" step="0.1" placeholder="3.5"></div>' +
    (withRedshift ? '<div><label class="f">Redshift (optional)</label><input type="number" id="' + prefix + '_z" step="0.01" placeholder="none"></div>' : '') +
    '</div><label class="f">Output folder</label><input type="text" id="' + prefix + '_out" placeholder="' + esc(status.workdir) + '/astrovision_output">' +
    '<label class="f">Catalog database (optional, SQLite path; gives object histories across runs)</label><input type="text" id="' + prefix + '_db" placeholder="none">' +
    '<label class="f">Reports</label><div class="inline"><label><input type="checkbox" id="' + prefix + '_html" checked> HTML</label><label><input type="checkbox" id="' + prefix + '_text" checked> text</label><label><input type="checkbox" id="' + prefix + '_json" checked> JSON</label></div></div>';
}
function readOptions(prefix) {
  const formats = ['html', 'text', 'json'].filter(f => $(prefix + '_' + f).checked);
  return { preset: $(prefix + '_preset').value || null, threshold: $(prefix + '_thr').value || null,
           redshift: $(prefix + '_z') ? ($(prefix + '_z').value || null) : null,
           output_dir: $(prefix + '_out').value || null, db: $(prefix + '_db').value || null, formats };
}

const views = {
  analyze() {
    $('main').innerHTML = '<div class="row"><div class="col narrow">' +
      '<div class="card"><h2>Image</h2><div id="fb"></div><div id="inspect" class="muted" style="margin-top:8px;font-size:12px"></div></div>' +
      optionsCard('an', true) +
      '<button class="btn primary" id="run" disabled>Run the analysis</button></div>' +
      '<div class="col wide" id="jobpane"><div class="card muted">Pick a FITS image on the left. The pipeline preprocesses it, detects and measures sources, classifies them, looks for lenses and outliers, and writes a report. Every stage is shown as it runs.</div></div></div>';
    const b = browser('fb', ['image'], async path => {
      state.analyze.path = path; $('run').disabled = false;
      $('inspect').innerHTML = 'reading header…';
      try {
        const i = await api('/api/inspect?path=' + encodeURIComponent(path));
        $('inspect').innerHTML = '<b>' + esc(i.name) + '</b> ' + i.shape[1] + '×' + i.shape[0] + ' px, band ' + esc(i.band) + (i.mjd ? ', MJD ' + fmt(i.mjd) : '') +
          (i.wcs ? ', ' + fmt(i.wcs.pixel_scale_arcsec) + '″/px, centre ' + fmt(i.centre[0]) + ', ' + fmt(i.centre[1]) + (i.wcs.derived_from ? ' <span class="warn">(' + esc(i.wcs.derived_from) + ')</span>' : '') : ', <span class="warn">no WCS</span>') +
          '<br>median ' + fmt(i.stats.median) + ', σ ' + fmt(i.stats.std) + ', max ' + fmt(i.stats.max) +
          '<br><img class="preview" src="/api/preview.png?path=' + encodeURIComponent(path) + '" style="margin-top:6px;max-height:220px">';
      } catch (e) { $('inspect').innerHTML = '<span class="bad">' + esc(e.message) + '</span>'; }
    });
    b.load(state.analyze.dir || status.workdir);
    $('run').onclick = async () => {
      try {
        const job = await post('/api/analyze', Object.assign({path: state.analyze.path}, readOptions('an')));
        state.analyze.dir = b.dir; watch(job.id); toast('analysis started');
      } catch (e) { toast(e.message, true); }
    };
    if (currentJob) showJob(currentJob);
  },

  series() {
    $('main').innerHTML = '<div class="row"><div class="col narrow">' +
      '<div class="card"><h2>Epochs <span class="r" id="nsel">none selected</span></h2><div id="fbs"></div><div class="muted" style="font-size:12px;margin-top:6px">Click each epoch of the same field. They are aligned, PSF-matched and differenced; transients are scored real/bogus.</div></div>' +
      optionsCard('se', true) +
      '<div class="card"><h2>Alerts</h2><label class="f">Write transients as an Avro alert file (optional)</label><input type="text" id="se_alerts" placeholder="alerts.avro"></div>' +
      '<button class="btn primary" id="run" disabled>Run the series</button></div>' +
      '<div class="col wide" id="jobpane"><div class="card muted">Select two or more epochs on the left.</div></div></div>';
    const b = browser('fbs', ['image'], paths => { state.series.paths = paths; $('nsel').textContent = paths.length + ' selected'; $('run').disabled = paths.length < 2; }, true);
    b.load(state.series.dir || status.workdir);
    $('run').onclick = async () => {
      try {
        const body = Object.assign({paths: state.series.paths, alerts: $('se_alerts').value || null}, readOptions('se'));
        const job = await post('/api/series', body); state.series.dir = b.dir; watch(job.id); toast('series started');
      } catch (e) { toast(e.message, true); }
    };
    if (currentJob) showJob(currentJob);
  },

  simulate() {
    $('main').innerHTML = '<div class="row"><div class="col narrow"><div class="card"><h2>Synthetic field</h2>' +
      '<div class="inline"><div><label class="f">Size (px)</label><input type="number" id="si_size" value="256"></div><div><label class="f">Seed</label><input type="number" id="si_seed" value="42"></div></div>' +
      '<div class="inline"><div><label class="f">Stars</label><input type="number" id="si_stars" value="80"></div><div><label class="f">Galaxies</label><input type="number" id="si_gal" value="15"></div><div><label class="f">Nebulae</label><input type="number" id="si_neb" value="1"></div></div>' +
      '<div class="inline"><div><label class="f">Clusters</label><input type="number" id="si_clu" value="1"></div><div><label class="f">Lenses</label><input type="number" id="si_len" value="1"></div><div><label class="f">Anomalies</label><input type="number" id="si_ano" value="1"></div></div>' +
      '<div class="inline"><div><label class="f">Epochs (1 = single image)</label><input type="number" id="si_ep" value="1"></div><div><label class="f">Transients (series only)</label><input type="number" id="si_tr" value="2"></div></div>' +
      '<label class="f">Write to</label><input type="text" id="si_out" placeholder="' + esc(status.workdir) + '/synthetic_field.fits">' +
      '<div style="margin-top:10px"><button class="btn primary" id="run">Generate</button></div></div>' +
      '<div class="card muted" style="font-size:12px">The simulator renders stars, Sérsic galaxies, nebulae, clusters, Einstein rings and anomalies through a Moffat PSF with Poisson and read noise, cosmic rays and bad columns, and writes the truth table beside the image. Everything the pipeline claims is measured against these truth tables in the test suite.</div></div>' +
      '<div class="col wide" id="jobpane"></div></div>';
    $('run').onclick = async () => {
      try {
        const job = await post('/api/simulate', {size: +$('si_size').value, seed: +$('si_seed').value, stars: +$('si_stars').value, galaxies: +$('si_gal').value,
          nebulae: +$('si_neb').value, clusters: +$('si_clu').value, lenses: +$('si_len').value, anomalies: +$('si_ano').value, epochs: +$('si_ep').value, transients: +$('si_tr').value, out: $('si_out').value || null});
        watch(job.id);
      } catch (e) { toast(e.message, true); }
    };
    if (currentJob) showJob(currentJob);
  },

  alerts() {
    $('main').innerHTML = '<div class="row"><div class="col narrow"><div class="card"><h2>Alert file</h2><div id="fba"></div></div>' +
      '<div class="card"><h2>Vetting</h2><label class="f">Verdict log</label><input type="text" id="al_log" placeholder="' + esc(status.workdir) + '/verdicts.json"><label class="f">Catalog database (optional)</label><input type="text" id="al_db"><div style="margin-top:10px"><button class="btn primary" id="vet" disabled>Open the vetting page</button></div></div></div>' +
      '<div class="col wide" id="alpane"><div class="card muted">Pick an Avro alert file: this package\'s, ZTF\'s or Rubin\'s. Packets are listed as received, and the vetting page shows each with its cutouts and light curve.</div></div></div>';
    const b = browser('fba', ['alerts'], async path => {
      state.alerts.path = path; $('vet').disabled = false; $('alpane').innerHTML = '<div class="card muted">reading…</div>';
      try {
        const a = await api('/api/alerts?path=' + encodeURIComponent(path));
        const cols = ['object_id', 'candid', 'ra', 'dec', 'mjd', 'band', 'mag', 'mag_err', 'real_bogus', 'deep_real_bogus', 'n_history', 'has_cutouts', 'publisher', 'format'];
        $('alpane').innerHTML = '<div class="card"><h2>' + a.n_packets + ' packet(s) <span class="r">' + esc(a.schema || '') + '</span></h2><div class="tablewrap"><table><thead><tr>' + cols.map(c => '<th>' + c.replace(/_/g, ' ') + '</th>').join('') + '</tr></thead><tbody>' +
          a.rows.map(r => '<tr>' + cols.map(c => '<td>' + esc(fmt(r[c])) + '</td>').join('') + '</tr>').join('') + '</tbody></table></div></div>';
      } catch (e) { $('alpane').innerHTML = '<div class="card bad">' + esc(e.message) + '</div>'; }
    });
    b.load(state.alerts.dir || status.workdir);
    $('vet').onclick = async () => {
      try { const v = await post('/api/vet', {path: state.alerts.path, log: $('al_log').value || null, db: $('al_db').value || null}); window.open(v.url, '_blank'); toast(v.n_items + ' items to vet'); }
      catch (e) { toast(e.message, true); }
    };
  },

  async jobs() {
    const jobs = await api('/api/jobs');
    $('main').innerHTML = '<div class="row"><div class="col narrow"><div class="card jobs"><h2>Runs this session</h2>' +
      (jobs.length ? jobs.map(j => '<div class="j" data-id="' + j.id + '"><span class="st ' + (j.status === 'done' ? 'ok' : j.status === 'failed' ? 'bad' : 'warn') + '">' + j.status + '</span><span>' + esc(j.kind) + ' · ' + esc(j.title) + '</span><span class="muted" style="margin-left:auto">' + (j.seconds ? j.seconds.toFixed(0) + ' s' : '') + '</span></div>').join('') : '<div class="muted">nothing yet</div>') +
      '</div></div><div class="col wide" id="jobpane"></div></div>';
    document.querySelectorAll('.jobs .j').forEach(el => el.onclick = () => watch(el.dataset.id));
    if (currentJob) showJob(currentJob);
  },

  about() {
    const b = status.backends;
    $('main').innerHTML = '<div class="card"><h2>AstroVision-X ' + esc(status.version) + '</h2><p>Computer vision and machine learning for astronomical imagery: detection, photometry, morphology, classification, lens and anomaly search, transients across epochs, alerts, a catalog database and a vetting page for the astronomer\'s verdict.</p>' +
      '<p><b>' + esc(status.boundary) + '</b></p><dl class="kv"><dt>Python</dt><dd>' + esc(status.python) + ' on ' + esc(status.platform) + '</dd><dt>NumPy</dt><dd>' + esc(status.numpy) + '</dd>' +
      Object.keys(b).sort().map(k => '<dt>' + esc(k) + '</dt><dd class="' + (b[k] ? 'ok' : 'muted') + '">' + (b[k] ? 'available' : 'not installed') + '</dd>').join('') +
      '<dt>Working folder</dt><dd>' + esc(status.workdir) + '</dd></dl>' +
      '<p class="muted" style="font-size:12px">This window is a local web page served by the application on 127.0.0.1. It can read any file you can; never expose it on a network. Closing the terminal or pressing Ctrl-C there stops it.</p>' +
      '<button class="btn" id="quit">Quit the application</button></div>';
    $('quit').onclick = async () => { await post('/api/shutdown'); document.body.innerHTML = '<div class="card" style="margin:40px">AstroVision-X has stopped. You can close this tab.</div>'; };
    // The backends are probed off the request thread at start-up; show them once known.
    if (!status.backends_ready) setTimeout(async () => { status = await api('/api/status'); if (view === 'about') show('about'); }, 1500);
  }
};

// ---- jobs -------------------------------------------------------------------------
function watch(id) { currentJob = id; if (poller) clearInterval(poller); showJob(id); poller = setInterval(() => showJob(id), 1500); }
let lastTab = 'summary', lastRendered = '';
async function showJob(id) {
  const pane = $('jobpane'); if (!pane) return;
  let j; try { j = await api('/api/jobs/' + id); } catch (e) { clearInterval(poller); return; }
  $('hdrjob').textContent = j.kind + ' · ' + j.title + ' · ' + j.status + (j.seconds ? ' · ' + j.seconds.toFixed(0) + ' s' : '');
  const done = j.stages.filter(s => ['ok', 'skipped', 'failed'].includes(s.status)).length, total = j.stages.length || 1;
  let html = '<div class="card"><h2>' + esc(j.kind) + ' · ' + esc(j.title) + ' <span class="r ' + (j.status === 'failed' ? 'bad' : j.status === 'done' ? 'ok' : 'warn') + '">' + j.status + '</span></h2>' +
    '<div class="bar"><i style="width:' + (j.status === 'done' ? 100 : Math.round(100 * done / total)) + '%"></i></div>' +
    '<div class="row"><div class="col"><div class="stages">' + j.stages.map(s => '<span class="n ' + s.status + '">' + (s.status === 'ok' ? '✓ ' : s.status === 'running' ? '▶ ' : s.status === 'failed' ? '✗ ' : s.status === 'skipped' ? '– ' : '· ') + esc(s.name) + '</span><span class="t">' + (s.seconds ? s.seconds.toFixed(1) + ' s' : '') + '</span>').join('') + '</div></div>' +
    '<div class="col wide"><pre class="log">' + esc((j.log || []).join('\n')) + '</pre></div></div>' +
    (j.error ? '<p class="bad">' + esc(j.error) + '</p>' : '') + '</div>';
  if (j.status === 'done' || j.status === 'failed') { clearInterval(poller); poller = null; }
  if (j.status === 'done') html += await resultsHtml(j);
  if (html !== lastRendered) { pane.innerHTML = html; lastRendered = html; if (j.status === 'done') bindResults(j); }
  const logEl = pane.querySelector('pre.log'); if (logEl) logEl.scrollTop = logEl.scrollHeight;
}
async function resultsHtml(j) {
  const r = j.result || {};
  if (j.kind === 'simulate') {
    return '<div class="card"><h2>Written</h2><ul class="plain">' + (r.paths || []).map(p => '<li>' + esc(p) + '</li>').join('') + '<li>' + esc(r.truth) + ' (truth table, ' + r.n_objects + ' objects)</li></ul>' +
      '<button class="btn primary" id="analyseit">Analyse ' + ((r.paths || []).length > 1 ? 'these epochs' : 'this image') + '</button></div>';
  }
  const s = r.summary || {}, counts = s.class_counts || {};
  let html = '<div class="tabs">' + ['summary', 'candidates', 'catalog', 'report', 'image', 'files'].map(t => '<button data-t="' + t + '"' + (t === lastTab ? ' class="on"' : '') + '>' + t + '</button>').join('') + '</div><div id="tab"></div>';
  return html;
}
async function renderTab(j, t) {
  lastTab = t; const el = $('tab'); if (!el) return; const r = j.result || {}, s = r.summary || {}, id = j.id;
  document.querySelectorAll('.tabs button').forEach(b => b.classList.toggle('on', b.dataset.t === t));
  if (t === 'summary') {
    const counts = s.class_counts || {};
    el.innerHTML = '<div class="card"><h2>What was found</h2><div class="nums"><div class="num"><b>' + fmt(s.n_sources) + '</b><span>sources</span></div>' +
      Object.keys(counts).map(k => '<div class="num"><b>' + counts[k] + '</b><span>' + esc(k) + '</span></div>').join('') +
      '<div class="num"><b>' + fmt(s.n_transients) + '</b><span>transients</span></div><div class="num"><b>' + fmt(s.n_lens_candidates) + '</b><span>lens candidates</span></div><div class="num"><b>' + fmt(s.n_anomalies) + '</b><span>anomalies ranked</span></div></div>' +
      (j.warnings.length ? '<h2 style="margin-top:12px">What to be careful about</h2><ul class="plain warn">' + j.warnings.map(w => '<li>' + esc(w) + '</li>').join('') + '</ul>' : '') +
      (r.database ? '<p class="muted">Stored in ' + esc(r.database.path) + ': field ' + r.database.field_id + ', ' + r.database.n_detections + ' detections, ' + r.database.n_matched + ' matched to known objects, ' + r.database.n_new_objects + ' new.</p>' : '') +
      (r.alerts ? '<p class="muted">Alerts written: ' + esc(r.alerts.path) + ' (' + r.alerts.n_packets + ' packets).</p>' : '') +
      '<p class="muted" style="font-size:12px">' + esc(status.boundary) + '</p>' +
      '<button class="btn primary" id="vetit">Open the vetting page for these candidates</button></div>';
    $('vetit').onclick = async () => { try { const v = await post('/api/vet', {job_id: id, db: j.params.db || null}); window.open(v.url, '_blank'); toast(v.n_items + ' items to vet'); } catch (e) { toast(e.message, true); } };
  } else if (t === 'candidates') {
    const c = await api('/api/jobs/' + id + '/candidates?limit=40');
    el.innerHTML = '<div class="card"><h2>Ranked follow-up list</h2>' + (c.length ? c.map(x => '<div class="cand"><img class="stamp" src="/api/jobs/' + id + '/cutout.png?x=' + x.position[0] + '&y=' + x.position[1] + '&size=48"><div class="body"><b>#' + x.rank + ' <span class="badge ' + esc(x.kind) + '">' + esc(x.kind) + '</span> score ' + fmt(x.score) + ' · ' + esc(String(x.verdict).replace(/_/g, ' ')) + '</b>' +
      '<div class="muted">pixel (' + fmt(x.position[0]) + ', ' + fmt(x.position[1]) + ')' + (x.sky_position ? ' · RA ' + fmt(x.sky_position[0]) + ' Dec ' + fmt(x.sky_position[1]) : '') + '</div><ul class="plain">' + (x.reasons || []).map(r => '<li>' + esc(r) + '</li>').join('') + '</ul>' +
      (x.caveats && x.caveats.length ? '<div class="muted" style="font-size:12px">' + esc(x.caveats.join(' ')) + '</div>' : '') + '</div></div>').join('') : '<p class="muted">Nothing ranked for follow-up.</p>') + '</div>';
  } else if (t === 'catalog') {
    const st = renderTab.cat || (renderTab.cat = {sort: 'snr', order: 'desc', q: '', offset: 0});
    const c = await api('/api/jobs/' + id + '/catalog?sort=' + st.sort + '&order=' + st.order + '&q=' + encodeURIComponent(st.q) + '&offset=' + st.offset + '&limit=200');
    el.innerHTML = '<div class="card"><h2>Catalog <span class="r">' + c.total + ' sources' + (c.total > 200 ? ', showing ' + (st.offset + 1) + '–' + Math.min(st.offset + 200, c.total) : '') + '</span></h2>' +
      '<div class="inline" style="margin-bottom:8px"><input type="text" id="catq" placeholder="filter (class, flag, id…)" value="' + esc(st.q) + '" style="width:260px"><button class="btn small" id="catprev">‹</button><button class="btn small" id="catnext">›</button><span class="muted" style="font-size:12px">click a column to sort, a row for its cutout</span></div>' +
      '<div class="row"><div class="col wide tablewrap"><table><thead><tr>' + c.columns.map(k => '<th data-k="' + k + '">' + k + (st.sort === k ? (st.order === 'desc' ? ' ▾' : ' ▴') : '') + '</th>').join('') + '</tr></thead><tbody>' +
      c.rows.map(r => '<tr data-x="' + r.x + '" data-y="' + r.y + '">' + c.columns.map(k => '<td>' + esc(fmt(r[k])) + '</td>').join('') + '</tr>').join('') + '</tbody></table></div>' +
      '<div class="col" style="flex:0 0 270px"><img class="stamp" id="catstamp" alt=""><div class="muted" style="font-size:12px" id="catcap">select a row</div></div></div></div>';
    $('catq').onchange = () => { st.q = $('catq').value; st.offset = 0; renderTab(j, 'catalog'); };
    $('catprev').onclick = () => { st.offset = Math.max(0, st.offset - 200); renderTab(j, 'catalog'); };
    $('catnext').onclick = () => { if (st.offset + 200 < c.total) { st.offset += 200; renderTab(j, 'catalog'); } };
    el.querySelectorAll('th').forEach(th => th.onclick = () => { if (st.sort === th.dataset.k) st.order = st.order === 'desc' ? 'asc' : 'desc'; else { st.sort = th.dataset.k; st.order = 'desc'; } renderTab(j, 'catalog'); });
    el.querySelectorAll('tbody tr').forEach(tr => tr.onclick = () => { el.querySelectorAll('tr').forEach(x => x.classList.remove('sel')); tr.classList.add('sel'); $('catstamp').src = '/api/jobs/' + id + '/cutout.png?x=' + tr.dataset.x + '&y=' + tr.dataset.y + '&size=48'; $('catcap').textContent = 'source at (' + tr.dataset.x + ', ' + tr.dataset.y + '), 48 px, asinh stretch'; });
  } else if (t === 'report') {
    el.innerHTML = '<iframe src="/api/jobs/' + id + '/report.html"></iframe>';
  } else if (t === 'image') {
    el.innerHTML = '<div class="card"><h2>The image, asinh stretch, north up</h2><img class="preview" src="/api/jobs/' + id + '/preview.png"></div>';
  } else if (t === 'files') {
    const f = r.files || {};
    el.innerHTML = '<div class="card"><h2>Written to ' + esc(r.output_dir) + '</h2><ul class="plain">' + Object.keys(f).map(k => '<li><b>' + esc(k) + '</b> ' + esc(f[k]) + '</li>').join('') + '</ul></div>';
  }
}
function bindResults(j) {
  document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => renderTab(j, b.dataset.t));
  if ($('tab')) renderTab(j, lastTab);
  const a = $('analyseit');
  if (a) a.onclick = () => { const paths = j.result.paths || []; if (paths.length > 1) { state.series.paths = paths; state.series.dir = paths[0].replace(/[\\/][^\\/]*$/, ''); show('series'); setTimeout(() => { $('nsel').textContent = paths.length + ' selected'; $('run').disabled = false; }, 300); }
    else { state.analyze.path = paths[0]; state.analyze.dir = paths[0].replace(/[\\/][^\\/]*$/, ''); show('analyze'); setTimeout(() => { $('run').disabled = false; $('inspect').textContent = paths[0]; }, 300); } };
}

// ---- boot --------------------------------------------------------------------------
function show(v) { view = v; lastRendered = ''; document.querySelectorAll('nav button').forEach(b => b.classList.toggle('on', b.dataset.v === v)); views[v](); }
document.querySelectorAll('nav button').forEach(b => b.onclick = () => show(b.dataset.v));
(async () => {
  status = await api('/api/status');
  $('ver').textContent = 'v' + status.version; $('boundary').textContent = status.boundary; document.title = 'AstroVision-X ' + status.version;
  show('analyze');
})();
</script>
</body>
</html>
"""

__all__ = ["PAGE"]
