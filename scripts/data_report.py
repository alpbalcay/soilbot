"""Self-contained HTML data-analysis report for the soilbot corpus.

Queries the LIVE DuckDB (so counts are current — unlike the stale markdown REPORT.md) plus the
ML cross-validation JSONs, assembles one data dict, and renders a single `data_report.html` with
the data embedded as JSON and charts drawn client-side by Plotly.js (loaded from CDN). Read-only:
opens the DB with read_only=True and writes nothing but the one HTML file.

Run: .venv/bin/python scripts/data_report.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jinja2 import Template  # noqa: E402

from pipeline import db  # noqa: E402
from pipeline.config import Config  # noqa: E402

# SPT-N sanity gate — same bounds the 3D dataset uses (ml/data3d.py); OCR values outside are noise.
SPT_MIN, SPT_MAX = 0, 100
DEPTH_MAX_FT = 200.0


def _rows(con, sql, params=None):
    return con.execute(sql, params or []).fetchall()


def _scalar(con, sql, default=0):
    r = con.execute(sql).fetchone()
    return r[0] if r and r[0] is not None else default


def gather_data(con) -> dict:
    """One SQL battery; each entry is shaped for a specific chart/section."""
    d: dict = {}

    # ---- overview cards -------------------------------------------------------------------
    d["n_borings"] = _scalar(con, "SELECT COUNT(*) FROM borings")
    d["n_labels"] = _scalar(con, "SELECT COUNT(*) FROM soil_labels")
    d["n_strata"] = _scalar(con, "SELECT COUNT(*) FROM strata")
    d["n_spt_borings"] = _scalar(
        con, "SELECT COUNT(DISTINCT boring_id) FROM strata WHERE spt_n IS NOT NULL")
    d["n_spt_meas"] = _scalar(con, "SELECT COUNT(*) FROM strata WHERE spt_n IS NOT NULL")
    d["n_uscs_borings"] = _scalar(
        con, "SELECT COUNT(DISTINCT boring_id) FROM strata WHERE uscs_class IS NOT NULL")
    d["n_gw_borings"] = _scalar(
        con, "SELECT COUNT(DISTINCT boring_id) FROM strata WHERE gw_depth IS NOT NULL")
    d["n_edges"] = _scalar(con, "SELECT COUNT(*) FROM edges")
    d["cov_surficial"] = _scalar(
        con, "SELECT COUNT(*) FROM boring_covariates WHERE surficial_unit IS NOT NULL")
    d["cov_bedrock"] = _scalar(
        con, "SELECT COUNT(*) FROM boring_covariates WHERE bedrock_unit IS NOT NULL")

    # ---- OCR yield (manifest parse outcomes) ----------------------------------------------
    d["manifest"] = {
        s: c for s, c in _rows(
            con, "SELECT status, COUNT(*) FROM manifest WHERE kind='parse' GROUP BY status")
    }

    # ---- spatial: borings (down-rounded), flagged by whether they carry SPT-N -------------
    spt_ids = {r[0] for r in _rows(
        con, "SELECT DISTINCT boring_id FROM strata WHERE spt_n IS NOT NULL")}
    pts = _rows(con, """
        SELECT round(lon,5), round(lat,5), boring_id FROM borings
        WHERE lon IS NOT NULL AND lat IS NOT NULL AND coord_quality_flag='ok'
    """)
    d["map_all_lon"] = [p[0] for p in pts if p[2] not in spt_ids]
    d["map_all_lat"] = [p[1] for p in pts if p[2] not in spt_ids]
    d["map_spt_lon"] = [p[0] for p in pts if p[2] in spt_ids]
    d["map_spt_lat"] = [p[1] for p in pts if p[2] in spt_ids]

    # ---- SPT-N histogram ------------------------------------------------------------------
    d["spt_values"] = [r[0] for r in _rows(
        con, f"SELECT spt_n FROM strata WHERE spt_n BETWEEN {SPT_MIN} AND {SPT_MAX}")]

    # ---- SPT-N vs depth: 5-ft bins, mean + p10/p90 ----------------------------------------
    d["spt_depth"] = [
        {"depth": r[0], "mean": r[1], "p10": r[2], "p90": r[3], "n": r[4]}
        for r in _rows(con, f"""
            SELECT floor(top_depth/5)*5 + 2.5 AS dmid,
                   avg(spt_n), quantile_cont(spt_n,0.10), quantile_cont(spt_n,0.90), COUNT(*)
            FROM strata
            WHERE spt_n BETWEEN {SPT_MIN} AND {SPT_MAX}
              AND top_depth BETWEEN 0 AND {DEPTH_MAX_FT}
            GROUP BY dmid HAVING COUNT(*) >= 20 ORDER BY dmid
        """)]

    # ---- SPT-N by top-8 surficial units ---------------------------------------------------
    top_units = [r[0] for r in _rows(con, f"""
        SELECT c.surficial_unit
        FROM strata s JOIN boring_covariates c USING (boring_id)
        WHERE s.spt_n BETWEEN {SPT_MIN} AND {SPT_MAX} AND c.surficial_unit IS NOT NULL
        GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 8
    """)]
    d["spt_by_unit"] = []
    for u in top_units:
        vals = [r[0] for r in _rows(con, f"""
            SELECT s.spt_n FROM strata s JOIN boring_covariates c USING (boring_id)
            WHERE s.spt_n BETWEEN {SPT_MIN} AND {SPT_MAX} AND c.surficial_unit = ?
        """, [u])]
        d["spt_by_unit"].append({"unit": u, "values": vals})

    # ---- USCS-at-depth frequency (top 15) -------------------------------------------------
    d["uscs_freq"] = [
        {"cls": r[0], "n": r[1]} for r in _rows(con, """
            SELECT uscs_class, COUNT(*) FROM strata WHERE uscs_class IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC LIMIT 15
        """)]

    # ---- groundwater depth (per-boring min) -----------------------------------------------
    d["gw_values"] = [r[0] for r in _rows(con, f"""
        SELECT MIN(gw_depth) FROM strata
        WHERE gw_depth IS NOT NULL AND gw_depth BETWEEN 0 AND {DEPTH_MAX_FT}
        GROUP BY boring_id
    """)]

    # ---- geology prevalence ---------------------------------------------------------------
    d["surficial_top"] = [
        {"unit": r[0], "n": r[1]} for r in _rows(con, """
            SELECT surficial_unit, COUNT(*) FROM boring_covariates
            WHERE surficial_unit IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 12
        """)]
    d["bedrock_top"] = [
        {"unit": r[0], "n": r[1]} for r in _rows(con, """
            SELECT bedrock_unit, COUNT(*) FROM boring_covariates
            WHERE bedrock_unit IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 12
        """)]

    # ---- soil labels: primary label + drainage --------------------------------------------
    d["label_top"] = [
        {"label": r[0], "n": r[1]} for r in _rows(con, """
            SELECT primary_label, COUNT(*) FROM soil_labels
            WHERE primary_label IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 12
        """)]
    d["drainage"] = [
        {"cls": r[0] or "(unknown)", "n": r[1]} for r in _rows(con, """
            SELECT drainage, COUNT(*) FROM soil_labels GROUP BY 1 ORDER BY 2 DESC
        """)]

    # ---- graph edge composition -----------------------------------------------------------
    d["edges_by_type"] = [
        {"type": r[0], "n": r[1]} for r in _rows(con, """
            SELECT edge_type, COUNT(*) FROM edges GROUP BY 1 ORDER BY 2 DESC
        """)]
    return d


def gather_model(out) -> dict:
    """Phase-A (classification) + B1 (3D SPT-N / USCS@depth) metrics from the cv JSONs."""
    def load(name):
        p = out / name
        return json.loads(p.read_text()) if p.exists() else None

    m: dict = {"phase_a": [], "b1": None}

    base = load("baselines.json")
    if base:
        for key, lbl in (("nearest_label", "Baseline: nearest label"),
                         ("rf_covariates", "Baseline: RF on covariates")):
            row = base.get("mean", {}).get(key)
            if row:
                m["phase_a"].append({"name": lbl, **row})
    for name, lbl in (("cv_a1.json", "A1 deterministic GraphSAGE"),
                      ("cv_a2.json", "A2 Bayesian GNN"),
                      ("cv_a3.json", "A3 + geology prior")):
        j = load(name)
        if j and j.get("mean"):
            m["phase_a"].append({"name": lbl, **j["mean"]})

    b1 = load("cv_b1.json")
    if b1:
        m["b1"] = {
            "spt": b1["model"]["mean"]["spt"],
            "uscs": b1["model"]["mean"]["uscs"],
            "baselines": b1.get("baselines", {}),
        }
    return m


TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Soilbot — NJDOT Boring Corpus Data Analysis</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root{--bg:#0f1419;--panel:#1a2129;--ink:#e6edf3;--muted:#8b98a5;--accent:#4ea1d3;--good:#3fb950;--warn:#d29922;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  header{padding:38px 32px 22px;border-bottom:1px solid #243039;background:linear-gradient(180deg,#172029,#0f1419)}
  h1{margin:0 0 6px;font-size:26px;letter-spacing:.2px}
  h2{margin:40px 0 14px;font-size:20px;border-left:3px solid var(--accent);padding-left:12px}
  .sub{color:var(--muted);font-size:13px}
  main{max-width:1160px;margin:0 auto;padding:0 24px 80px}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:22px 0}
  .card{background:var(--panel);border:1px solid #243039;border-radius:10px;padding:16px 18px}
  .card .v{font-size:25px;font-weight:650;color:var(--accent)}
  .card .l{font-size:12px;color:var(--muted);margin-top:3px}
  .grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}
  .chart{background:var(--panel);border:1px solid #243039;border-radius:10px;padding:8px 8px 4px;min-height:340px}
  .note{background:#1c2530;border:1px solid #2d3a47;border-left:3px solid var(--warn);border-radius:8px;padding:12px 16px;color:#cdd9e5;font-size:13.5px;margin:14px 0}
  table{width:100%;border-collapse:collapse;background:var(--panel);border-radius:10px;overflow:hidden;font-size:13.5px}
  th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #243039}
  th:first-child,td:first-child{text-align:left}
  thead th{background:#202a34;color:var(--muted);font-weight:600}
  .best{color:var(--good);font-weight:650}
  footer{color:var(--muted);font-size:12.5px;border-top:1px solid #243039;padding:20px 0;margin-top:50px}
  a{color:var(--accent)}
</style></head>
<body>
<header>
  <h1>NJDOT Soil Boring Corpus — Data Analysis</h1>
  <div class="sub">Generated {{ now }} · live query of <code>data/soilbot.duckdb</code> · charts via Plotly.js</div>
</header>
<main>
  <div class="cards">
    <div class="card"><div class="v">{{ '{:,}'.format(d.n_borings) }}</div><div class="l">Borings (nodes)</div></div>
    <div class="card"><div class="v">{{ '{:,}'.format(d.n_labels) }}</div><div class="l">Soil-label points</div></div>
    <div class="card"><div class="v">{{ '{:,}'.format(d.n_strata) }}</div><div class="l">Strata rows (OCR)</div></div>
    <div class="card"><div class="v">{{ '{:,}'.format(d.n_spt_borings) }}</div><div class="l">Borings w/ SPT-N</div></div>
    <div class="card"><div class="v">{{ '{:,}'.format(d.n_uscs_borings) }}</div><div class="l">Borings w/ USCS</div></div>
    <div class="card"><div class="v">{{ '{:,}'.format(d.n_gw_borings) }}</div><div class="l">Borings w/ GW</div></div>
    <div class="card"><div class="v">{{ '{:,}'.format(d.n_edges) }}</div><div class="l">Graph edges</div></div>
  </div>

  <h2>Coverage &amp; OCR yield</h2>
  <div class="note">SPT-N comes only from the split-spoon (&ldquo;Blows on Spoon&rdquo;) logs &mdash;
    <b>{{ '{:,}'.format(d.n_spt_borings) }}</b> of {{ '{:,}'.format(d.n_borings) }} borings
    (&approx;{{ '%.1f'|format(100*d.n_spt_borings/d.n_borings) }}%). The rest were OCR&rsquo;d but
    carry no depth-resolved blow counts. Treat individual OCR&rsquo;d N-values as noisy
    (no hand-labeled gold set).</div>
  <div class="grid2">
    <div class="chart" id="c_manifest"></div>
    <div class="chart" id="c_yield"></div>
  </div>

  <h2>Spatial distribution</h2>
  <div class="note">Borings cluster on transportation corridors (NJDOT projects), not a uniform grid &mdash;
    a sampling bias to keep in mind for any spatial model. SPT-bearing borings highlighted.</div>
  <div class="chart" id="c_map" style="min-height:560px"></div>

  <h2>Geotechnical signal — SPT-N</h2>
  <div class="grid2">
    <div class="chart" id="c_spthist"></div>
    <div class="chart" id="c_sptdepth"></div>
  </div>
  <div class="chart" id="c_sptunit" style="margin-top:18px"></div>

  <h2>Soil classification</h2>
  <div class="grid2">
    <div class="chart" id="c_uscs"></div>
    <div class="chart" id="c_label"></div>
  </div>
  <div class="grid2" style="margin-top:18px">
    <div class="chart" id="c_drainage"></div>
    <div class="chart" id="c_gw"></div>
  </div>

  <h2>Geology &amp; covariates</h2>
  <div class="grid2">
    <div class="chart" id="c_surficial"></div>
    <div class="chart" id="c_bedrock"></div>
  </div>

  <h2>Graph topology</h2>
  <div class="chart" id="c_edges"></div>

  <h2>Model results</h2>
  {% if m.phase_a %}
  <p class="sub">Phase A — soil-type classification (spatial-block 5-fold CV; lower NLL/ECE = better-calibrated):</p>
  <table><thead><tr><th>Model</th><th>macro-F1</th><th>balanced acc</th><th>accuracy</th><th>NLL</th><th>ECE</th></tr></thead><tbody>
  {% for r in m.phase_a %}<tr><td>{{ r.name }}</td><td>{{ '%.3f'|format(r.macro_f1) }}</td>
    <td>{{ '%.3f'|format(r.balanced_acc) }}</td><td>{{ '%.3f'|format(r.accuracy) }}</td>
    <td>{{ '%.3f'|format(r.nll) }}</td><td>{{ '%.3f'|format(r.ece) }}</td></tr>{% endfor %}
  </tbody></table>
  {% endif %}
  {% if m.b1 %}
  <p class="sub" style="margin-top:22px">B1 — depth-resolved SPT-N (log1p space; CRPS &amp; 90% coverage are the calibration deliverable):</p>
  <table><thead><tr><th>Model</th><th>CRPS &darr;</th><th>cov90</th><th>RMSE(log)</th></tr></thead><tbody>
    <tr><td class="best">B1 Bayesian GNN (depth-resolved)</td><td class="best">{{ '%.3f'|format(m.b1.spt.crps) }}</td>
      <td>{{ '%.3f'|format(m.b1.spt.cov90) }}</td><td>{{ '%.3f'|format(m.b1.spt.rmse) }}</td></tr>
    {% for k,v in m.b1.baselines.items() %}<tr><td>baseline: {{ k }}</td><td>{{ '%.3f'|format(v.crps) }}</td>
      <td>{{ '%.3f'|format(v.cov90) }}</td><td>{{ '%.3f'|format(v.rmse) }}</td></tr>{% endfor %}
  </tbody></table>
  <div class="note">Honest read: B1 <b>ties the depth-mean baseline</b> on CRPS ({{ '%.3f'|format(m.b1.spt.crps) }})
    while being the <b>best-calibrated</b> (90% intervals cover {{ '%.1f'|format(100*m.b1.spt.cov90) }}%), and it
    beats the geology baseline. USCS-at-depth is weak (macro-F1 {{ '%.3f'|format(m.b1.uscs.macro_f1) }},
    acc {{ '%.3f'|format(m.b1.uscs.accuracy) }}) &mdash; OCR class noise + class imbalance.</div>
  {% endif %}

  <footer>
    Sources: NJDOT GDMS borings &amp; soil labels · NJDEP surficial/bedrock geology · NRCS SSURGO ·
    OCR via easyocr. Read-only snapshot; regenerate with
    <code>.venv/bin/python scripts/data_report.py</code>.
  </footer>
</main>
<script>
const D = {{ data_json }}, M = {{ model_json }};
const DARK = {paper_bgcolor:'#1a2129',plot_bgcolor:'#1a2129',font:{color:'#e6edf3',size:12},
  margin:{t:42,r:16,b:48,l:56},legend:{orientation:'h',y:-0.18}};
const CFG = {responsive:true,displayModeBar:false};
const A = '#4ea1d3', G = '#3fb950', cat = ['#4ea1d3','#d29922','#3fb950','#bc8cff','#f778ba','#56d4dd','#e3b341','#ff7b72'];
const lay = (t,x)=>Object.assign({title:{text:t,font:{size:14}}},DARK,x||{});

// OCR manifest outcomes
const mk = Object.keys(D.manifest);
Plotly.newPlot('c_manifest',[{type:'bar',x:mk,y:mk.map(k=>D.manifest[k]),marker:{color:cat}}],
  lay('OCR parse outcomes (manifest)'),CFG);
// yield funnel
Plotly.newPlot('c_yield',[{type:'bar',orientation:'h',
  y:['w/ groundwater','w/ SPT-N','w/ USCS','total borings'],
  x:[D.n_gw_borings,D.n_spt_borings,D.n_uscs_borings,D.n_borings],
  marker:{color:[G,A,'#bc8cff','#56616c']}}],lay('Borings by extracted signal'),CFG);

// NJ map
Plotly.newPlot('c_map',[
  {type:'scattergeo',lon:D.map_all_lon,lat:D.map_all_lat,mode:'markers',name:'no SPT-N',
   marker:{size:2.5,color:'#56616c',opacity:.5}},
  {type:'scattergeo',lon:D.map_spt_lon,lat:D.map_spt_lat,mode:'markers',name:'has SPT-N',
   marker:{size:3.5,color:A,opacity:.8}}],
  Object.assign(lay('Boring locations across New Jersey'),
   {geo:{scope:'usa',fitbounds:'locations',bgcolor:'#1a2129',landcolor:'#222c36',
     subunitcolor:'#3a4753',showlakes:false,showland:true,showcountries:false,resolution:50}}),CFG);

// SPT-N histogram
Plotly.newPlot('c_spthist',[{type:'histogram',x:D.spt_values,nbinsx:50,marker:{color:A}}],
  lay('SPT-N distribution (0–100 blows)',{xaxis:{title:'N (blows/ft)'},yaxis:{title:'count'}}),CFG);
// SPT-N vs depth (mean + p10/p90 band)
const dd=D.spt_depth, dz=dd.map(r=>r.depth);
Plotly.newPlot('c_sptdepth',[
  {x:dd.map(r=>r.p90),y:dz,mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'},
  {x:dd.map(r=>r.p10),y:dz,mode:'lines',fill:'tonextx',fillcolor:'rgba(78,161,211,.18)',
   line:{width:0},name:'p10–p90',hoverinfo:'skip'},
  {x:dd.map(r=>r.mean),y:dz,mode:'lines+markers',line:{color:A},name:'mean N'}],
  lay('SPT-N vs depth',{xaxis:{title:'N (blows/ft)'},yaxis:{title:'depth (ft)',autorange:'reversed'}}),CFG);
// SPT-N by surficial unit (box)
Plotly.newPlot('c_sptunit',D.spt_by_unit.map((u,i)=>({type:'box',y:u.values,name:u.unit,
  marker:{color:cat[i%cat.length]},boxpoints:false})),
  lay('SPT-N by surficial geology unit (top 8)',{yaxis:{title:'N (blows/ft)'},showlegend:false,
   margin:{t:42,r:16,b:120,l:56},xaxis:{tickangle:-35}}),CFG);

// USCS frequency
Plotly.newPlot('c_uscs',[{type:'bar',x:D.uscs_freq.map(r=>r.cls),y:D.uscs_freq.map(r=>r.n),marker:{color:'#bc8cff'}}],
  lay('USCS-at-depth class frequency',{yaxis:{title:'intervals'}}),CFG);
// primary label
Plotly.newPlot('c_label',[{type:'bar',orientation:'h',y:D.label_top.map(r=>r.label).reverse(),
  x:D.label_top.map(r=>r.n).reverse(),marker:{color:A}}],
  lay('Soil-label primary code (top 12)',{margin:{t:42,r:16,b:48,l:90}}),CFG);
// drainage
Plotly.newPlot('c_drainage',[{type:'pie',labels:D.drainage.map(r=>r.cls),values:D.drainage.map(r=>r.n),
  marker:{colors:cat},textinfo:'percent'}],lay('Soil-label drainage class'),CFG);
// groundwater
Plotly.newPlot('c_gw',[{type:'histogram',x:D.gw_values,nbinsx:40,marker:{color:'#56d4dd'}}],
  lay('Groundwater depth (per boring)',{xaxis:{title:'depth (ft)'},yaxis:{title:'borings'}}),CFG);

// geology
Plotly.newPlot('c_surficial',[{type:'bar',orientation:'h',y:D.surficial_top.map(r=>r.unit).reverse(),
  x:D.surficial_top.map(r=>r.n).reverse(),marker:{color:G}}],
  lay('Surficial geology units (top 12)',{margin:{t:42,r:16,b:48,l:180}}),CFG);
Plotly.newPlot('c_bedrock',[{type:'bar',orientation:'h',y:D.bedrock_top.map(r=>r.unit).reverse(),
  x:D.bedrock_top.map(r=>r.n).reverse(),marker:{color:'#e3b341'}}],
  lay('Bedrock geology units (top 12)',{margin:{t:42,r:16,b:48,l:180}}),CFG);

// edges
Plotly.newPlot('c_edges',[{type:'bar',x:D.edges_by_type.map(r=>r.type),y:D.edges_by_type.map(r=>r.n),
  marker:{color:cat}}],lay('Graph edges by relation type'),CFG);
</script>
</body></html>
""")


def main():
    cfg = Config.load(None)
    con = db.connect(cfg, read_only=True)
    try:
        data = gather_data(con)
    finally:
        con.close()
    out = cfg.abspath(cfg.get("ml", "out_dir", default="data/ml"))
    model = gather_model(out)

    html = TEMPLATE.render(
        d=data, m=model, now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        data_json=json.dumps(data, default=float),
        model_json=json.dumps(model, default=float),
    )
    dest = cfg.abspath("data_report.html")
    dest.write_text(html, encoding="utf-8")

    print(f"wrote {dest} ({len(html)//1024} KB)")
    print(f"  borings={data['n_borings']:,}  strata={data['n_strata']:,}  "
          f"spt_borings={data['n_spt_borings']:,}  uscs_borings={data['n_uscs_borings']:,}  "
          f"gw_borings={data['n_gw_borings']:,}")
    print(f"  spt_measurements={data['n_spt_meas']:,}  edges={data['n_edges']:,}  "
          f"map_points={len(data['map_all_lon'])+len(data['map_spt_lon']):,}")


if __name__ == "__main__":
    main()
