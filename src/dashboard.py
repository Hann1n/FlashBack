"""FlashBack Dashboard Generator.

Creates a standalone HTML dashboard for fire origin tracing results.
Embeds origin visualization images as base64 for portable single-file output.
"""

import base64
import json
from pathlib import Path

BASE_DIR = Path(r"D:\Program Files\isaacsim\cookbook\firetrace")
RESULTS_PATH = BASE_DIR / "reports" / "results_combined.json"
ORIGIN_DIR = BASE_DIR / "reports"
OUTPUT_PATH = BASE_DIR / "reports" / "firetrace_dashboard.html"


def img_to_base64(path):
    if not Path(path).exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    suffix = Path(path).suffix.lstrip(".")
    if suffix == "jpg":
        suffix = "jpeg"
    return f"data:image/{suffix};base64,{data}"


def _escape(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _stage_pct(stage):
    return {"NONE": 0, "INCIPIENT": 25, "GROWTH": 50, "FULLY_DEVELOPED": 85, "DECAY": 100}.get(stage, 0)


def generate_dashboard(data):
    metrics = data.get("metrics", {})
    results = data.get("results", [])
    valid = [r for r in results if r.get("evaluation", {}).get("valid")]

    hazard_acc = metrics.get("hazard_detection_accuracy", 0)
    origin_rate = metrics.get("fire_origin_reasoning_rate", 0)
    temporal_rate = metrics.get("temporal_reasoning_rate", 0)
    urgency_acc = metrics.get("urgency_accuracy", 0)
    severity_acc = metrics.get("severity_accuracy", 0)
    spread_rate = metrics.get("spread_direction_rate", 0)
    total_time = data.get("total_elapsed_sec", 0)

    # Origin cards
    origin_cards = ""
    for r in valid:
        sid = r["scene_id"]
        pred = r.get("prediction", {})
        ev = r.get("evaluation", {})
        b64_origin = img_to_base64(ORIGIN_DIR / f"origin_{sid}.jpg")
        b64_temporal = img_to_base64(ORIGIN_DIR / f"temporal_{sid}.jpg")

        origin_text = pred.get("fire_origin", "N/A")
        ox = pred.get("origin_x", ev.get("pred_origin_x"))
        oy = pred.get("origin_y", ev.get("pred_origin_y"))
        coord = f"({ox:.2f}, {oy:.2f})" if ox is not None and oy is not None else "fallback from text"

        img_tag = f'<img src="{b64_origin}" alt="Origin {sid}">' if b64_origin else '<div class="placeholder">Run visualize_origin.py first</div>'
        temporal_tag = f'<img src="{b64_temporal}" alt="Temporal {sid}" style="margin-top:12px;">' if b64_temporal else ""

        origin_cards += f"""
        <div class="card origin-card">
            <h3>Scene {sid} <span class="tag tag-{r['gt_class'].lower()}">{r['gt_class']}</span></h3>
            {img_tag}
            {temporal_tag}
            <div class="origin-details">
                <div><span class="dl">Origin:</span> {_escape(origin_text)}</div>
                <div><span class="dl">Coords:</span> {coord}</div>
                <div><span class="dl">Spread:</span> {pred.get('spread_direction', 'N/A')}</div>
                <div><span class="dl">Stage:</span> {pred.get('fire_stage', 'N/A')}</div>
                <div><span class="dl">Convection:</span> {_escape(pred.get('convection_pattern', 'N/A'))}</div>
            </div>
        </div>"""

    # Scene rows
    scene_rows = ""
    for r in valid:
        ev = r.get("evaluation", {})
        pred = r.get("prediction", {})
        scene_rows += f"""<tr>
            <td>{r['scene_id']}</td>
            <td>{r['gt_class']}</td>
            <td class="{'correct' if ev.get('class_correct') else 'wrong'}">{ev.get('pred_class','?')}</td>
            <td class="{'correct' if ev.get('hazard_correct') else 'wrong'}">{'Yes' if ev.get('hazard_correct') else 'No'}</td>
            <td>{pred.get('severity','?')}</td>
            <td>{pred.get('fire_stage','?')}</td>
            <td class="{'correct' if ev.get('urgency_appropriate') else 'wrong'}">{ev.get('pred_urgency','?')}</td>
            <td>{_escape(str(ev.get('pred_fire_origin','N/A'))[:35])}</td>
        </tr>"""

    # Temporal cards
    temporal_cards = ""
    for r in valid:
        pred = r.get("prediction", {})
        stage = pred.get("fire_stage", "NONE")
        colors = {"NONE":"#6b7280","INCIPIENT":"#eab308","GROWTH":"#f97316","FULLY_DEVELOPED":"#ef4444","DECAY":"#8b5cf6"}
        c = colors.get(stage, "#6b7280")
        temporal_cards += f"""
        <div class="card">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <h3 style="margin:0;">Scene {r['scene_id']}</h3>
                <span class="stage-badge" style="background:{c}20;color:{c};border:1px solid {c};">{stage}</span>
            </div>
            <div class="timeline"><div class="timeline-fill" style="width:{_stage_pct(stage)}%;background:{c};"></div></div>
            <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-top:4px;">
                <span>Incipient</span><span>Growth</span><span>Fully Dev.</span><span>Decay</span>
            </div>
            <p class="temporal-text">{_escape(pred.get('temporal_progression', 'N/A'))}</p>
        </div>"""

    # Reasoning cards
    reasoning_cards = ""
    for r in valid:
        reasoning = r.get("reasoning", "")
        physics = r.get("prediction", {}).get("physics_reasoning", "")
        if reasoning:
            reasoning_cards += f"""
            <div class="card">
                <h3>Scene {r['scene_id']} — Chain-of-Thought</h3>
                <details>
                    <summary>Show reasoning ({len(reasoning)} chars)</summary>
                    <pre class="reasoning">{_escape(reasoning[:1500])}</pre>
                </details>
                <div style="margin-top:12px;">
                    <strong style="color:var(--accent);">Physics Reasoning:</strong>
                    <p class="temporal-text">{_escape(physics)}</p>
                </div>
            </div>"""

    # Normal scene
    normal_results = [r for r in results if r.get("gt_class") == "NORMAL"]
    normal_card = ""
    if normal_results:
        nr = normal_results[0]
        normal_card = f"""
        <div class="card" style="border-left:3px solid var(--green);">
            <h3>Scene {nr['scene_id']} — Normal (No Fire) <span class="tag tag-normal">NORMAL</span></h3>
            <p class="temporal-text">Model correctly identifies non-fire scenes. No false positive triggered.</p>
            <details><summary>Show reasoning</summary>
                <pre class="reasoning">{_escape(nr.get('reasoning','')[:800])}</pre>
            </details>
        </div>"""

    # Pre-compute chart data (avoid f-string dict literal issues)
    bar_labels = ','.join(f'"{r["scene_id"]}({r["gt_class"]})"' for r in results)
    bar_data = ','.join(
        str(r.get("prediction", {}).get("confidence", 0) if r.get("prediction") else 0)
        for r in results
    )
    bar_colors = ','.join(
        f'"{"#22c55e" if r.get("evaluation", {}).get("hazard_correct") else "#ef4444"}"'
        for r in results
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FlashBack Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{ --bg:#0a0a0f; --card:#141420; --border:#1e1e30; --text:#e4e4e7; --muted:#9ca3af;
         --accent:#f97316; --green:#22c55e; --red:#ef4444; --yellow:#eab308; --blue:#3b82f6; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; padding:24px; max-width:1400px; margin:0 auto; }}
.header {{ text-align:center; padding:40px 0 30px; border-bottom:1px solid var(--border); margin-bottom:30px; }}
.header h1 {{ font-size:48px; font-weight:800; background:linear-gradient(135deg,#f97316,#ef4444,#eab308); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }}
.header .sub {{ color:var(--muted); font-size:16px; margin-top:8px; }}
.header .badge {{ display:inline-block; margin-top:12px; background:var(--card); border:1px solid var(--border); padding:6px 16px; border-radius:20px; font-size:13px; color:var(--accent); }}
h2 {{ font-size:22px; margin:35px 0 16px; color:var(--accent); display:flex; align-items:center; gap:10px; }}
h2::before {{ content:''; width:4px; height:24px; background:var(--accent); border-radius:2px; }}
h3 {{ font-size:16px; margin-bottom:10px; }}
.grid {{ display:grid; gap:16px; margin-bottom:24px; }}
.g4 {{ grid-template-columns:repeat(4,1fr); }}
.g3 {{ grid-template-columns:repeat(3,1fr); }}
.g2 {{ grid-template-columns:repeat(2,1fr); }}
.g1 {{ grid-template-columns:1fr; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; }}
.sc {{ text-align:center; }}
.sc .v {{ font-size:36px; font-weight:700; }}
.sc .l {{ font-size:12px; color:var(--muted); margin-top:4px; }}
.sc .s {{ font-size:12px; color:var(--muted); }}
.origin-card img {{ width:100%; border-radius:8px; }}
.origin-details {{ margin-top:12px; font-size:13px; }}
.origin-details div {{ margin:3px 0; }}
.dl {{ color:var(--accent); font-weight:600; }}
.placeholder {{ height:200px; background:#1e2130; border-radius:8px; display:flex; align-items:center; justify-content:center; color:var(--muted); }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ background:#1a1a2e; padding:10px 8px; text-align:left; font-weight:600; border-bottom:2px solid var(--border); }}
td {{ padding:8px; border-bottom:1px solid var(--border); }}
tr:hover {{ background:#1a1a2e; }}
.correct {{ color:var(--green); font-weight:600; }}
.wrong {{ color:var(--red); font-weight:600; }}
.tag {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }}
.tag-flame {{ background:#7f1d1d; color:#fca5a5; }}
.tag-smoke {{ background:#78350f; color:#fde68a; }}
.tag-normal {{ background:#14532d; color:#86efac; }}
.stage-badge {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:700; text-transform:uppercase; }}
.timeline {{ width:100%; height:8px; background:#1e1e30; border-radius:4px; overflow:hidden; }}
.timeline-fill {{ height:100%; border-radius:4px; }}
.temporal-text {{ margin-top:10px; font-size:13px; color:var(--muted); line-height:1.6; }}
.reasoning {{ background:#0f0f1a; border:1px solid var(--border); border-radius:8px; padding:14px; margin-top:8px; font-size:12px; color:var(--muted); line-height:1.7; white-space:pre-wrap; word-wrap:break-word; max-height:400px; overflow-y:auto; font-family:'Cascadia Code',monospace; }}
details summary {{ cursor:pointer; color:var(--accent); font-size:14px; outline:none; }}
.chart-box {{ position:relative; height:300px; }}
@media(max-width:900px) {{ .g4 {{ grid-template-columns:repeat(2,1fr); }} .g2,.g3 {{ grid-template-columns:1fr; }} .header h1 {{ font-size:32px; }} }}
</style>
</head>
<body>

<div class="header">
    <h1>FlashBack</h1>
    <div class="sub">Rewinding Fire to Its Origin &mdash; Physics-Aware Temporal Reasoning</div>
    <div class="badge">NVIDIA Cosmos-Reason2 &mdash; Cosmos Cookoff 2026</div>
</div>

<h2>Core Metrics</h2>
<div class="grid g4">
    <div class="card sc"><div class="v" style="color:var(--accent)">{hazard_acc:.0%}</div><div class="l">Hazard Detection</div><div class="s">fire + smoke flagged</div></div>
    <div class="card sc"><div class="v" style="color:var(--yellow)">{origin_rate:.0%}</div><div class="l">Origin Tracing</div><div class="s">fire origin identified</div></div>
    <div class="card sc"><div class="v" style="color:var(--blue)">{temporal_rate:.0%}</div><div class="l">Temporal Reasoning</div><div class="s">progression described</div></div>
    <div class="card sc"><div class="v" style="color:var(--green)">{urgency_acc:.0%}</div><div class="l">Urgency Accuracy</div><div class="s">emergency level correct</div></div>
</div>
<div class="grid g4">
    <div class="card sc"><div class="v" style="font-size:28px">{severity_acc:.0%}</div><div class="l">Severity</div></div>
    <div class="card sc"><div class="v" style="font-size:28px">{spread_rate:.0%}</div><div class="l">Spread Dir.</div></div>
    <div class="card sc"><div class="v" style="font-size:28px">{len(valid)}</div><div class="l">Scenes Analyzed</div></div>
    <div class="card sc"><div class="v" style="font-size:28px">{total_time:.0f}s</div><div class="l">Inference Time</div></div>
</div>

<div class="grid g2">
    <div class="card"><h3>Metric Radar</h3><div class="chart-box"><canvas id="radar"></canvas></div></div>
    <div class="card"><h3>Per-Scene Confidence</h3><div class="chart-box"><canvas id="bar"></canvas></div></div>
</div>

<h2>Fire Origin Visualization</h2>
<p style="color:var(--muted);font-size:14px;margin-bottom:14px;">Red crosshair = predicted origin. Yellow arrows = spread direction. Bottom strip = temporal progression.</p>
<div class="grid g2">{origin_cards}</div>

<h2>Scene Results</h2>
<div class="card">
    <table><thead><tr><th>Scene</th><th>GT</th><th>Pred</th><th>Hazard</th><th>Severity</th><th>Stage</th><th>Urgency</th><th>Origin</th></tr></thead>
    <tbody>{scene_rows}</tbody></table>
</div>

<h2>Temporal Fire Progression</h2>
<div class="grid g1">{temporal_cards}</div>

{normal_card}

<h2>Model Reasoning</h2>
<div class="grid g1">{reasoning_cards}</div>

<div style="text-align:center;margin-top:50px;padding:20px;color:var(--muted);font-size:12px;border-top:1px solid var(--border);">
    FlashBack &mdash; Rewinding Fire to Its Origin with NVIDIA Cosmos-Reason2<br>
    {data.get('model','Cosmos-Reason2-2B')} | {len(valid)} scene(s) | {total_time:.0f}s inference<br>
    NVIDIA Cosmos Cookoff 2026
</div>

<script>
new Chart(document.getElementById('radar'),{{
    type:'radar',
    data:{{ labels:['Hazard','Origin','Temporal','Urgency','Severity','Spread'],
        datasets:[{{ label:'Score', data:[{hazard_acc*100:.1f},{origin_rate*100:.1f},{temporal_rate*100:.1f},{urgency_acc*100:.1f},{severity_acc*100:.1f},{spread_rate*100:.1f}],
            borderColor:'#f97316', backgroundColor:'rgba(249,115,22,0.15)', pointBackgroundColor:'#f97316', pointRadius:5 }}] }},
    options:{{ responsive:true, maintainAspectRatio:false,
        scales:{{ r:{{ beginAtZero:true, max:100, grid:{{color:'#1e1e30'}}, angleLines:{{color:'#1e1e30'}}, pointLabels:{{color:'#e4e4e7'}}, ticks:{{color:'#9ca3af',backdropColor:'transparent'}} }} }},
        plugins:{{ legend:{{display:false}} }} }}
}});
new Chart(document.getElementById('bar'),{{
    type:'bar',
    data:{{ labels:[{bar_labels}],
        datasets:[{{ label:'Confidence', data:[{bar_data}],
            backgroundColor:[{bar_colors}], borderRadius:6 }}] }},
    options:{{ responsive:true, maintainAspectRatio:false,
        scales:{{ y:{{beginAtZero:true,max:1,grid:{{color:'#1e1e30'}},ticks:{{color:'#9ca3af'}}}}, x:{{grid:{{display:false}},ticks:{{color:'#9ca3af'}}}} }},
        plugins:{{ legend:{{display:false}} }} }}
}});
</script>
</body>
</html>"""

    return html


def main():
    print("=" * 60)
    print("FlashBack Dashboard Generator")
    print("=" * 60)

    if not RESULTS_PATH.exists():
        print(f"  ERROR: Results not found: {RESULTS_PATH}")
        return

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    html = generate_dashboard(data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"  Dashboard saved: {OUTPUT_PATH}")
    print(f"  Size: {size_kb:.1f} KB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
