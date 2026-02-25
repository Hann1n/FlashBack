"""FireTrace: Interactive Streamlit Dashboard.

Clean, light-themed dashboard for fire origin tracing results.
Run: streamlit run firetrace_app.py --server.port 8501 --server.headless true
"""

import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from PIL import Image

BASE_DIR = Path(r"D:\Program Files\isaacsim\cookbook\firetrace")
RESULTS_PATH = BASE_DIR / "reports" / "results_combined.json"
ORIGIN_DIR = BASE_DIR / "reports"
DATA_DIR = Path(r"D:\Program Files\isaacsim\cookbook\data\fire_detection_aihub\Sample")
RAW_BASE = DATA_DIR / "01.원천데이터" / "화재현상"
NEW_DATA_DIR = BASE_DIR / "data" / "fire_dataset"

SUBDIRS = {"FL": "불꽃", "SM": "연기", "NONE": "정상"}
CLASS_TO_CODE = {"FLAME": "FL", "SMOKE": "SM", "NORMAL": "NONE"}

CLASS_COLORS = {"FLAME": "#EF4444", "SMOKE": "#F59E0B", "NORMAL": "#22C55E"}
CLASS_EMOJI = {"FLAME": "🔥", "SMOKE": "💨", "NORMAL": "✅"}


def fallback_origin(text):
    text = str(text).lower()
    x, y = 0.5, 0.5
    if "left" in text: x = 0.25
    elif "right" in text: x = 0.75
    if "top" in text or "upper" in text: y = 0.25
    elif "bottom" in text or "lower" in text or "base" in text: y = 0.75
    if "center" in text or "middle" in text:
        if "left" not in text and "right" not in text: x = 0.5
        if "top" not in text and "bottom" not in text: y = 0.5
    return x, y


@st.cache_data
def load_results():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_frame(scene_id, gt_class, frame_pct=0.5, video_path=None):
    """Load a frame at given percentage position."""
    class_code = CLASS_TO_CODE.get(gt_class, "NONE")
    subdir = SUBDIRS.get(class_code, "정상")
    jpg_dir = RAW_BASE / subdir / scene_id / "JPG"
    if jpg_dir.exists():
        frames = sorted(jpg_dir.glob("*.jpg"))
        if frames:
            idx = min(int(len(frames) * frame_pct), len(frames) - 1)
            return Image.open(frames[idx])

    if video_path:
        import cv2
        vp = Path(video_path)
        if not vp.exists():
            vp = NEW_DATA_DIR / "preprocessed" / f"{scene_id}_prep.mp4"
        if not vp.exists():
            for sub in ["samples/part1", "samples/part2", "samples/part3", ""]:
                candidate = NEW_DATA_DIR / sub / f"{scene_id}.mp4"
                if candidate.exists():
                    vp = candidate
                    break
        if vp.exists():
            cap = cv2.VideoCapture(str(vp))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(n_frames * frame_pct))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
    return None


def render_metric_card(label, value, color="#1E40AF", icon=""):
    """Render a clean metric card."""
    st.markdown(f"""
    <div style="background: white; border: 1px solid #E5E7EB; border-radius: 12px;
                padding: 16px 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size: 13px; color: #6B7280; margin-bottom: 4px;">{icon} {label}</div>
        <div style="font-size: 28px; font-weight: 700; color: {color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_badge(text, color="#1E40AF"):
    """Render an inline badge."""
    return f'<span style="background:{color}; color:white; padding:3px 10px; border-radius:20px; font-size:13px; font-weight:600;">{text}</span>'


def render_result_icon(ok):
    return "✅" if ok else "❌"


def main():
    st.set_page_config(
        page_title="FireTrace",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Light theme CSS
    st.markdown("""
    <style>
    /* Force light background */
    .stApp { background-color: #F8FAFC !important; }
    section[data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E5E7EB; }
    section[data-testid="stSidebar"] * { color: #1F2937 !important; }

    /* Header */
    .firetrace-header {
        background: linear-gradient(135deg, #1E40AF 0%, #7C3AED 100%);
        border-radius: 16px; padding: 28px 36px; margin-bottom: 24px;
        color: white;
    }
    .firetrace-header h1 { margin: 0; font-size: 36px; font-weight: 800; color: white; }
    .firetrace-header p { margin: 4px 0 0; font-size: 15px; color: rgba(255,255,255,0.85); }

    /* Section headers */
    .section-title {
        font-size: 20px; font-weight: 700; color: #1F2937;
        border-bottom: 2px solid #E5E7EB; padding-bottom: 8px; margin: 24px 0 16px;
    }

    /* Scene card */
    .scene-card {
        background: white; border: 1px solid #E5E7EB; border-radius: 12px;
        padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* Info row */
    .info-row { display: flex; justify-content: space-between; padding: 8px 0;
                border-bottom: 1px solid #F3F4F6; }
    .info-row:last-child { border-bottom: none; }
    .info-label { color: #6B7280; font-size: 13px; font-weight: 500; }
    .info-value { color: #1F2937; font-size: 14px; font-weight: 600; }

    /* Summary table */
    .summary-table { width: 100%; border-collapse: collapse; font-size: 14px; }
    .summary-table th { background: #F8FAFC; color: #6B7280; font-weight: 600;
                        padding: 10px 12px; text-align: left; border-bottom: 2px solid #E5E7EB; }
    .summary-table td { padding: 10px 12px; border-bottom: 1px solid #F3F4F6; color: #1F2937; }
    .summary-table tr:hover { background: #F0F9FF; }

    /* Override streamlit defaults for light theme */
    .stMarkdown, .stText, p, span, label, .stSelectbox label { color: #1F2937 !important; }
    h1, h2, h3, h4 { color: #1F2937 !important; }
    .stSlider label { color: #1F2937 !important; }
    div[data-testid="stMetricValue"] { color: #1F2937 !important; }
    </style>
    """, unsafe_allow_html=True)

    data = load_results()
    metrics = data.get("metrics", {})
    results = data.get("results", [])
    valid = [r for r in results if r.get("evaluation", {}).get("valid")]

    # ==================== HEADER ====================
    st.markdown("""
    <div class="firetrace-header">
        <h1>🔥 FireTrace</h1>
        <p>Fire Origin Tracing with Physics-Aware Temporal Reasoning &mdash; NVIDIA Cosmos-Reason2</p>
    </div>
    """, unsafe_allow_html=True)

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("### 🎯 Scene Explorer")
        scene_options = [f"{r['scene_id']}  {CLASS_EMOJI.get(r['gt_class'], '')} {r['gt_class']}" for r in results]
        selected_scene = st.selectbox("Select a scene", scene_options, label_visibility="collapsed")
        selected_idx = scene_options.index(selected_scene)
        selected_result = results[selected_idx]

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.markdown(f"**Model:** Cosmos-Reason2-2B")
        st.markdown(f"**Scenes:** {data.get('total_scenes', 0)}")
        st.markdown(f"**Valid predictions:** {len(valid)} / {len(results)}")

        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        st.markdown("- [FiftyOne](http://localhost:5151)")
        st.markdown(f"- [HTML Dashboard](file:///{str(BASE_DIR / 'reports' / 'firetrace_dashboard.html')})")

    # ==================== METRICS ====================
    st.markdown('<div class="section-title">📈 Performance Overview</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric_card("Hazard Detection", f"{metrics.get('hazard_detection_accuracy', 0):.0%}", "#22C55E", "🛡️")
    with m2:
        render_metric_card("Origin Tracing", f"{metrics.get('fire_origin_reasoning_rate', 0):.0%}", "#3B82F6", "📍")
    with m3:
        render_metric_card("Temporal Reasoning", f"{metrics.get('temporal_reasoning_rate', 0):.0%}", "#8B5CF6", "⏱️")
    with m4:
        render_metric_card("Severity Accuracy", f"{metrics.get('severity_accuracy', 0):.0%}", "#F59E0B", "⚡")
    with m5:
        render_metric_card("Spread Detection", f"{metrics.get('spread_direction_rate', 0):.0%}", "#EC4899", "↗️")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Charts row
    col_radar, col_bar = st.columns(2)

    with col_radar:
        st.markdown('<div class="scene-card">', unsafe_allow_html=True)
        categories = ['Hazard', 'Origin', 'Temporal', 'Urgency', 'Severity', 'Spread']
        values = [
            metrics.get('hazard_detection_accuracy', 0) * 100,
            metrics.get('fire_origin_reasoning_rate', 0) * 100,
            metrics.get('temporal_reasoning_rate', 0) * 100,
            metrics.get('urgency_accuracy', 0) * 100,
            metrics.get('severity_accuracy', 0) * 100,
            metrics.get('spread_direction_rate', 0) * 100,
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            line_color='#3B82F6',
            fillcolor='rgba(59,130,246,0.12)',
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 100], gridcolor='#E5E7EB', tickfont=dict(size=11, color='#9CA3AF')),
                angularaxis=dict(gridcolor='#E5E7EB', tickfont=dict(size=12, color='#374151')),
                bgcolor='white',
            ),
            paper_bgcolor='white', plot_bgcolor='white',
            font_color='#374151', height=340, margin=dict(t=30, b=30, l=60, r=60),
            title=dict(text="Metric Radar", font=dict(size=15, color='#374151')),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_bar:
        st.markdown('<div class="scene-card">', unsafe_allow_html=True)
        scene_labels = [r['scene_id'] for r in results]
        confidences = [r.get("prediction", {}).get("confidence", 0) if r.get("prediction") else 0 for r in results]
        bar_colors = [CLASS_COLORS.get(r['gt_class'], '#94A3B8') for r in results]

        fig_bar = go.Figure(go.Bar(
            x=scene_labels, y=confidences,
            marker_color=bar_colors,
            marker_line=dict(width=0),
        ))
        fig_bar.update_layout(
            yaxis=dict(range=[0, 1], title="Confidence", gridcolor='#F3F4F6',
                       tickfont=dict(color='#6B7280'), title_font=dict(color='#374151')),
            xaxis=dict(title="Scene", tickangle=-45,
                       tickfont=dict(color='#6B7280', size=10), title_font=dict(color='#374151')),
            paper_bgcolor='white', plot_bgcolor='white',
            font_color='#374151', height=340, margin=dict(t=40, b=80),
            title=dict(text="Per-Scene Confidence", font=dict(size=15, color='#374151')),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==================== SCENE DETAIL ====================
    r = selected_result
    ev = r.get("evaluation", {})
    pred = r.get("prediction", {}) or {}
    gt_class = r['gt_class']

    badge_color = CLASS_COLORS.get(gt_class, '#94A3B8')
    st.markdown(f'''
    <div class="section-title">
        🎬 Scene: <strong>{r["scene_id"]}</strong> &nbsp;
        {render_badge(gt_class, badge_color)}
        &nbsp;
        {render_badge("Source: " + r.get("source", "unknown"), "#6B7280")}
    </div>
    ''', unsafe_allow_html=True)

    if ev.get("valid") and pred:
        # === Image + Info columns ===
        col_img, col_info = st.columns([3, 2])

        with col_img:
            st.markdown('<div class="scene-card">', unsafe_allow_html=True)

            frame_pct = st.slider("Scrub through video frames", 0.0, 1.0, 0.5, 0.01,
                                  key=f"slider_{r['scene_id']}")

            frame_img = load_frame(r['scene_id'], gt_class, frame_pct,
                                   video_path=r.get('video_path'))

            if frame_img is not None:
                ox = pred.get("origin_x")
                oy = pred.get("origin_y")
                if ox is None or oy is None:
                    ox, oy = fallback_origin(pred.get("fire_origin", ""))

                w, h = frame_img.size

                fig = go.Figure()
                fig.add_layout_image(
                    dict(source=frame_img, xref="x", yref="y",
                         x=0, y=0, sizex=w, sizey=h,
                         sizing="stretch", layer="below")
                )

                # Predicted origin (red)
                if ox > 0 or oy > 0:
                    fig.add_trace(go.Scatter(
                        x=[ox * w], y=[oy * h],
                        mode='markers+text',
                        marker=dict(size=16, color='#EF4444', symbol='x-thin',
                                    line=dict(width=3, color='#EF4444')),
                        text=['Predicted'], textposition='top center',
                        textfont=dict(color='#EF4444', size=12),
                        name='Predicted Origin',
                    ))

                # GT origin (green)
                gt_ox = r.get("gt_origin_x")
                gt_oy = r.get("gt_origin_y")
                if gt_ox is not None and gt_oy is not None:
                    fig.add_trace(go.Scatter(
                        x=[gt_ox * w], y=[gt_oy * h],
                        mode='markers+text',
                        marker=dict(size=14, color='#22C55E', symbol='circle',
                                    line=dict(width=2, color='#22C55E')),
                        text=['GT'], textposition='bottom center',
                        textfont=dict(color='#22C55E', size=11),
                        name='GT Origin',
                    ))

                # Spread arrows
                arrows_raw = pred.get("spread_arrows", [])
                if not arrows_raw:
                    sd = str(pred.get("spread_direction", "")).lower()
                    if "up" in sd:
                        arrows_raw.append({"from_x": ox, "from_y": oy, "to_x": ox, "to_y": max(0, oy - 0.25)})
                    if "outward" in sd:
                        arrows_raw.append({"from_x": ox, "from_y": oy, "to_x": min(1, ox + 0.2), "to_y": max(0, oy - 0.15)})
                        arrows_raw.append({"from_x": ox, "from_y": oy, "to_x": max(0, ox - 0.2), "to_y": max(0, oy - 0.15)})

                for arrow in arrows_raw:
                    fig.add_annotation(
                        x=arrow["to_x"] * w, y=arrow["to_y"] * h,
                        ax=arrow["from_x"] * w, ay=arrow["from_y"] * h,
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=3, arrowsize=1.5,
                        arrowwidth=2, arrowcolor="#F59E0B",
                    )

                fig.update_xaxes(range=[0, w], showgrid=False, visible=False)
                fig.update_yaxes(range=[h, 0], showgrid=False, visible=False, scaleanchor="x")
                fig.update_layout(
                    height=max(360, h // 2),
                    paper_bgcolor='white', plot_bgcolor='white',
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5,
                                font=dict(size=11, color='#374151'), bgcolor='rgba(255,255,255,0.8)'),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not load frame for this scene.")

            st.markdown('</div>', unsafe_allow_html=True)

        with col_info:
            # Prediction card
            st.markdown('<div class="scene-card">', unsafe_allow_html=True)
            st.markdown("**🔍 Prediction**")

            pred_class = ev.get("pred_class", "?")
            pred_color = CLASS_COLORS.get(pred_class.split("|")[0], "#6B7280")
            st.markdown(f"""
            <div class="info-row"><span class="info-label">Classification</span>
                <span class="info-value">{render_badge(pred_class, pred_color)}</span></div>
            <div class="info-row"><span class="info-label">Severity</span>
                <span class="info-value">{pred.get('severity', '—')}</span></div>
            <div class="info-row"><span class="info-label">Fire Stage</span>
                <span class="info-value">{pred.get('fire_stage', '—')}</span></div>
            <div class="info-row"><span class="info-label">Urgency</span>
                <span class="info-value">{pred.get('urgency', '—')}</span></div>
            <div class="info-row"><span class="info-label">Confidence</span>
                <span class="info-value">{pred.get('confidence', 0):.0%}</span></div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # Origin card
            st.markdown('<div class="scene-card">', unsafe_allow_html=True)
            st.markdown("**📍 Fire Origin**")

            origin_text = pred.get('fire_origin', 'N/A')
            if len(origin_text) > 60:
                origin_text = origin_text[:57] + "..."
            coord_x = pred.get("origin_x")
            coord_y = pred.get("origin_y")

            html = f'<div class="info-row"><span class="info-label">Location</span><span class="info-value">{origin_text}</span></div>'
            if coord_x is not None:
                html += f'<div class="info-row"><span class="info-label">Predicted Coord</span><span class="info-value">({coord_x:.2f}, {coord_y:.2f})</span></div>'

            gt_ox = r.get("gt_origin_x")
            gt_oy = r.get("gt_origin_y")
            if gt_ox is not None and gt_oy is not None:
                html += f'<div class="info-row"><span class="info-label">GT Coord</span><span class="info-value" style="color:#22C55E">({gt_ox:.2f}, {gt_oy:.2f})</span></div>'
                origin_dist = ev.get("origin_distance")
                if origin_dist is not None:
                    dist_color = "#22C55E" if origin_dist < 0.2 else ("#F59E0B" if origin_dist < 0.4 else "#EF4444")
                    html += f'<div class="info-row"><span class="info-label">Distance</span><span class="info-value" style="color:{dist_color}">{origin_dist:.3f}</span></div>'

            html += f'<div class="info-row"><span class="info-label">Spread Dir.</span><span class="info-value">{pred.get("spread_direction", "—")}</span></div>'
            st.markdown(html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # Evaluation card
            st.markdown('<div class="scene-card">', unsafe_allow_html=True)
            st.markdown("**✅ Evaluation**")
            st.markdown(f"""
            <div class="info-row"><span class="info-label">Hazard Detected</span>
                <span class="info-value">{render_result_icon(ev.get('hazard_correct'))}</span></div>
            <div class="info-row"><span class="info-label">Severity Correct</span>
                <span class="info-value">{render_result_icon(ev.get('severity_correct'))}</span></div>
            <div class="info-row"><span class="info-label">Urgency Appropriate</span>
                <span class="info-value">{render_result_icon(ev.get('urgency_appropriate'))}</span></div>
            <div class="info-row"><span class="info-label">Inference Time</span>
                <span class="info-value">{r.get('elapsed_sec', 0):.0f}s</span></div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # === Below the columns: details ===
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            # Temporal progression
            st.markdown('<div class="scene-card">', unsafe_allow_html=True)
            st.markdown("**⏱️ Temporal Progression**")
            stage = pred.get("fire_stage", "NONE")
            stage_pct = {"NONE": 0, "INCIPIENT": 25, "GROWTH": 50, "FULLY_DEVELOPED": 85, "DECAY": 100}.get(stage, 0)
            st.progress(stage_pct / 100, text=f"Stage: {stage} ({stage_pct}%)")
            st.markdown(f"<p style='color:#374151; font-size:13px; line-height:1.6;'>{pred.get('temporal_progression', 'N/A')}</p>",
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with detail_col2:
            # Physics reasoning
            st.markdown('<div class="scene-card">', unsafe_allow_html=True)
            st.markdown("**🧪 Physics Reasoning**")
            st.markdown(f"<p style='color:#374151; font-size:13px; line-height:1.6;'>{pred.get('physics_reasoning', 'N/A')}</p>",
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Media section
        media_tabs = []
        media_keys = []

        origin_img_path = ORIGIN_DIR / f"origin_{r['scene_id']}.jpg"
        temporal_path = ORIGIN_DIR / f"temporal_{r['scene_id']}.jpg"
        video_path = ORIGIN_DIR / f"firetrace_{r['scene_id']}.mp4"

        if origin_img_path.exists():
            media_tabs.append("Origin Overlay")
            media_keys.append("origin")
        if temporal_path.exists():
            media_tabs.append("Temporal Strip")
            media_keys.append("temporal")
        if video_path.exists():
            media_tabs.append("Tracked Video")
            media_keys.append("video")

        reasoning = r.get("reasoning", "")
        if reasoning:
            media_tabs.append("Chain-of-Thought")
            media_keys.append("cot")

        if media_tabs:
            tabs = st.tabs(media_tabs)
            for tab, key in zip(tabs, media_keys):
                with tab:
                    if key == "origin":
                        st.image(str(origin_img_path), use_container_width=True)
                    elif key == "temporal":
                        st.image(str(temporal_path), use_container_width=True)
                    elif key == "video":
                        st.video(str(video_path))
                    elif key == "cot":
                        st.code(reasoning[:3000], language=None)

    else:
        st.markdown('<div class="scene-card">', unsafe_allow_html=True)
        st.warning("No valid prediction for this scene (model output could not be parsed or scene classified as NORMAL).")
        reasoning = r.get("reasoning", "")
        if reasoning:
            with st.expander("View model reasoning"):
                st.code(reasoning[:2000], language=None)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==================== ALL SCENES TABLE ====================
    st.markdown('<div class="section-title">📋 All Scenes Summary</div>', unsafe_allow_html=True)

    # Build HTML table for clean look
    rows_html = ""
    for res in results:
        ev2 = res.get("evaluation", {})
        pred2 = res.get("prediction", {}) or {}
        sid = res["scene_id"]
        gt = res["gt_class"]
        gt_badge = render_badge(gt, CLASS_COLORS.get(gt, '#94A3B8'))

        pred_cls = ev2.get("pred_class", "—")
        if pred_cls != "—":
            pred_badge = render_badge(pred_cls, CLASS_COLORS.get(pred_cls.split("|")[0], '#94A3B8'))
        else:
            pred_badge = '<span style="color:#9CA3AF">—</span>'

        hazard = render_result_icon(ev2.get("hazard_correct")) if ev2.get("valid") else "—"
        sev = pred2.get("severity", "—")
        urg = pred2.get("urgency", "—")
        origin = str(pred2.get("fire_origin", "—"))[:35]
        if len(origin) == 35:
            origin += "..."
        elapsed = f"{res.get('elapsed_sec', 0):.0f}s"
        src = res.get("source", "—")

        rows_html += f"""
        <tr>
            <td><strong>{sid}</strong></td>
            <td>{gt_badge}</td>
            <td>{pred_badge}</td>
            <td style="text-align:center">{hazard}</td>
            <td>{sev}</td>
            <td>{urg}</td>
            <td style="font-size:12px; color:#6B7280">{origin}</td>
            <td>{elapsed}</td>
        </tr>
        """

    st.markdown(f"""
    <div class="scene-card" style="overflow-x:auto;">
    <table class="summary-table">
        <thead>
            <tr>
                <th>Scene</th><th>GT</th><th>Pred</th><th>Hazard</th>
                <th>Severity</th><th>Urgency</th><th>Origin</th><th>Time</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#9CA3AF; font-size:13px; padding:16px 0;">
        FireTrace &mdash; Fire Origin Tracing with NVIDIA Cosmos-Reason2 | Cosmos Cookoff 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
