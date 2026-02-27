"""
app.py — StrideFlow Streamlit Dashboard
=======================================
Run with:
    streamlit run app.py
"""

import os
import sys
import math

import numpy as np
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "videos")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "data", "pose_landmarker_heavy.task")

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.insert(0, SRC_DIR)

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="StrideFlow · Running Form Analyzer",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1c2230 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.metric-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 10px;
}
.metric-card .value {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-card .unit {
    font-size: 0.8rem;
    color: #8b949e;
    margin-bottom: 10px;
}
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 99px;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-good    { background:#0d4a1f; color:#3fb950; border:1px solid #238636; }
.badge-warn    { background:#4a2a0d; color:#d29922; border:1px solid #9e6a03; }
.badge-danger  { background:#4a0d12; color:#f85149; border:1px solid #da3633; }
.badge-neutral { background:#21262d; color:#8b949e; border:1px solid #30363d; }

/* Knee warning table */
.knee-row {
    display:flex; justify-content:space-between; align-items:center;
    padding: 8px 0; border-bottom: 1px solid #21262d; font-size:0.88rem;
}
.knee-row:last-child { border-bottom: none; }

/* Status pills */
.status-pill {
    display:inline-flex; align-items:center; gap:6px;
    padding:6px 14px; border-radius:99px;
    font-size:0.82rem; font-weight:600;
}
.pill-ready   { background:#0d4a1f; color:#3fb950; border:1px solid #238636; }
.pill-waiting { background:#21262d; color:#8b949e; border:1px solid #30363d; }
.pill-running { background:#0d2d4a; color:#58a6ff; border:1px solid #1f6feb; }

/* Section headings */
.section-heading {
    font-size:0.78rem; font-weight:700; letter-spacing:0.10em;
    text-transform:uppercase; color:#58a6ff;
    margin:28px 0 14px; padding-bottom:6px;
    border-bottom:1px solid #21262d;
}

[data-testid="stFileUploader"] {
    border:2px dashed #30363d !important;
    border-radius:12px !important;
    background:#161b22 !important;
}
[data-testid="stFileUploader"]:hover { border-color:#58a6ff !important; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
defaults = {
    "video_path":   None,
    "csv_path":     None,
    "report":       None,
    "df_landmarks": None,
    "df_header":    None,
    "processing":   False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:8px 0 20px'>
        <div style='font-size:2.4rem'>🏃</div>
        <div style='font-size:1.15rem; font-weight:700; color:#e6edf3; letter-spacing:0.04em'>
            StrideFlow
        </div>
        <div style='font-size:0.75rem; color:#8b949e; margin-top:2px'>
            Running Form Analyzer
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**📂 Video Input**")
    uploaded = st.file_uploader(
        "Drop a running video here",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed",
    )

    st.markdown("**⚙️ Settings**")
    fps_override = st.number_input(
        "Video FPS", min_value=1.0, max_value=240.0, value=30.0, step=1.0,
        help="Must match the actual recording frame rate for accurate cadence.",
    )

    model_ok = os.path.isfile(MODEL_PATH)
    if not model_ok:
        st.warning(
            "⚠️ Pose model not found.\n\n"
            "```bash\ncurl -L -o data/pose_landmarker_heavy.task \\\n"
            "  https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_heavy/float16/latest/"
            "pose_landmarker_heavy.task\n```"
        )

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem; color:#484f58; text-align:center'>"
        "MediaPipe PoseLandmarker · Heavy model<br>"
        "Tracks Hip · Knee · Ankle · Shoulder</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Handle upload → persist to disk
# ══════════════════════════════════════════════════════════════════════════════

if uploaded is not None:
    save_path = os.path.join(DATA_DIR, uploaded.name)
    if st.session_state.video_path != save_path:
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.video_path   = save_path
        st.session_state.csv_path     = None
        st.session_state.report       = None
        st.session_state.df_landmarks = None
        st.session_state.df_header    = None
        st.session_state.processing   = False


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='font-size:1.9rem; font-weight:800; color:#e6edf3;
           letter-spacing:-0.01em; margin-bottom:2px'>
    🏃 StrideFlow <span style='color:#58a6ff'>Dashboard</span>
</h1>
<p style='color:#8b949e; font-size:0.9rem; margin-top:0'>
    Upload a running video · Analyze biomechanics · View real-time metrics
</p>
""", unsafe_allow_html=True)

st.divider()

# Status pills
col_s1, col_s2, _ = st.columns([1.6, 1.8, 5])
with col_s1:
    if st.session_state.video_path:
        st.markdown(
            f"<div class='status-pill pill-ready'>✅ "
            f"{os.path.basename(st.session_state.video_path)[:24]}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='status-pill pill-waiting'>📂 No video</div>",
            unsafe_allow_html=True,
        )
with col_s2:
    if st.session_state.csv_path:
        st.markdown(
            "<div class='status-pill pill-ready'>✅ Analysis complete</div>",
            unsafe_allow_html=True,
        )
    elif st.session_state.processing:
        st.markdown(
            "<div class='status-pill pill-running'>⏳ Processing…</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='status-pill pill-waiting'>⏸ Awaiting run</div>",
            unsafe_allow_html=True,
        )

if st.session_state.video_path is None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Upload a video** in the sidebar to get started.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Run button
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.csv_path is None:
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        run_clicked = st.button(
            "▶  Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=not model_ok,
        )
    if run_clicked:
        st.session_state.processing = True
        st.rerun()

if st.session_state.csv_path is None and not st.session_state.processing:
    st.markdown(
        "<p style='color:#8b949e; margin-top:12px'>"
        "Click <b>▶ Run Analysis</b> to begin pose extraction.</p>",
        unsafe_allow_html=True,
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Analysis pipeline  (runs once per video upload)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.processing and st.session_state.csv_path is None:
    with st.spinner("🔍 Extracting pose landmarks (headless)… this may take a minute."):
        from pose_engine import process_video
        csv_path = process_video(
            video_path = st.session_state.video_path,
            model_path = MODEL_PATH,
            output_dir = OUTPUT_DIR,
            display    = False,          # headless — no OpenCV window
        )
        st.session_state.csv_path   = csv_path
        st.session_state.processing = False

    with st.spinner("📊 Computing biomechanical metrics…"):
        from analyzer import analyze, load_csv
        report = analyze(
            csv_path    = csv_path,
            fps         = fps_override,
            output_dir  = OUTPUT_DIR,
            save_report = True,
        )
        data, header = load_csv(csv_path)
        st.session_state.report       = report
        st.session_state.df_landmarks = data
        st.session_state.df_header    = header

    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Results — guard until analysis is done
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.report is None:
    st.stop()

report = st.session_state.report
data   = st.session_state.df_landmarks
header = st.session_state.df_header
k      = report.knee
c      = report.cadence
v      = report.oscillation


# ══════════════════════════════════════════════════════════════════════════════
# Helper: render one metric card
# ══════════════════════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, unit: str, badge: str, badge_class: str) -> str:
    return f"""
    <div class='metric-card'>
        <div class='label'>{label}</div>
        <div class='value'>{value}</div>
        <div class='unit'>{unit}</div>
        <div class='badge {badge_class}'>{badge}</div>
    </div>
    """


def cadence_badge(spm: float) -> tuple[str, str]:
    if spm <= 0:     return "No data",  "badge-neutral"
    if spm < 160:    return "Low",      "badge-danger"
    if spm <= 180:   return "Good",     "badge-good"
    return               "Elite",      "badge-warn"

def bounce_badge(pct: float) -> tuple[str, str]:
    if pct <= 0:     return "No data",  "badge-neutral"
    if pct < 8:      return "Optimal",  "badge-good"
    if pct < 12:     return "Moderate", "badge-warn"
    return               "High",       "badge-danger"

def knee_badge(deg: float) -> tuple[str, str]:
    if deg > 170:    return "Overstride ⚠", "badge-danger"
    if deg > 150:    return "Extended",  "badge-warn"
    return               "Healthy",    "badge-good"


# ══════════════════════════════════════════════════════════════════════════════
# Metric cards row
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='section-heading'>📊 Summary Metrics</div>",
            unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

# Cadence
spm   = c.cadence_spm if c else 0.0
cb, cc = cadence_badge(spm)
with col1:
    st.markdown(
        metric_card("Cadence", f"{spm:.0f}", "steps / min", cb, cc),
        unsafe_allow_html=True,
    )

# Vertical bounce
vb    = v.vertical_bounce_pct if v else 0.0
vb_b, vb_c = bounce_badge(vb)
with col2:
    st.markdown(
        metric_card("Vertical Bounce", f"{vb:.1f}", "% of frame height", vb_b, vb_c),
        unsafe_allow_html=True,
    )

# Combined knee angle
ka    = k.combined_mean if k else 0.0
ka_b, ka_c = knee_badge(ka)
with col3:
    st.markdown(
        metric_card("Avg Knee Angle", f"{ka:.1f}", "degrees", ka_b, ka_c),
        unsafe_allow_html=True,
    )

# Frames
valid_pct = (report.valid_frames / report.total_frames * 100) if report.total_frames else 0
fp_b = "badge-good" if valid_pct > 80 else "badge-warn"
with col4:
    st.markdown(
        metric_card(
            "Pose Detection", f"{valid_pct:.0f}",
            f"% ({report.valid_frames} / {report.total_frames} frames)",
            f"{report.valid_frames} frames", fp_b,
        ),
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Knee detail table
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='section-heading'>🦵 Knee Angle Detail</div>",
            unsafe_allow_html=True)

kl, kr = st.columns(2)

def knee_detail(side: str, mean: float, mn: float, mx: float) -> str:
    warn_mean = mean > 170
    warn_max  = mx   > 170
    rows = [
        ("Mean",  f"{mean:.1f}°", "badge-danger" if warn_mean else "badge-good"),
        ("Min",   f"{mn:.1f}°",   "badge-good"),
        ("Max",   f"{mx:.1f}°",   "badge-danger" if warn_max  else "badge-warn"),
    ]
    html = f"<div style='background:#161b22; border:1px solid #21262d; border-radius:10px; padding:16px'>"
    html += f"<div style='font-size:0.82rem; font-weight:700; color:#e6edf3; margin-bottom:12px'>{side} Knee</div>"
    for label, val, cls in rows:
        html += (
            f"<div class='knee-row'>"
            f"<span style='color:#8b949e'>{label}</span>"
            f"<span class='badge {cls}'>{val}</span>"
            f"</div>"
        )
    html += "</div>"
    return html

with kl:
    if k:
        st.markdown(knee_detail("Left",  k.left_mean,  k.left_min,  k.left_max),
                    unsafe_allow_html=True)
with kr:
    if k:
        st.markdown(knee_detail("Right", k.right_mean, k.right_min, k.right_max),
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Full text report expander
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='section-heading'>📋 Full Report</div>",
            unsafe_allow_html=True)

from analyzer import build_report_text
with st.expander("View full analysis report", expanded=False):
    st.code(build_report_text(report), language=None)

report_path = os.path.join(
    OUTPUT_DIR,
    os.path.splitext(os.path.basename(st.session_state.csv_path))[0] + "_report.txt"
)
if os.path.isfile(report_path):
    with open(report_path) as f:
        dl_col, _ = st.columns([1, 4])
        with dl_col:
            st.download_button(
                "⬇ Download Report (.txt)",
                data=f.read(),
                file_name=os.path.basename(report_path),
                mime="text/plain",
                use_container_width=True,
            )
