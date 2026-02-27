"""
app.py — StrideFlow Streamlit Dashboard
=======================================
Run with:
    streamlit run app.py
"""

import os
import sys
import tempfile

import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "videos")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "data", "pose_landmarker_heavy.task")

os.makedirs(DATA_DIR,  exist_ok=True)
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
/* ── Base theme tweaks ── */
[data-testid="stAppViewContainer"] {
    background: #0e1117;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1c2230 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 8px;
}
.metric-card .value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-card .unit {
    font-size: 0.8rem;
    color: #8b949e;
}
.metric-card .badge {
    display: inline-block;
    margin-top: 10px;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-good    { background: #0d4a1f; color: #3fb950; border: 1px solid #238636; }
.badge-warn    { background: #4a2a0d; color: #d29922; border: 1px solid #9e6a03; }
.badge-danger  { background: #4a0d12; color: #f85149; border: 1px solid #da3633; }
.badge-neutral { background: #21262d; color: #8b949e; border: 1px solid #30363d; }

/* ── Section headings ── */
.section-heading {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #58a6ff;
    margin: 24px 0 14px;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
}

/* ── Status pills ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 99px;
    font-size: 0.82rem;
    font-weight: 600;
}
.pill-ready   { background: #0d4a1f; color: #3fb950; border: 1px solid #238636; }
.pill-waiting { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
.pill-running { background: #0d2d4a; color: #58a6ff; border: 1px solid #1f6feb; }

/* ── Upload zone styling ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #30363d !important;
    border-radius: 12px !important;
    background: #161b22 !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #58a6ff !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
defaults = {
    "video_path":   None,    # path to the saved upload
    "csv_path":     None,    # path to the landmarks CSV
    "report":       None,    # AnalysisReport dataclass
    "df_landmarks": None,    # raw numpy array from CSV
    "df_header":    None,    # column names
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
    <div style='text-align:center; padding: 8px 0 20px'>
        <div style='font-size:2.4rem'>🏃</div>
        <div style='font-size:1.15rem; font-weight:700; color:#e6edf3;
                    letter-spacing:0.04em'>StrideFlow</div>
        <div style='font-size:0.75rem; color:#8b949e; margin-top:2px'>
            Running Form Analyzer
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("**📂 Video Input**")
    uploaded = st.file_uploader(
        label="Drop a running video here",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed",
        key="uploader",
    )

    st.markdown("**⚙️ Settings**")
    fps_override = st.number_input(
        "Video FPS",
        min_value=1.0,
        max_value=240.0,
        value=30.0,
        step=1.0,
        help="Must match the actual recording frame rate for accurate cadence.",
    )

    model_ok = os.path.isfile(MODEL_PATH)
    if not model_ok:
        st.warning(
            "⚠️ Pose model not found.\n\n"
            "Run this in your terminal to download it:\n\n"
            "```bash\ncurl -L -o data/pose_landmarker_heavy.task \\\n"
            "  https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task\n```"
        )

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem; color:#484f58; text-align:center'>"
        "MediaPipe PoseLandmarker · Heavy model<br>"
        "Tracks Hip · Knee · Ankle · Shoulder</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Handle upload → save to data/videos/
# ══════════════════════════════════════════════════════════════════════════════

if uploaded is not None:
    save_path = os.path.join(DATA_DIR, uploaded.name)
    if st.session_state.video_path != save_path:
        # New file uploaded — reset all downstream state
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.video_path   = save_path
        st.session_state.csv_path     = None
        st.session_state.report       = None
        st.session_state.df_landmarks = None
        st.session_state.df_header    = None
        st.session_state.processing   = False


# ══════════════════════════════════════════════════════════════════════════════
# Main area — header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='font-size:1.9rem; font-weight:800; color:#e6edf3;
           letter-spacing:-0.01em; margin-bottom:2px'>
    🏃 StrideFlow <span style='color:#58a6ff'>Dashboard</span>
</h1>
<p style='color:#8b949e; font-size:0.9rem; margin-top:0;'>
    Upload a running video · Analyze biomechanics · View real-time metrics
</p>
""", unsafe_allow_html=True)

st.divider()

# ── Status row ─────────────────────────────────────────────────────────────────
col_s1, col_s2, col_s3, _ = st.columns([1.6, 1.6, 1.6, 4])

with col_s1:
    if st.session_state.video_path:
        name = os.path.basename(st.session_state.video_path)
        st.markdown(
            f"<div class='status-pill pill-ready'>✅ {name[:22]}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='status-pill pill-waiting'>📂 No video loaded</div>",
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
            "<div class='status-pill pill-waiting'>⏸ Awaiting analysis</div>",
            unsafe_allow_html=True,
        )

# ── Placeholder sections when nothing is loaded ───────────────────────────────
if st.session_state.video_path is None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "👈 **Upload a video** in the sidebar to get started.\n\n"
        "Supported formats: MP4, MOV, AVI, MKV"
    )
    st.stop()

# ── Run button (shown after upload, before analysis) ──────────────────────────
if st.session_state.csv_path is None:
    st.markdown("<br>", unsafe_allow_html=True)
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_clicked = st.button(
            "▶  Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=not model_ok,
        )
    if run_clicked:
        st.session_state.processing = True
        st.rerun()

# ── Placeholder: analysis not yet run ─────────────────────────────────────────
if st.session_state.csv_path is None and not st.session_state.processing:
    st.markdown(
        "<p style='color:#8b949e; margin-top:12px'>"
        "Video loaded — click <b>▶ Run Analysis</b> to extract pose data.</p>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Processing state placeholder ──────────────────────────────────────────────
if st.session_state.processing and st.session_state.csv_path is None:
    with st.spinner("🔍 Extracting pose landmarks… this may take a minute."):
        st.info(
            "Analysis pipeline will be wired up in the next commit.\n\n"
            "*(Scaffold only — pipeline integration coming next.)*"
        )
    st.stop()
