"""
pose_engine.py
--------------
StrideFlow — Running Form Analyzer
Processes a running video using the MediaPipe Tasks PoseLandmarker (v0.10+).

For every frame, extracts (x, y, z) for Left & Right  →  Shoulder, Hip, Knee, Ankle.
Results are saved to a per-video CSV in /output.

Real-time overlay features
──────────────────────────
• Knee angle (°) displayed next to each knee joint
• Leg bones (Hip→Knee, Knee→Ankle) turn RED when angle > 170° (overstride warning)
  and GREEN when angle is within a healthy range
• HUD panel (top-left) shows live Cadence (SPM) and Vertical Bounce (%)

Usage:
    python src/pose_engine.py --video data/videos/<your_video.mp4>
    python src/pose_engine.py --video data/videos/<your_video.mp4> --no-display
    python src/pose_engine.py --video data/videos/<your_video.mp4> \\
        --model data/pose_landmarker_lite.task
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

# ── MediaPipe Tasks (v0.10+) ──────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
)
from mediapipe.tasks.python.vision import PoseLandmark

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL  = os.path.join(PROJECT_ROOT, "data",   "pose_landmarker_heavy.task")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "output")

# ── Landmarks of interest (for CSV export) ────────────────────────────────────
LANDMARKS_OF_INTEREST: dict[str, int] = {
    "left_shoulder":  PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": PoseLandmark.RIGHT_SHOULDER,
    "left_hip":       PoseLandmark.LEFT_HIP,
    "right_hip":      PoseLandmark.RIGHT_HIP,
    "left_knee":      PoseLandmark.LEFT_KNEE,
    "right_knee":     PoseLandmark.RIGHT_KNEE,
    "left_ankle":     PoseLandmark.LEFT_ANKLE,
    "right_ankle":    PoseLandmark.RIGHT_ANKLE,
}

CSV_HEADER = ["frame", "timestamp_s"]
for _n in LANDMARKS_OF_INTEREST:
    CSV_HEADER += [f"{_n}_x", f"{_n}_y", f"{_n}_z", f"{_n}_visibility"]

# ── Overlay constants ─────────────────────────────────────────────────────────
KNEE_WARN_THRESHOLD = 170.0      # degrees — above this = overt-stride warning
ROLLING_WINDOW_S    = 5.0        # seconds of history for cadence / oscillation
PEAK_MIN_DIST       = 8          # frames  — minimum gap between ankle-Y peaks
PEAK_MIN_PROM       = 0.012      # normalised units — minimum peak prominence

# Skeleton-wide connections list
_ALL_CONNECTIONS = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]

# Leg-specific connections that receive the warning colour
_LEFT_LEG_CONNECTIONS  = frozenset([
    (int(PoseLandmark.LEFT_HIP),  int(PoseLandmark.LEFT_KNEE)),
    (int(PoseLandmark.LEFT_KNEE), int(PoseLandmark.LEFT_ANKLE)),
])
_RIGHT_LEG_CONNECTIONS = frozenset([
    (int(PoseLandmark.RIGHT_HIP),  int(PoseLandmark.RIGHT_KNEE)),
    (int(PoseLandmark.RIGHT_KNEE), int(PoseLandmark.RIGHT_ANKLE)),
])

# Colours (BGR)
_COL_GREEN   = (0,  220,  80)
_COL_RED     = (0,   50, 230)
_COL_GREY    = (200, 200, 200)
_COL_WHITE   = (255, 255, 255)
_COL_BLACK   = (0,   0,   0)
_COL_HUD_BG  = (20,  20,  20)

_JOINT_COLS: dict[str, tuple] = {
    "shoulder": (0, 180, 255),
    "hip":      (255, 200,  0),
    "knee":     (0,  220, 100),
    "ankle":    (80,  80, 255),
}


# ══════════════════════════════════════════════════════════════════════════════
# Inline math helpers  (self-contained – no import from analyzer.py)
# ══════════════════════════════════════════════════════════════════════════════

def _lm_xyz(lm) -> np.ndarray:
    """Return (x, y, z) as a float64 array for a NormalizedLandmark."""
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def _knee_angle_deg(hip_lm, knee_lm, ankle_lm) -> float:
    """
    Angle at the knee vertex (degrees) using the 3-D dot product:

        v1 = hip   − knee  (knee → hip direction)
        v2 = ankle − knee  (knee → ankle direction)
        θ  = arccos( v1·v2 / |v1|·|v2| )
    """
    v1 = _lm_xyz(hip_lm)   - _lm_xyz(knee_lm)
    v2 = _lm_xyz(ankle_lm) - _lm_xyz(knee_lm)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_a)))


def _moving_avg(arr: np.ndarray, w: int = 5) -> np.ndarray:
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def _count_peaks(arr: np.ndarray) -> int:
    """Count local maxima with minimum distance and prominence constraints."""
    n = len(arr)
    if n < 2 * PEAK_MIN_DIST + 1:
        return 0
    peaks = []
    for i in range(PEAK_MIN_DIST, n - PEAK_MIN_DIST):
        window = arr[i - PEAK_MIN_DIST: i + PEAK_MIN_DIST + 1]
        if arr[i] != np.nanmax(window):
            continue
        if np.nanmax(window) - np.nanmin(window) >= PEAK_MIN_PROM:
            peaks.append(i)
    # de-duplicate clusters
    if not peaks:
        return 0
    deduped = [peaks[0]]
    for p in peaks[1:]:
        if p - deduped[-1] >= PEAK_MIN_DIST:
            deduped.append(p)
        elif arr[p] > arr[deduped[-1]]:
            deduped[-1] = p
    return len(deduped)


# ══════════════════════════════════════════════════════════════════════════════
# Rolling metrics state
# ══════════════════════════════════════════════════════════════════════════════

class RealtimeMetrics:
    """
    Maintains rolling buffers of recent ankle-Y and mid-hip-Y values and
    recomputes cadence + vertical oscillation every *update_every* frames.
    """

    def __init__(self, fps: float, window_s: float = ROLLING_WINDOW_S):
        self.fps          = fps
        self.max_buf      = max(16, int(fps * window_s))
        self.update_every = max(1, int(fps * 0.5))          # recompute every 0.5 s

        self._left_ank_y  : deque[float] = deque(maxlen=self.max_buf)
        self._right_ank_y : deque[float] = deque(maxlen=self.max_buf)
        self._mid_hip_y   : deque[float] = deque(maxlen=self.max_buf)
        self._frame_count : int   = 0

        # Publicly readable results
        self.cadence_spm         : float = 0.0
        self.vertical_bounce_pct : float = 0.0   # oscillation range as % of frame

    def push(self, landmarks: list) -> None:
        """Feed one frame of landmarks and recompute if interval elapsed."""
        L = int(PoseLandmark.LEFT_ANKLE)
        R = int(PoseLandmark.RIGHT_ANKLE)
        LH = int(PoseLandmark.LEFT_HIP)
        RH = int(PoseLandmark.RIGHT_HIP)

        lank = landmarks[L]
        rank = landmarks[R]
        lhip = landmarks[LH]
        rhip = landmarks[RH]

        vis_ok = lambda lm: lm.visibility >= 0.5

        if vis_ok(lank):
            self._left_ank_y.append(lank.y)
        if vis_ok(rank):
            self._right_ank_y.append(rank.y)
        if vis_ok(lhip) and vis_ok(rhip):
            self._mid_hip_y.append((lhip.y + rhip.y) / 2.0)

        self._frame_count += 1
        if self._frame_count % self.update_every == 0:
            self._recompute()

    def _recompute(self) -> None:
        """Recompute cadence and vertical oscillation from rolling buffers."""
        # ── Cadence ───────────────────────────────────────────
        l_arr = np.array(self._left_ank_y,  dtype=np.float64)
        r_arr = np.array(self._right_ank_y, dtype=np.float64)
        total_peaks = 0
        dur_frames  = 0
        for arr in (l_arr, r_arr):
            if len(arr) >= 2 * PEAK_MIN_DIST + 1:
                smooth = _moving_avg(arr, w=7)
                total_peaks += _count_peaks(smooth)
                dur_frames   = max(dur_frames, len(arr))

        duration_s = dur_frames / self.fps if self.fps > 0 else 0
        self.cadence_spm = (total_peaks / duration_s * 60.0) if duration_s > 0 else 0.0

        # ── Vertical oscillation ───────────────────────────────
        h_arr = np.array(self._mid_hip_y, dtype=np.float64)
        if len(h_arr) >= 9:
            smoothed = _moving_avg(h_arr, w=9)
            self.vertical_bounce_pct = float((np.max(smoothed) - np.min(smoothed)) * 100.0)
        else:
            self.vertical_bounce_pct = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _alpha_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    colour: tuple[int, int, int],
    alpha: float = 0.55,
) -> None:
    """Draw a filled rectangle with alpha blending (in-place)."""
    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
    x2, y2 = min(img.shape[1], pt2[0]), min(img.shape[0], pt2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    rect = np.full_like(roi, colour, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, roi, 1.0 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def _draw_hud(
    frame:   np.ndarray,
    metrics: RealtimeMetrics,
    l_angle: float,
    r_angle: float,
    frame_idx: int,
    timestamp_s: float,
) -> None:
    """
    Render the semi-transparent HUD panel in the top-left corner.

    Displays:
      • Frame / Timestamp
      • Cadence (SPM)  with traffic-light colour coding
      • Vertical Bounce (% frame height)
      • Left & Right knee angles with warning icon when overstriding
    """
    pad  = 12
    lh   = 26        # line height
    rows = 7         # number of text rows
    w    = 290
    h    = pad * 2 + rows * lh

    _alpha_rect(frame, (0, 0), (w, h), _COL_HUD_BG, alpha=0.60)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    small      = 0.52
    normal     = 0.60
    thick_thin = 1
    thick_bold = 2

    def put(text, row, colour=_COL_WHITE, scale=small, thickness=thick_thin):
        y = pad + row * lh + lh - 6
        cv2.putText(frame, text, (pad, y), font, scale, colour, thickness, cv2.LINE_AA)

    # Row 0 — header
    put("StrideFlow  Live Metrics", 0, colour=(0, 220, 140), scale=normal,
        thickness=thick_bold)

    # Row 1 — frame/time
    put(f"Frame {frame_idx:05d}   {timestamp_s:.2f} s", 1, colour=(180, 180, 180))

    # Row 2 — cadence  (green / amber / red banding)
    spm = metrics.cadence_spm
    if spm <= 0:
        cad_col = (160, 160, 160)
        cad_str = "Cadence:  --  SPM"
    elif spm < 160:
        cad_col = (0, 100, 255)   # red  → overstriding
        cad_str = f"Cadence:  {spm:.0f}  SPM  LOW"
    elif spm <= 180:
        cad_col = (0, 200, 80)    # green
        cad_str = f"Cadence:  {spm:.0f}  SPM"
    else:
        cad_col = (0, 220, 255)   # yellow → elite range
        cad_str = f"Cadence:  {spm:.0f}  SPM  ELITE"
    put(cad_str, 2, colour=cad_col, scale=small, thickness=thick_bold)

    # Row 3 — vertical bounce  (green < 8%, amber < 12%, red >= 12%)
    vb = metrics.vertical_bounce_pct
    if vb <= 0:
        vb_col = (160, 160, 160)
        vb_str = "V-Bounce: --  %"
    elif vb < 8:
        vb_col = (0, 200, 80)
        vb_str = f"V-Bounce: {vb:.1f} %  GOOD"
    elif vb < 12:
        vb_col = (0, 200, 255)
        vb_str = f"V-Bounce: {vb:.1f} %  OK"
    else:
        vb_col = (0, 80, 230)
        vb_str = f"V-Bounce: {vb:.1f} %  HIGH"
    put(vb_str, 3, colour=vb_col, scale=small, thickness=thick_bold)

    # Separator
    cv2.line(frame, (pad, pad + 4 * lh + 4), (w - pad, pad + 4 * lh + 4),
             (80, 80, 80), 1)

    # Row 5 — Left knee
    if not math.isnan(l_angle):
        warn_l = l_angle > KNEE_WARN_THRESHOLD
        l_col  = _COL_RED if warn_l else _COL_GREEN
        l_warn = "  ⚠ OVERSTRIDE" if warn_l else ""
        put(f"L Knee: {l_angle:5.1f} deg{l_warn}", 5, colour=l_col,
            scale=small, thickness=thick_bold)
    else:
        put("L Knee:  --", 5, colour=(120, 120, 120))

    # Row 6 — Right knee
    if not math.isnan(r_angle):
        warn_r = r_angle > KNEE_WARN_THRESHOLD
        r_col  = _COL_RED if warn_r else _COL_GREEN
        r_warn = "  ⚠ OVERSTRIDE" if warn_r else ""
        put(f"R Knee: {r_angle:5.1f} deg{r_warn}", 6, colour=r_col,
            scale=small, thickness=thick_bold)
    else:
        put("R Knee:  --", 6, colour=(120, 120, 120))


def draw_annotated_frame(
    frame:       np.ndarray,
    landmarks:   list,
    metrics:     RealtimeMetrics,
    l_angle:     float,
    r_angle:     float,
    frame_idx:   int,
    timestamp_s: float,
    total_frames: int,
) -> np.ndarray:
    """
    Compose the full annotated output frame:
      1. Full skeleton with leg bones colour-coded by knee angle
      2. Knee-angle labels next to each knee joint
      3. Highlighted key joints (circles)
      4. HUD panel top-left
      5. Progress bar bottom
    """
    out = frame.copy()
    h, w = out.shape[:2]

    warn_left  = (not math.isnan(l_angle)) and l_angle > KNEE_WARN_THRESHOLD
    warn_right = (not math.isnan(r_angle)) and r_angle > KNEE_WARN_THRESHOLD

    # ── 1. Skeleton connections ────────────────────────────────────────────
    for start_idx, end_idx in _ALL_CONNECTIONS:
        lm_s = landmarks[start_idx]
        lm_e = landmarks[end_idx]
        if lm_s.visibility < 0.4 or lm_e.visibility < 0.4:
            continue

        ps = (int(lm_s.x * w), int(lm_s.y * h))
        pe = (int(lm_e.x * w), int(lm_e.y * h))

        conn = (start_idx, end_idx)

        if conn in _LEFT_LEG_CONNECTIONS or (end_idx, start_idx) in _LEFT_LEG_CONNECTIONS:
            colour    = _COL_RED   if warn_left  else _COL_GREEN
            thickness = 3
        elif conn in _RIGHT_LEG_CONNECTIONS or (end_idx, start_idx) in _RIGHT_LEG_CONNECTIONS:
            colour    = _COL_RED   if warn_right else _COL_GREEN
            thickness = 3
        else:
            colour    = _COL_GREY
            thickness = 2

        cv2.line(out, ps, pe, colour, thickness, cv2.LINE_AA)

    # ── 2. Key-joint highlight circles ────────────────────────────────────
    for name, lm_enum in LANDMARKS_OF_INTEREST.items():
        lm = landmarks[int(lm_enum)]
        if lm.visibility < 0.4:
            continue
        px, py  = int(lm.x * w), int(lm.y * h)
        jtype   = name.split("_")[1]
        jcolour = _JOINT_COLS.get(jtype, _COL_WHITE)
        cv2.circle(out, (px, py), 9,  jcolour,   -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), 11, _COL_WHITE,  2, cv2.LINE_AA)

    # ── 3. Knee angle labels next to each knee joint ───────────────────────
    for side, lm_enum, angle, warn in [
        ("left",  PoseLandmark.LEFT_KNEE,  l_angle, warn_left),
        ("right", PoseLandmark.RIGHT_KNEE, r_angle, warn_right),
    ]:
        lm = landmarks[int(lm_enum)]
        if lm.visibility < 0.4 or math.isnan(angle):
            continue

        px = int(lm.x * w)
        py = int(lm.y * h)

        label_text = f"{angle:.1f}{chr(176)}"   # e.g. "142.3°"
        font_scale = 0.60
        font_thick = 2
        lbl_colour = _COL_RED if warn else _COL_GREEN

        # Offset: push right for left knee, left for right knee
        offset_x = 14 if side == "left" else -80
        offset_y = -14

        tx, ty = px + offset_x, py + offset_y

        # Shadow for contrast
        cv2.putText(out, label_text, (tx + 1, ty + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, _COL_BLACK, font_thick + 1,
                    cv2.LINE_AA)
        cv2.putText(out, label_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, lbl_colour, font_thick,
                    cv2.LINE_AA)

    # ── 4. HUD panel ──────────────────────────────────────────────────────
    _draw_hud(out, metrics, l_angle, r_angle, frame_idx, timestamp_s)

    # ── 5. Progress bar (bottom edge) ─────────────────────────────────────
    progress = frame_idx / max(total_frames - 1, 1)
    bar_w    = int(w * progress)
    cv2.rectangle(out, (0, h - 5), (bar_w, h), (0, 255, 160), -1)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Core processing loop
# ══════════════════════════════════════════════════════════════════════════════

def process_video(
    video_path:  str,
    model_path:  str,
    output_dir:  str,
    display:     bool = True,
) -> str:
    """
    Run PoseLandmarker on every frame, overlay live metrics, and save CSV.

    Returns:
        Absolute path to the generated CSV file.
    """
    # ── Validate ─────────────────────────────────────────────────────────
    if not os.path.isfile(video_path):
        sys.exit(f"[ERROR] Video not found: {video_path}")
    if not os.path.isfile(model_path):
        sys.exit(
            f"[ERROR] Model not found: {model_path}\n"
            "Download with:\n"
            "  curl -L -o data/pose_landmarker_heavy.task \\\n"
            "    https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        )

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_dir, f"{basename}_landmarks.csv")

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'─'*62}")
    print(f"  StrideFlow — Pose Engine  (with live metric overlay)")
    print(f"{'─'*62}")
    print(f"  Input  : {video_path}")
    print(f"  Model  : {model_path}")
    print(f"  Output : {csv_path}")
    print(f"  FPS    : {fps:.1f}  |  Frames: {total_frames}  |  {width}×{height}")
    print(f"{'─'*62}\n")

    # ── MediaPipe landmarker ──────────────────────────────────────────────
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options   = PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    # ── Rolling metrics state ─────────────────────────────────────────────
    metrics    = RealtimeMetrics(fps=fps)
    frame_idx  = 0
    start_time = time.time()

    with PoseLandmarker.create_from_options(options) as landmarker, \
         open(csv_path, "w", newline="") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)

        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            timestamp_s  = round(frame_idx / fps, 4)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # ── Build CSV row ─────────────────────────────────────────────
            row = [frame_idx, timestamp_s]
            pose_list = result.pose_landmarks

            l_angle = float("nan")
            r_angle = float("nan")

            if pose_list:
                lms = pose_list[0]

                # Per-frame knee angles
                l_angle = _knee_angle_deg(
                    lms[int(PoseLandmark.LEFT_HIP)],
                    lms[int(PoseLandmark.LEFT_KNEE)],
                    lms[int(PoseLandmark.LEFT_ANKLE)],
                )
                r_angle = _knee_angle_deg(
                    lms[int(PoseLandmark.RIGHT_HIP)],
                    lms[int(PoseLandmark.RIGHT_KNEE)],
                    lms[int(PoseLandmark.RIGHT_ANKLE)],
                )

                # Feed rolling metrics
                metrics.push(lms)

                # CSV columns
                for _, lm_enum in LANDMARKS_OF_INTEREST.items():
                    lm = lms[int(lm_enum)]
                    row += [
                        round(lm.x,          6),
                        round(lm.y,          6),
                        round(lm.z,          6),
                        round(lm.visibility, 4),
                    ]
            else:
                row += [np.nan] * (len(LANDMARKS_OF_INTEREST) * 4)

            writer.writerow(row)

            # ── Overlay & display ─────────────────────────────────────────
            if display:
                if pose_list:
                    annotated = draw_annotated_frame(
                        frame        = bgr_frame,
                        landmarks    = pose_list[0],
                        metrics      = metrics,
                        l_angle      = l_angle,
                        r_angle      = r_angle,
                        frame_idx    = frame_idx,
                        timestamp_s  = timestamp_s,
                        total_frames = total_frames,
                    )
                else:
                    # No pose detected — show plain frame + "no pose" notice
                    annotated = bgr_frame.copy()
                    _alpha_rect(annotated, (0, 0), (290, 50), _COL_HUD_BG, 0.6)
                    cv2.putText(annotated, "No pose detected", (12, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2,
                                cv2.LINE_AA)

                cv2.imshow("StrideFlow — Pose Engine  [Q / Esc to quit]", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\n[INFO] User quit early.")
                    break

            frame_idx += 1

            # Console progress every 60 frames
            if frame_idx % 60 == 0:
                elapsed  = time.time() - start_time
                pct      = (frame_idx / total_frames * 100) if total_frames else 0
                fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                print(
                    f"  [{pct:5.1f}%]  Frame {frame_idx:>6}/{total_frames}"
                    f"  {fps_proc:.1f} fps  ({elapsed:.1f}s elapsed)"
                    f"  cadence={metrics.cadence_spm:.0f} SPM"
                    f"  bounce={metrics.vertical_bounce_pct:.1f}%"
                )

    cap.release()
    if display:
        cv2.destroyAllWindows()

    elapsed_total = time.time() - start_time
    print(f"\n{'─'*62}")
    print(f"  Done!  Processed {frame_idx} frames in {elapsed_total:.1f}s")
    print(f"  CSV saved → {csv_path}")
    print(f"{'─'*62}\n")

    return csv_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="StrideFlow Pose Engine — extract joint landmarks with live overlay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the input video (absolute or relative to project root).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Path to the MediaPipe PoseLandmarker .task bundle.",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT,
        help="Directory for the output CSV.",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable the live video window (headless / server mode).",
    )

    args = parser.parse_args()

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    process_video(
        video_path = resolve(args.video),
        model_path = resolve(args.model),
        output_dir = resolve(args.output_dir),
        display    = not args.no_display,
    )
