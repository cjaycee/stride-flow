"""
analyzer.py
-----------
StrideFlow — Running Form Analyzer
Reads the landmark CSV produced by pose_engine.py and computes
three key biomechanical metrics:

    1. Knee Angle          — Hip ➜ Knee ➜ Ankle angle via dot product (degrees)
    2. Cadence             — Steps per minute via ankle-Y peak detection
    3. Vertical Oscillation — Mid-hip vertical range during the gait cycle (normalised units)

A formatted summary report is printed to the console and saved to
output/<basename>_report.txt.

Usage:
    python src/analyzer.py --csv output/<video>_landmarks.csv
    python src/analyzer.py --csv output/<video>_landmarks.csv --fps 30
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "output")

# ── Visibility threshold ───────────────────────────────────────────────────────
# Rows where any required landmark has visibility below this are excluded.
VIS_THRESHOLD = 0.50

# ── Cadence tuning ─────────────────────────────────────────────────────────────
# Minimum distance (frames) between consecutive ankle Y peaks.
# At 30 fps a full single-leg step cycle takes ≥ 9 frames (≤ 200 SPM).
PEAK_MIN_DISTANCE_FRAMES = 8

# Minimum signal rise required for a peak to be counted (normalised units 0–1).
PEAK_MIN_PROMINENCE = 0.012


# ══════════════════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class KneeAngleResult:
    left_angles_deg:  np.ndarray        # per-frame angles  (left side)
    right_angles_deg: np.ndarray        # per-frame angles  (right side)
    left_mean:  float = 0.0
    left_min:   float = 0.0
    left_max:   float = 0.0
    right_mean: float = 0.0
    right_min:  float = 0.0
    right_max:  float = 0.0
    combined_mean: float = 0.0


@dataclass
class CadenceResult:
    left_peaks:  np.ndarray            # frame indices of left ankle peaks
    right_peaks: np.ndarray            # frame indices of right ankle peaks
    total_steps: int = 0
    duration_s:  float = 0.0
    cadence_spm: float = 0.0           # steps per minute


@dataclass
class VerticalOscillationResult:
    mid_hip_y:  np.ndarray             # per-frame mid-hip Y (normalised, 0=top)
    range_norm: float = 0.0            # max − min in normalised units
    mean_norm:  float = 0.0
    std_norm:   float = 0.0
    # Convenience: range expressed as % of frame height (multiply by height px)
    range_pct:  float = 0.0            # range_norm × 100


@dataclass
class AnalysisReport:
    source_csv:    str = ""
    total_frames:  int = 0
    valid_frames:  int = 0
    fps:           float = 30.0
    knee:          Optional[KneeAngleResult] = None
    cadence:       Optional[CadenceResult]  = None
    oscillation:   Optional[VerticalOscillationResult] = None


# ══════════════════════════════════════════════════════════════════════════════
# 1. CSV loader
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(csv_path: str) -> tuple[np.ndarray, list[str]]:
    """
    Load the landmarks CSV produced by pose_engine.py.

    Returns:
        data    – float32 array  shape (N, num_columns)
        columns – list of column name strings
    """
    if not os.path.isfile(csv_path):
        sys.exit(f"[ERROR] CSV not found: {csv_path}")

    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")

    try:
        data = np.genfromtxt(
            csv_path,
            delimiter=",",
            skip_header=1,
            dtype=np.float64,
            filling_values=np.nan,
        )
    except Exception as exc:
        sys.exit(f"[ERROR] Failed to parse CSV: {exc}")

    if data.ndim == 1:
        data = data[np.newaxis, :]   # single-row file edge case

    print(f"[INFO] Loaded {len(data)} frames × {len(header)} columns from:\n"
          f"       {csv_path}\n")
    return data, header


def _col(header: list[str], name: str) -> int:
    """Return the column index for *name*, exiting clearly on failure."""
    try:
        return header.index(name)
    except ValueError:
        sys.exit(
            f"[ERROR] Expected column '{name}' not found in CSV.\n"
            "        Make sure this CSV was produced by pose_engine.py."
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Knee Angle  (dot product of Hip→Knee and Ankle→Knee vectors)
# ══════════════════════════════════════════════════════════════════════════════

def _angle_between_three_points(
    A: np.ndarray,   # (N, 3)  Hip
    B: np.ndarray,   # (N, 3)  Knee  ← vertex
    C: np.ndarray,   # (N, 3)  Ankle
) -> np.ndarray:
    """
    Return the angle at vertex B in degrees for each of N rows.

        v1 = A − B   (knee → hip)
        v2 = C − B   (knee → ankle)
        θ  = arccos( v1·v2 / (|v1|·|v2|) )

    Rows with degenerate vectors (zero-length) are returned as NaN.
    """
    v1 = A - B   # (N, 3)
    v2 = C - B   # (N, 3)

    dot   = np.einsum("ij,ij->i", v1, v2)                # (N,)
    norm1 = np.linalg.norm(v1, axis=1)                    # (N,)
    norm2 = np.linalg.norm(v2, axis=1)                    # (N,)
    denom = norm1 * norm2

    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = np.where(denom > 1e-9, dot / denom, np.nan)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)         # numerical safety
        angles    = np.degrees(np.arccos(cos_theta))       # (N,)

    return angles


def compute_knee_angles(
    data: np.ndarray,
    header: list[str],
) -> KneeAngleResult:
    """
    Compute left and right knee angles per frame.

    Uses 3-D (x, y, z) landmarks so depth is accounted for, giving a more
    accurate angle than 2-D projection alone.
    """

    def _xyz_vis(side: str, joint: str):
        """Extract (N,3) coords and (N,) visibility for a landmark."""
        prefix = f"{side}_{joint}"
        xi = _col(header, f"{prefix}_x")
        yi = _col(header, f"{prefix}_y")
        zi = _col(header, f"{prefix}_z")
        vi = _col(header, f"{prefix}_visibility")
        return data[:, [xi, yi, zi]], data[:, vi]

    left_hip_xyz,   lhip_vis  = _xyz_vis("left",  "hip")
    left_knee_xyz,  lkne_vis  = _xyz_vis("left",  "knee")
    left_ankle_xyz, lank_vis  = _xyz_vis("left",  "ankle")

    right_hip_xyz,   rhip_vis = _xyz_vis("right", "hip")
    right_knee_xyz,  rkne_vis = _xyz_vis("right", "knee")
    right_ankle_xyz, rank_vis = _xyz_vis("right", "ankle")

    # Angles – all frames (NaN where data is missing)
    left_angles  = _angle_between_three_points(left_hip_xyz,  left_knee_xyz,  left_ankle_xyz)
    right_angles = _angle_between_three_points(right_hip_xyz, right_knee_xyz, right_ankle_xyz)

    # Mask out low-visibility frames
    left_mask  = (lhip_vis  >= VIS_THRESHOLD) & \
                 (lkne_vis  >= VIS_THRESHOLD) & \
                 (lank_vis  >= VIS_THRESHOLD)
    right_mask = (rhip_vis  >= VIS_THRESHOLD) & \
                 (rkne_vis  >= VIS_THRESHOLD) & \
                 (rank_vis  >= VIS_THRESHOLD)

    # Also exclude NaN rows introduced by undetected frames in the CSV
    left_mask  &= ~np.isnan(left_angles)
    right_mask &= ~np.isnan(right_angles)

    left_valid  = left_angles[left_mask]
    right_valid = right_angles[right_mask]

    res = KneeAngleResult(
        left_angles_deg=left_angles,
        right_angles_deg=right_angles,
    )
    if len(left_valid):
        res.left_mean = float(np.mean(left_valid))
        res.left_min  = float(np.min(left_valid))
        res.left_max  = float(np.max(left_valid))
    if len(right_valid):
        res.right_mean = float(np.mean(right_valid))
        res.right_min  = float(np.min(right_valid))
        res.right_max  = float(np.max(right_valid))

    both = np.concatenate([left_valid, right_valid])
    if len(both):
        res.combined_mean = float(np.mean(both))

    return res


# ══════════════════════════════════════════════════════════════════════════════
# 3. Cadence  (ankle-Y peak detection)
# ══════════════════════════════════════════════════════════════════════════════

def _smooth(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Simple causal moving-average smoother.
    Handles shorter signals by truncating the kernel.
    """
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def _find_peaks(
    signal: np.ndarray,
    min_distance: int = PEAK_MIN_DISTANCE_FRAMES,
    min_prominence: float = PEAK_MIN_PROMINENCE,
) -> np.ndarray:
    """
    Identify local maxima in *signal* without scipy.

    A sample i is a peak if:
      - it is the maximum within the window [i-min_distance, i+min_distance]
      - its value exceeds the local minimum in that window by at least min_prominence
    """
    n = len(signal)
    peaks = []
    for i in range(min_distance, n - min_distance):
        window = signal[i - min_distance : i + min_distance + 1]
        if signal[i] != np.nanmax(window):
            continue
        local_range = np.nanmax(window) - np.nanmin(window)
        if local_range >= min_prominence:
            peaks.append(i)

    # Enforce minimum spacing: keep only the best peak within each cluster
    if not peaks:
        return np.array([], dtype=int)

    deduplicated = [peaks[0]]
    for p in peaks[1:]:
        if p - deduplicated[-1] >= min_distance:
            deduplicated.append(p)
        elif signal[p] > signal[deduplicated[-1]]:
            deduplicated[-1] = p   # replace with the higher peak in the cluster

    return np.array(deduplicated, dtype=int)


def compute_cadence(
    data: np.ndarray,
    header: list[str],
    fps: float,
) -> CadenceResult:
    """
    Estimate running cadence (steps per minute).

    Strategy:
        In normalised image coordinates Y increases downward (0 = top, 1 = bottom).
        At foot-strike (stance phase) the ankle is at its lowest pixel position,
        producing a *maximum* in Y.  We count these maxima on each ankle and
        combine them into a total step count.

    Steps per minute = (total_peaks / duration_seconds) × 60
    """
    li = _col(header, "left_ankle_y")
    ri = _col(header, "right_ankle_y")
    lv = _col(header, "left_ankle_visibility")
    rv = _col(header, "right_ankle_visibility")
    ts = _col(header, "timestamp_s")

    left_y   = data[:, li].copy()
    right_y  = data[:, ri].copy()
    left_vis = data[:, lv]
    right_vis= data[:, rv]

    # Replace low-visibility / NaN frames with linear interpolation
    for arr, vis in [(left_y, left_vis), (right_y, right_vis)]:
        bad = (vis < VIS_THRESHOLD) | np.isnan(arr)
        if bad.all():
            arr[:] = np.nan
            continue
        x_all   = np.arange(len(arr))
        x_good  = x_all[~bad]
        y_good  = arr[~bad]
        arr[bad] = np.interp(x_all[bad], x_good, y_good)

    # Smooth to reduce jitter
    left_smooth  = _smooth(left_y,  window=7)
    right_smooth = _smooth(right_y, window=7)

    left_peaks  = _find_peaks(left_smooth)
    right_peaks = _find_peaks(right_smooth)

    timestamps  = data[:, ts]
    valid_ts    = timestamps[~np.isnan(timestamps)]
    duration_s  = float(valid_ts[-1] - valid_ts[0]) if len(valid_ts) > 1 else 0.0

    total_steps = len(left_peaks) + len(right_peaks)
    cadence_spm = (total_steps / duration_s * 60.0) if duration_s > 0 else 0.0

    return CadenceResult(
        left_peaks=left_peaks,
        right_peaks=right_peaks,
        total_steps=total_steps,
        duration_s=duration_s,
        cadence_spm=cadence_spm,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Vertical Oscillation  (mid-hip Y range)
# ══════════════════════════════════════════════════════════════════════════════

def compute_vertical_oscillation(
    data: np.ndarray,
    header: list[str],
) -> VerticalOscillationResult:
    """
    Measure vertical oscillation from the Mid-Hip point.

    Mid-Hip Y = (left_hip_y + right_hip_y) / 2   (normalised 0–1)

    Because Y increases downward:
        • A *lower* Y value → hip is higher on screen → running tall
        • A *higher* Y value → hip is lower on screen → crouching

    Vertical oscillation = max(mid_hip_y) − min(mid_hip_y)
    A smaller range indicates less wasted vertical energy.
    """
    lhy_i = _col(header, "left_hip_y")
    rhy_i = _col(header, "right_hip_y")
    lhv_i = _col(header, "left_hip_visibility")
    rhv_i = _col(header, "right_hip_visibility")

    left_hip_y   = data[:, lhy_i]
    right_hip_y  = data[:, rhy_i]
    left_hip_vis = data[:, lhv_i]
    right_hip_vis= data[:, rhv_i]

    # Build validity mask
    valid = (
        (left_hip_vis  >= VIS_THRESHOLD) &
        (right_hip_vis >= VIS_THRESHOLD) &
        (~np.isnan(left_hip_y)) &
        (~np.isnan(right_hip_y))
    )

    mid_hip_y = (left_hip_y + right_hip_y) / 2.0
    mid_hip_valid = mid_hip_y[valid]

    res = VerticalOscillationResult(mid_hip_y=mid_hip_y)
    if len(mid_hip_valid) < 2:
        return res

    # Smooth before measuring range to reduce landmark jitter
    smoothed   = _smooth(mid_hip_valid, window=9)
    osc_range  = float(np.max(smoothed) - np.min(smoothed))

    res.range_norm = osc_range
    res.mean_norm  = float(np.mean(smoothed))
    res.std_norm   = float(np.std(smoothed))
    res.range_pct  = osc_range * 100.0

    return res


# ══════════════════════════════════════════════════════════════════════════════
# 5. Report printer
# ══════════════════════════════════════════════════════════════════════════════

_W = 64   # report width

def _line(char: str = "─") -> str:
    return char * _W

def _title(text: str) -> str:
    pad = (_W - len(text) - 2) // 2
    return f"{'─'*pad}  {text}  {'─'*(_W - pad - len(text) - 2)}"

def _row(label: str, value: str, unit: str = "") -> str:
    unit_str = f"  {unit}" if unit else ""
    dots = "." * max(0, _W - len(label) - len(value) - len(unit_str) - 2)
    return f"  {label} {dots} {value}{unit_str}"


def build_report_text(report: AnalysisReport) -> str:
    lines = []
    banner = "  StrideFlow — Biomechanical Analysis Report"
    lines += [
        _line("═"),
        banner,
        f"  Generated : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        f"  Source    : {os.path.basename(report.source_csv)}",
        f"  Frames    : {report.valid_frames} valid / {report.total_frames} total"
        f"  ({report.fps:.1f} fps)",
        _line("═"),
        "",
    ]

    # ── Knee Angle ────────────────────────────────────────────────────────────
    lines += [_title("① KNEE ANGLE  (Hip ➜ Knee ➜ Ankle)"), ""]
    k = report.knee
    if k:
        lines += [
            _row("Left  – Mean",  f"{k.left_mean:.1f}",  "°"),
            _row("Left  – Min",   f"{k.left_min:.1f}",   "°"),
            _row("Left  – Max",   f"{k.left_max:.1f}",   "°"),
            "",
            _row("Right – Mean",  f"{k.right_mean:.1f}", "°"),
            _row("Right – Min",   f"{k.right_min:.1f}",  "°"),
            _row("Right – Max",   f"{k.right_max:.1f}",  "°"),
            "",
            _row("Combined Mean", f"{k.combined_mean:.1f}", "°"),
            "",
            "  Interpretation:",
            "   • Max angle near 170–180° = full knee extension (efficient stride)",
            "   • Min angle near 90–110°  = strong knee drive (propulsion)",
        ]
    else:
        lines += ["  ⚠  Insufficient data to compute knee angles."]

    lines += ["", _line(), ""]

    # ── Cadence ───────────────────────────────────────────────────────────────
    lines += [_title("② CADENCE  (ankle-Y peak detection)"), ""]
    c = report.cadence
    if c:
        lines += [
            _row("Duration",           f"{c.duration_s:.1f}",   "s"),
            _row("Left ankle steps",   f"{len(c.left_peaks)}",  "peaks"),
            _row("Right ankle steps",  f"{len(c.right_peaks)}", "peaks"),
            _row("Total steps counted",f"{c.total_steps}",      ""),
            "",
            _row("Cadence",            f"{c.cadence_spm:.1f}",  "SPM"),
            "",
            "  Interpretation:",
            "   • < 160 SPM — overstriding, consider shorter/quicker steps",
            "   • 160–180 SPM — typical recreational runner range",
            "   • > 180 SPM — elite / efficient cadence",
        ]
    else:
        lines += ["  ⚠  Insufficient data to compute cadence."]

    lines += ["", _line(), ""]

    # ── Vertical Oscillation ─────────────────────────────────────────────────
    lines += [_title("③ VERTICAL OSCILLATION  (mid-hip Y range)"), ""]
    v = report.oscillation
    if v:
        lines += [
            _row("Range (normalised)",  f"{v.range_norm:.4f}", "units  (0–1 scale)"),
            _row("Range (% of frame)",  f"{v.range_pct:.2f}",  "%"),
            _row("Mean hip height (Y)", f"{v.mean_norm:.4f}",  "norm."),
            _row("Std deviation",       f"{v.std_norm:.4f}",   "units"),
            "",
            "  Interpretation:",
            "   • range_norm × frame_height_px  =  oscillation in pixels",
            "   • Smaller range → less vertical bounce → more efficient running",
            "   • Elite runners oscillate ≈ 6–8 cm (mid-range for 720p video)",
        ]
    else:
        lines += ["  ⚠  Insufficient data to compute vertical oscillation."]

    lines += ["", _line("═"), ""]
    return "\n".join(lines)


def print_report(report: AnalysisReport, save_path: Optional[str] = None) -> None:
    text = build_report_text(report)
    print(text)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(text)
        print(f"  Report saved → {save_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def analyze(
    csv_path: str,
    fps: float = 30.0,
    output_dir: str = DEFAULT_OUTPUT,
    save_report: bool = True,
) -> AnalysisReport:
    """
    Full analysis pipeline.

    Args:
        csv_path:    Path to the landmark CSV from pose_engine.py.
        fps:         Frame rate of the original video (used for cadence timing).
        output_dir:  Where to write the text report.
        save_report: If True, also write the report to a .txt file.

    Returns:
        A populated AnalysisReport dataclass.
    """
    data, header = load_csv(csv_path)

    # Count valid (non-NaN) frames
    valid_mask   = ~np.all(np.isnan(data[:, 2:]), axis=1)  # skip frame/timestamp cols
    total_frames = len(data)
    valid_frames = int(np.sum(valid_mask))

    report = AnalysisReport(
        source_csv   = csv_path,
        total_frames = total_frames,
        valid_frames = valid_frames,
        fps          = fps,
    )

    print(f"[INFO] Computing knee angles...")
    report.knee = compute_knee_angles(data, header)

    print(f"[INFO] Computing cadence...")
    report.cadence = compute_cadence(data, header, fps)

    print(f"[INFO] Computing vertical oscillation...")
    report.oscillation = compute_vertical_oscillation(data, header)

    # Build report path
    report_path = None
    if save_report:
        basename   = os.path.splitext(os.path.basename(csv_path))[0]
        report_path = os.path.join(output_dir, f"{basename}_report.txt")

    print()
    print_report(report, save_path=report_path)

    return report


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="StrideFlow Analyzer — compute Knee Angle, Cadence & Vertical Oscillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the landmarks CSV (absolute or relative to project root).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate of the original video (needed for accurate cadence).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help="Directory where the text report is saved.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print report to console only; do not write a report file.",
    )

    args = parser.parse_args()

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    analyze(
        csv_path   = resolve(args.csv),
        fps        = args.fps,
        output_dir = resolve(args.output_dir),
        save_report= not args.no_save,
    )
