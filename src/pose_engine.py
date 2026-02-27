"""
pose_engine.py
--------------
StrideFlow — Running Form Analyzer
Processes a running video using the MediaPipe Tasks PoseLandmarker (v0.10+).

For every frame, it extracts (x, y, z) coordinates for:
    Left & Right  →  Shoulder, Hip, Knee, Ankle

Results are saved to a per-video CSV in /output.
The video is displayed in real-time with the skeletal overlay.

Usage:
    python src/pose_engine.py --video data/videos/<your_video.mp4>
    python src/pose_engine.py --video data/videos/<your_video.mp4> --no-display
    python src/pose_engine.py --video data/videos/<your_video.mp4> --model data/pose_landmarker_lite.task
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

# ── MediaPipe Tasks (v0.10+) ─────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
    drawing_utils,
)
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec
from mediapipe.tasks.python.vision import PoseLandmark

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "data", "pose_landmarker_heavy.task")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "output")

# ── Landmarks of interest ─────────────────────────────────────────────────────
# Maps a human-readable label → PoseLandmark index
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

# CSV header columns
CSV_HEADER = ["frame", "timestamp_s"]
for _name in LANDMARKS_OF_INTEREST:
    CSV_HEADER += [
        f"{_name}_x",
        f"{_name}_y",
        f"{_name}_z",
        f"{_name}_visibility",
    ]

# ── Drawing helpers ───────────────────────────────────────────────────────────
# All skeleton connections provided by MediaPipe Tasks
_POSE_CONNECTIONS = [
    (c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS
]

_JOINT_COLOURS: dict[str, tuple[int, int, int]] = {
    "shoulder": (0, 180, 255),   # amber-orange
    "hip":      (255, 200, 0),   # cyan
    "knee":     (0, 220, 100),   # green
    "ankle":    (80,  80, 255),  # blue
}


def draw_skeleton_overlay(
    frame: np.ndarray,
    landmarks: list,          # list[NormalizedLandmark]
    image_h: int,
    image_w: int,
) -> np.ndarray:
    """Draw the full skeleton plus highlighted key joints onto *frame*."""
    annotated = frame.copy()

    # ── Draw all skeleton connections ─────────────────────────────────────
    for start_idx, end_idx in _POSE_CONNECTIONS:
        lm_s = landmarks[start_idx]
        lm_e = landmarks[end_idx]
        if lm_s.visibility < 0.4 or lm_e.visibility < 0.4:
            continue
        ps = (int(lm_s.x * image_w), int(lm_s.y * image_h))
        pe = (int(lm_e.x * image_w), int(lm_e.y * image_h))
        cv2.line(annotated, ps, pe, (200, 200, 200), 2, cv2.LINE_AA)

    # ── Highlight our key joints ──────────────────────────────────────────
    for name, idx in LANDMARKS_OF_INTEREST.items():
        lm = landmarks[idx]
        if lm.visibility < 0.4:
            continue
        px = int(lm.x * image_w)
        py = int(lm.y * image_h)
        joint_type = name.split("_")[1]          # e.g. "knee" from "left_knee"
        colour = _JOINT_COLOURS.get(joint_type, (255, 255, 255))
        cv2.circle(annotated, (px, py), 9,  colour,       -1,  cv2.LINE_AA)
        cv2.circle(annotated, (px, py), 11, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


# ── Core processing loop ──────────────────────────────────────────────────────

def process_video(
    video_path: str,
    model_path: str,
    output_dir: str,
    display: bool = True,
) -> str:
    """
    Run PoseLandmarker on every frame of *video_path*.

    Args:
        video_path:  Path to the input video file.
        model_path:  Path to the .task model bundle.
        output_dir:  Directory where the CSV will be written.
        display:     Show annotated video window while processing.

    Returns:
        Absolute path to the generated CSV file.
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    if not os.path.isfile(video_path):
        sys.exit(f"[ERROR] Video not found: {video_path}")
    if not os.path.isfile(model_path):
        sys.exit(
            f"[ERROR] Model not found: {model_path}\n"
            "Download it with:\n"
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
    print(f"  StrideFlow — Pose Engine")
    print(f"{'─'*62}")
    print(f"  Input  : {video_path}")
    print(f"  Model  : {model_path}")
    print(f"  Output : {csv_path}")
    print(f"  FPS    : {fps:.1f}  |  Frames: {total_frames}  |  {width}×{height}")
    print(f"{'─'*62}\n")

    # ── Build landmarker ──────────────────────────────────────────────────
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,    # frame-by-frame with timestamps
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

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

            # MediaPipe Tasks expects an mp.Image in RGB format
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect pose
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # ── Build CSV row ─────────────────────────────────────────────
            row = [frame_idx, timestamp_s]

            pose_landmarks_list = detection_result.pose_landmarks
            if pose_landmarks_list:
                landmarks = pose_landmarks_list[0]   # first (and only) pose
                for _, idx in LANDMARKS_OF_INTEREST.items():
                    lm = landmarks[idx]
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
                if pose_landmarks_list:
                    annotated = draw_skeleton_overlay(
                        bgr_frame, pose_landmarks_list[0], height, width
                    )
                else:
                    annotated = bgr_frame.copy()

                # HUD: frame + time
                hud = f"Frame {frame_idx:05d}  |  {timestamp_s:.2f}s"
                cv2.rectangle(annotated, (0, 0), (340, 34), (0, 0, 0), -1)
                cv2.putText(
                    annotated, hud,
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    (0, 255, 160), 2, cv2.LINE_AA,
                )

                # Legend (bottom-left)
                legend_items = [
                    ("Shoulder", _JOINT_COLOURS["shoulder"]),
                    ("Hip",      _JOINT_COLOURS["hip"]),
                    ("Knee",     _JOINT_COLOURS["knee"]),
                    ("Ankle",    _JOINT_COLOURS["ankle"]),
                ]
                ly = height - 10
                for label, colour in reversed(legend_items):
                    cv2.circle(annotated, (20, ly - 4), 6, colour, -1, cv2.LINE_AA)
                    cv2.putText(
                        annotated, label,
                        (32, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )
                    ly -= 24

                # Progress bar (bottom)
                progress = frame_idx / max(total_frames - 1, 1)
                bar_w    = int(width * progress)
                cv2.rectangle(annotated, (0, height - 5), (bar_w, height), (0, 255, 160), -1)

                cv2.imshow("StrideFlow — Pose Engine  [Q / Esc to quit]", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\n[INFO] User quit early.")
                    break

            frame_idx += 1

            # Console progress log every 60 frames
            if frame_idx % 60 == 0:
                elapsed = time.time() - start_time
                pct = (frame_idx / total_frames * 100) if total_frames else 0
                fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                print(
                    f"  [{pct:5.1f}%]  Frame {frame_idx:>6}/{total_frames}"
                    f"  {fps_proc:.1f} fps  ({elapsed:.1f}s elapsed)"
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


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="StrideFlow Pose Engine — extract joint landmarks from running video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video (absolute or relative to project root).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to the MediaPipe PoseLandmarker .task bundle.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help="Directory for the output CSV.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the live video window (headless / server mode).",
    )

    args = parser.parse_args()

    # Resolve paths relative to project root if not absolute
    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    process_video(
        video_path=resolve(args.video),
        model_path=resolve(args.model),
        output_dir=resolve(args.output_dir),
        display=not args.no_display,
    )
