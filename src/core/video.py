"""FlashBack: Fire origin tracking video with optical flow.

Tracks the fire origin point across ALL frames using Lucas-Kanade
optical flow. As the camera moves, the origin marker follows naturally.

1. Cosmos-Reason2 predicts origin on the video (normalized coords)
2. Reference frame: middle frame → origin pixel position
3. Optical flow tracks that point forward and backward through all frames
4. Origin marker + spread arrows drawn per-frame at tracked position

Output: demo/demo_2b.mp4, demo/demo_8b.mp4
        reports/firetrace_sample1.mp4, firetrace_sample2.mp4
"""

import cv2
import json
import numpy as np
from pathlib import Path

from src.config import (
    RESULTS_COMBINED, FIRE_FRAMES_DIR, FIRE_DATASET_DIR,
    REPORTS_DIR, DEMO_VIDEO, SUBDIRS, CLASS_TO_CODE,
)
from src.utils.common import imread_unicode, fallback_origin_from_text, generate_spread_offsets, compute_dynamic_spread_arrows

W, H = 1920, 1080
FPS = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

BG_COLOR = (10, 10, 15)
ORIGIN_COLOR = (0, 0, 255)
ARROW_COLOR = (0, 200, 255)
WHITE = (237, 228, 228)
GRAY = (159, 163, 156)
GREEN = (94, 197, 34)
ACCENT = (22, 115, 249)

# Optical flow parameters
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def track_origin_optical_flow(frame_paths, ref_idx, ref_point, read_fn=None):
    """Track a point across all frames using Lucas-Kanade optical flow.

    Args:
        frame_paths: List of frame file paths or numpy arrays (sorted)
        ref_idx: Index of the reference frame where origin is known
        ref_point: (x, y) pixel coordinates on the reference frame
        read_fn: Optional function to read a frame item (path or array)

    Returns:
        List of (x, y) tracked positions for each frame, or None if lost.
    """
    if read_fn is None:
        read_fn = imread_unicode

    n = len(frame_paths)
    tracked = [None] * n
    tracked[ref_idx] = ref_point

    # Load reference frame
    ref_img = read_fn(frame_paths[ref_idx])
    if ref_img is None:
        return tracked

    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # --- Track FORWARD from ref_idx ---
    prev_gray = ref_gray.copy()
    prev_pt = np.array([[ref_point]], dtype=np.float32)

    for i in range(ref_idx + 1, n):
        cur_img = read_fn(frame_paths[i])
        if cur_img is None:
            break
        cur_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, prev_pt, None, **LK_PARAMS
        )

        if status is not None and status[0][0] == 1:
            px, py = next_pts[0][0]
            h, w = cur_gray.shape
            # Clamp to image bounds
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            tracked[i] = (float(px), float(py))
            prev_pt = next_pts
        else:
            # Lost tracking -- extrapolate from last known
            if tracked[i - 1] is not None:
                tracked[i] = tracked[i - 1]
            break

        prev_gray = cur_gray

    # --- Track BACKWARD from ref_idx ---
    prev_gray = ref_gray.copy()
    prev_pt = np.array([[ref_point]], dtype=np.float32)

    for i in range(ref_idx - 1, -1, -1):
        cur_img = read_fn(frame_paths[i])
        if cur_img is None:
            break
        cur_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, prev_pt, None, **LK_PARAMS
        )

        if status is not None and status[0][0] == 1:
            px, py = next_pts[0][0]
            h, w = cur_gray.shape
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            tracked[i] = (float(px), float(py))
            prev_pt = next_pts
        else:
            if tracked[i + 1] is not None:
                tracked[i] = tracked[i + 1]
            break

        prev_gray = cur_gray

    # Fill gaps with nearest known position
    last_known = ref_point
    for i in range(n):
        if tracked[i] is not None:
            last_known = tracked[i]
        else:
            tracked[i] = last_known

    return tracked


def draw_origin_on_frame(frame, cx, cy, arrow_offsets, pulse_phase=0.0):
    """Draw animated origin marker at pixel coordinates."""
    cx, cy = int(cx), int(cy)
    base_r = 20
    pulse_r = base_r + int(6 * abs(np.sin(pulse_phase)))

    # Outer rings
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), pulse_r + 12, ORIGIN_COLOR, 2)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), pulse_r + 6, ORIGIN_COLOR, 2)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Solid center
    cv2.circle(frame, (cx, cy), base_r // 2, ORIGIN_COLOR, -1)
    cv2.circle(frame, (cx, cy), base_r // 2, (255, 255, 255), 2)

    # Crosshair
    ll = pulse_r + 8
    gap = base_r // 2 + 3
    cv2.line(frame, (cx - ll, cy), (cx - gap, cy), ORIGIN_COLOR, 2)
    cv2.line(frame, (cx + gap, cy), (cx + ll, cy), ORIGIN_COLOR, 2)
    cv2.line(frame, (cx, cy - ll), (cx, cy - gap), ORIGIN_COLOR, 2)
    cv2.line(frame, (cx, cy + gap), (cx, cy + ll), ORIGIN_COLOR, 2)

    # ORIGIN label
    label = "ORIGIN"
    (tw, th), bl = cv2.getTextSize(label, FONT, 0.55, 2)
    lx = cx - tw // 2
    ly = cy - pulse_r - 10
    cv2.rectangle(frame, (lx - 3, ly - th - 3), (lx + tw + 3, ly + bl + 3), (0, 0, 0), -1)
    cv2.putText(frame, label, (lx, ly), FONT, 0.55, ORIGIN_COLOR, 2, cv2.LINE_AA)

    # Spread arrows (relative to tracked origin)
    for (dx, dy) in arrow_offsets:
        tx, ty = cx + dx, cy + dy
        cv2.arrowedLine(frame, (cx, cy), (tx, ty), ARROW_COLOR, 2, tipLength=0.15)

    return frame


def draw_hud(frame, scene_id, gt_class, pred, frame_idx, total_frames):
    """Draw heads-up display."""
    h, w = frame.shape[:2]
    lines = [
        f"FlashBack | Scene {scene_id}",
        f"GT: {gt_class} | Pred: {pred.get('classification', '?')}",
        f"Origin: {str(pred.get('fire_origin', 'N/A'))[:45]}",
        f"Severity: {pred.get('severity', '?')} | Urgency: {pred.get('urgency', '?')}",
    ]
    max_tw = max(cv2.getTextSize(l, FONT, 0.55, 1)[0][0] for l in lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (max_tw + 30, 22 + 25 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = 32
    for i, line in enumerate(lines):
        color = ACCENT if i == 0 else WHITE if i < 2 else GRAY
        cv2.putText(frame, line, (18, y), FONT, 0.55, color, 1, cv2.LINE_AA)
        y += 25

    # Progress bar
    bar_y = h - 8
    progress = frame_idx / max(total_frames - 1, 1)
    cv2.rectangle(frame, (0, bar_y), (w, h), (30, 30, 30), -1)
    cv2.rectangle(frame, (0, bar_y), (int(w * progress), h), ACCENT, -1)

    counter = f"Frame {frame_idx+1}/{total_frames}"
    (cw, _), _ = cv2.getTextSize(counter, FONT, 0.45, 1)
    cv2.putText(frame, counter, (w - cw - 15, h - 15), FONT, 0.45, GRAY, 1, cv2.LINE_AA)

    return frame


def create_title_card(title, subtitle="", sub2="", duration_sec=3):
    frames = []
    total = int(FPS * duration_sec)
    for i in range(total):
        frame = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (W, 4), ACCENT, -1)
        for x in range(W):
            r_val = int(249 * (1 - x / W) + 239 * (x / W))
            g_val = int(115 * (1 - x / W) + 68 * (x / W))
            b_val = int(22 * (1 - x / W) + 68 * (x / W))
            cv2.line(frame, (x, 4), (x, 8), (b_val, g_val, r_val), 1)

        (tw, _), _ = cv2.getTextSize(title, FONT_BOLD, 2.2, 4)
        cv2.putText(frame, title, ((W - tw) // 2, H // 2 - 20), FONT_BOLD, 2.2, WHITE, 4, cv2.LINE_AA)
        if subtitle:
            (sw, _), _ = cv2.getTextSize(subtitle, FONT, 0.9, 2)
            cv2.putText(frame, subtitle, ((W - sw) // 2, H // 2 + 40), FONT, 0.9, GRAY, 2, cv2.LINE_AA)
        if sub2:
            (s2w, _), _ = cv2.getTextSize(sub2, FONT, 0.7, 1)
            cv2.putText(frame, sub2, ((W - s2w) // 2, H // 2 + 85), FONT, 0.7, ACCENT, 1, cv2.LINE_AA)

        fade = int(FPS * 0.4)
        alpha = 1.0
        if i < fade: alpha = i / fade
        elif i > total - fade: alpha = (total - i) / fade
        if alpha < 1.0:
            frame = (frame * alpha).astype(np.uint8)
        frames.append(frame)
    return frames


def create_metrics_card(metrics, duration_sec=4):
    frames = []
    total = int(FPS * duration_sec)
    items = [
        ("Hazard Detection", f"{metrics.get('hazard_detection_accuracy', 0):.0%}", GREEN),
        ("Origin Tracing", f"{metrics.get('fire_origin_reasoning_rate', 0):.0%}", (8, 179, 234)),
        ("Temporal Reasoning", f"{metrics.get('temporal_reasoning_rate', 0):.0%}", ACCENT),
        ("Urgency Accuracy", f"{metrics.get('urgency_accuracy', 0):.0%}", WHITE),
    ]
    for i in range(total):
        frame = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (W, 4), ACCENT, -1)
        cv2.putText(frame, "FlashBack Results", (100, 100), FONT_BOLD, 1.2, WHITE, 2, cv2.LINE_AA)
        card_w, card_h = 380, 140
        total_w = len(items) * card_w + (len(items) - 1) * 30
        sx = (W - total_w) // 2
        for idx, (label, value, color) in enumerate(items):
            cx = sx + idx * (card_w + 30)
            cy = 200
            cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), (30, 30, 35), -1)
            cv2.rectangle(frame, (cx, cy), (cx + card_w, cy + card_h), (50, 50, 55), 2)
            (vw, _), _ = cv2.getTextSize(value, FONT_BOLD, 1.6, 3)
            cv2.putText(frame, value, (cx + (card_w - vw) // 2, cy + 70), FONT_BOLD, 1.6, color, 3, cv2.LINE_AA)
            (lw, _), _ = cv2.getTextSize(label, FONT, 0.55, 1)
            cv2.putText(frame, label, (cx + (card_w - lw) // 2, cy + 110), FONT, 0.55, GRAY, 1, cv2.LINE_AA)
        fade = int(FPS * 0.3)
        if i < fade:
            frame = (frame * (i / fade)).astype(np.uint8)
        frames.append(frame)
    return frames


def extract_frames_from_video(video_path, max_frames=180):
    """Extract frames from a video file as numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0 and len(frames) < max_frames:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def build_tracked_scene_video(scene_id, gt_class, pred, max_frames=None, video_path=None):
    """Build video with optical-flow tracked origin marker.

    Args:
        max_frames: Maximum frames to use. None = use all frames from video.
    """
    # Try video file first (preferred — gives all frames for smooth playback)
    selected = None
    use_video_source = False

    vp = Path(video_path) if video_path and Path(video_path).exists() else None
    if vp is None:
        # Search common locations
        for sub in ["preprocessed", "samples/part1", "samples/part2", "samples/part3", ""]:
            candidate = FIRE_DATASET_DIR / sub / f"{scene_id}.mp4"
            if candidate.exists():
                vp = candidate
                break
        if vp is None:
            vp = FIRE_DATASET_DIR / "preprocessed" / f"{scene_id}_prep.mp4"

    if vp is not None and vp.exists():
        print(f"    Extracting frames from video: {vp.name}")
        cap = cv2.VideoCapture(str(vp))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"    Source: {total} frames @ {src_fps:.0f}fps")

        # Extract all frames (or subsample if max_frames is set)
        step = 1
        limit = total
        if max_frames is not None and max_frames < total:
            step = max(1, total // max_frames)
            limit = max_frames

        video_frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0 and len(video_frames) < limit:
                video_frames.append(frame)
            idx += 1
        cap.release()

        if video_frames:
            use_video_source = True
            selected = video_frames
            print(f"    Extracted {len(selected)} frames (step={step})")

    # Fallback to exported frame images
    if selected is None and FIRE_FRAMES_DIR.exists():
        all_jpgs = sorted(FIRE_FRAMES_DIR.glob(f"{scene_id}_*.jpg"))
        if all_jpgs:
            if max_frames is not None and max_frames < len(all_jpgs):
                step = max(1, len(all_jpgs) // max_frames)
                selected = all_jpgs[::step][:max_frames]
            else:
                selected = all_jpgs

    if selected is None or len(selected) == 0:
        print(f"    No frames or video found for {scene_id}")
        return []

    n = len(selected)

    # Get origin coordinates on reference frame
    ox = pred.get("origin_x")
    oy = pred.get("origin_y")
    if ox is None or oy is None:
        ox, oy = fallback_origin_from_text(pred.get("fire_origin", ""))

    # Helper to read frame regardless of source type
    def read_frame(item):
        if isinstance(item, np.ndarray):
            return item
        return imread_unicode(item)

    # Reference frame = middle
    ref_idx = n // 2
    ref_img = read_frame(selected[ref_idx])
    if ref_img is None:
        return []

    fh, fw = ref_img.shape[:2]
    ref_px = (ox * fw, oy * fh)

    print(f"    Tracking origin from frame {ref_idx} at ({ref_px[0]:.0f}, {ref_px[1]:.0f})...")
    print(f"    Frames: {n}, Source resolution: {fw}x{fh}")

    # Track with optical flow
    tracked_positions = track_origin_optical_flow(selected, ref_idx, ref_px, read_fn=read_frame)

    # Prepare spread arrows from model coordinates or fallback
    model_arrows = pred.get("spread_arrows", [])
    if model_arrows:
        # Convert model's normalized arrows to pixel offsets relative to origin
        arrow_pixel_offsets = []
        for arrow in model_arrows:
            dx = (arrow["to_x"] - arrow["from_x"]) * fw
            dy = (arrow["to_y"] - arrow["from_y"]) * fh
            arrow_pixel_offsets.append((dx, dy))
        print(f"    Arrows: {len(arrow_pixel_offsets)} from model coordinates")
    else:
        arrow_pixel_offsets = [(dx, dy) for dx, dy in generate_spread_offsets(pred.get("spread_direction", ""))]
        print(f"    Arrows: {len(arrow_pixel_offsets)} from text fallback")

    # Render frames
    frames_out = []
    for idx in range(n):
        img = read_frame(selected[idx])
        if img is None:
            continue

        # Resize to output resolution
        cur_fh, cur_fw = img.shape[:2]
        scale = min(W / cur_fw, H / cur_fh)
        nw, nh = int(cur_fw * scale), int(cur_fh * scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
        y_off = (H - nh) // 2
        x_off = (W - nw) // 2
        canvas[y_off:y_off + nh, x_off:x_off + nw] = resized

        # Map tracked position to canvas coordinates
        tx, ty = tracked_positions[idx]
        canvas_x = x_off + tx * scale
        canvas_y = y_off + ty * scale

        # Scale arrow offsets to canvas resolution
        scaled_arrows = [(int(dx * scale), int(dy * scale)) for dx, dy in arrow_pixel_offsets]

        # Draw
        phase = idx * 0.15
        canvas = draw_origin_on_frame(canvas, canvas_x, canvas_y, scaled_arrows, phase)
        canvas = draw_hud(canvas, scene_id, gt_class, pred, idx, n)

        frames_out.append(canvas)

    return frames_out


def create_scene_title_card(scene_id, gt_class, pred, duration_sec=2.5):
    """Create a scene introduction card with key prediction info."""
    frames = []
    total = int(FPS * duration_sec)

    origin_text = pred.get("fire_origin", "N/A")
    if len(origin_text) > 60:
        origin_text = origin_text[:57] + "..."
    severity = pred.get("severity", "?")
    stage = pred.get("fire_stage", "?")
    urgency = pred.get("urgency", "?")

    # Color by class
    class_color = (0, 100, 255) if gt_class == "FLAME" else (180, 180, 0) if gt_class == "SMOKE" else GRAY

    for i in range(total):
        frame = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)

        # Top accent bar
        cv2.rectangle(frame, (0, 0), (W, 4), ACCENT, -1)

        # Scene badge
        badge = f"SCENE: {scene_id.upper()}"
        (bw, bh), _ = cv2.getTextSize(badge, FONT_BOLD, 1.0, 2)
        bx = (W - bw) // 2
        cv2.putText(frame, badge, (bx, H // 2 - 80), FONT_BOLD, 1.0, WHITE, 2, cv2.LINE_AA)

        # Class label with color
        cls_label = f"Ground Truth: {gt_class}"
        (cw, _), _ = cv2.getTextSize(cls_label, FONT, 0.7, 2)
        cv2.putText(frame, cls_label, ((W - cw) // 2, H // 2 - 30), FONT, 0.7, class_color, 2, cv2.LINE_AA)

        # Prediction details
        details = [
            f"Origin: {origin_text}",
            f"Severity: {severity}  |  Stage: {stage}  |  Urgency: {urgency}",
        ]
        for j, line in enumerate(details):
            (lw, _), _ = cv2.getTextSize(line, FONT, 0.55, 1)
            cv2.putText(frame, line, ((W - lw) // 2, H // 2 + 20 + j * 30),
                        FONT, 0.55, GRAY, 1, cv2.LINE_AA)

        # Bottom accent
        cv2.rectangle(frame, (0, H - 4), (W, H), class_color, -1)

        # Fade in
        fade = int(FPS * 0.3)
        if i < fade:
            frame = (frame * (i / fade)).astype(np.uint8)
        elif i > total - fade:
            frame = (frame * ((total - i) / fade)).astype(np.uint8)

        frames.append(frame)
    return frames


def generate_demo(results_path, output_path, model_label="Cosmos-Reason2"):
    """Generate a professional demo video from a results JSON file.

    Structure: Opening -> Scene Title -> Scene Video -> Scene Title -> Scene Video
    """
    print(f"\n  Results: {results_path}")
    print(f"  Output:  {output_path}")

    if not Path(results_path).exists():
        print(f"  ERROR: {results_path} not found")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (W, H))
    total_frames = 0

    def write(frames, label=""):
        nonlocal total_frames
        for f in frames:
            writer.write(f)
        total_frames += len(frames)
        if label:
            print(f"  {label}: {len(frames)} frames ({len(frames)/FPS:.1f}s)")

    # === OPENING ===
    write(create_title_card(
        "FlashBack",
        f"Rewinding Fire to Its Origin",
        f"{model_label} | Lucas-Kanade Optical Flow | NVIDIA Cosmos Cookoff 2026", 4
    ), "Opening")

    # === SCENES ===
    DEMO_SCENES = ["sample1", "sample2"]
    scene_map = {r["scene_id"]: r for r in results}

    for scene_id in DEMO_SCENES:
        r = scene_map.get(scene_id)
        if not r:
            print(f"\n  SKIP: {scene_id} not in results")
            continue

        ev = r.get("evaluation", {})
        pred = r.get("prediction", {}) or {}
        gt_class = r["gt_class"]

        if not (ev.get("valid") and pred):
            continue

        # Scene title card
        write(create_scene_title_card(scene_id, gt_class, pred, duration_sec=2.5),
              f"Title: {scene_id}")

        # Scene video with tracked origin
        print(f"\n  Scene {scene_id} ({gt_class}) - building tracked origin video...")
        scene_frames = build_tracked_scene_video(
            scene_id, gt_class, pred, max_frames=None, video_path=r.get("video_path"))

        if scene_frames:
            # sample1: use first half only
            if scene_id == "sample1":
                scene_frames = scene_frames[:len(scene_frames) // 2]
            write(scene_frames, f"Scene {scene_id} ({gt_class})")

            # Save individual scene video
            scene_path = REPORTS_DIR / f"firetrace_{scene_id}.mp4"
            sw = cv2.VideoWriter(str(scene_path), fourcc, FPS, (W, H))
            for f in scene_frames:
                sw.write(f)
            sw.release()
            print(f"    Saved: {scene_path}")

    writer.release()

    duration = total_frames / FPS
    out = Path(output_path)
    size_mb = out.stat().st_size / (1024 * 1024) if out.exists() else 0
    print(f"\n  Demo: {output_path}")
    print(f"  Duration: {duration:.1f}s | Frames: {total_frames} | Size: {size_mb:.1f} MB")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FlashBack Demo Video Generator")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "2b", "8b", "combined"],
                        help="Which model results to use (default: all)")
    args = parser.parse_args()

    print("=" * 60)
    print("FlashBack: Tracked Origin Video Generator")
    print(f"  Resolution: {W}x{H} @ {FPS}fps")
    print(f"  Tracking: Lucas-Kanade Optical Flow + Dense Flow Arrows")
    print("=" * 60)

    DEMO_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.model in ("all", "2b"):
        jobs.append((
            REPORTS_DIR / "results_cosmos_reason2_2b.json",
            DEMO_VIDEO.parent / "demo_2b.mp4",
            "Cosmos-Reason2-2B",
        ))
    if args.model in ("all", "8b"):
        jobs.append((
            REPORTS_DIR / "results_cosmos_reason2_8b.json",
            DEMO_VIDEO.parent / "demo_8b.mp4",
            "Cosmos-Reason2-8B",
        ))
    if args.model in ("all", "combined"):
        jobs.append((
            RESULTS_COMBINED,
            DEMO_VIDEO,
            "Cosmos-Reason2",
        ))

    for results_path, output_path, model_label in jobs:
        if Path(results_path).exists():
            print(f"\n{'='*60}")
            print(f"  Generating demo for {model_label}...")
            print(f"{'='*60}")
            generate_demo(results_path, output_path, model_label)
        else:
            print(f"\n  SKIP: {results_path} not found")

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
