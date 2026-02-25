"""FireTrace: Fire origin tracking video with optical flow.

Tracks the fire origin point across ALL frames using Lucas-Kanade
optical flow. As the camera moves, the origin marker follows naturally.

1. Cosmos-Reason2 predicts origin on the video (normalized coords)
2. Reference frame: middle frame → origin pixel position
3. Optical flow tracks that point forward and backward through all frames
4. Origin marker + spread arrows drawn per-frame at tracked position

Output: reports/scenario5/firetrace_0087.mp4, firetrace_0096.mp4
        demo/firetrace_demo.mp4
"""

import cv2
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"D:\Program Files\isaacsim\cookbook\firetrace")
RESULTS_PATH = BASE_DIR / "reports" / "results_combined.json"
DATA_DIR = Path(r"D:\Program Files\isaacsim\cookbook\data\fire_detection_aihub\Sample")
RAW_BASE = DATA_DIR / "01.원천데이터" / "화재현상"
NEW_DATA_DIR = BASE_DIR / "data" / "fire_dataset"
OUTPUT_DIR = BASE_DIR / "reports"
DEMO_PATH = BASE_DIR / "demo" / "firetrace_demo.mp4"

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

SUBDIRS = {"FL": "불꽃", "SM": "연기", "NONE": "정상"}
CLASS_TO_CODE = {"FLAME": "FL", "SMOKE": "SM", "NORMAL": "NONE"}

# Optical flow parameters
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def imread_unicode(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def fallback_origin_from_text(text):
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


def generate_spread_offsets(spread_dir):
    """Generate relative spread arrow offsets from origin."""
    offsets = []
    sd = str(spread_dir).lower()
    if "up" in sd:
        offsets.append((0, -80))
    if "outward" in sd:
        offsets.append((60, -50))
        offsets.append((-60, -50))
    if "left" in sd:
        offsets.append((-80, 0))
    if "right" in sd:
        offsets.append((80, 0))
    if not offsets:
        offsets.append((0, -60))
    return offsets


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
        f"FireTrace | Scene {scene_id}",
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
        cv2.putText(frame, "FireTrace Results", (100, 100), FONT_BOLD, 1.2, WHITE, 2, cv2.LINE_AA)
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


def build_tracked_scene_video(scene_id, gt_class, pred, max_frames=180, video_path=None):
    """Build video with optical-flow tracked origin marker."""
    # Try JPG frame directory first (AIHub dataset)
    class_code = CLASS_TO_CODE.get(gt_class, "NONE")
    subdir = SUBDIRS.get(class_code, "")
    jpg_dir = RAW_BASE / subdir / scene_id / "JPG"

    use_video_source = False
    if jpg_dir.exists():
        all_jpgs = sorted(jpg_dir.glob("*.jpg"))
        if all_jpgs:
            step = max(1, len(all_jpgs) // max_frames)
            selected = all_jpgs[::step][:max_frames]
    elif video_path:
        # Try to find original video
        vp = Path(video_path) if Path(video_path).exists() else None
        if vp is None:
            for sub in ["samples/part1", "samples/part2", "samples/part3", ""]:
                candidate = NEW_DATA_DIR / sub / f"{scene_id}.mp4"
                if candidate.exists():
                    vp = candidate
                    break
        if vp is None:
            vp = NEW_DATA_DIR / "preprocessed" / f"{scene_id}_prep.mp4"
        if vp is not None and vp.exists():
            print(f"    Extracting frames from video: {vp.name}")
            video_frames = extract_frames_from_video(vp, max_frames)
            if not video_frames:
                print(f"    No frames extracted from {vp}")
                return []
            use_video_source = True
            selected = video_frames
        else:
            print(f"    Scene dir not found: {jpg_dir}, no video found")
            return []
    else:
        print(f"    Scene dir not found: {jpg_dir}")
        return []

    if not use_video_source:
        all_jpgs = sorted(jpg_dir.glob("*.jpg"))
        if not all_jpgs:
            return []
        step = max(1, len(all_jpgs) // max_frames)
        selected = all_jpgs[::step][:max_frames]

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

    # Spread arrow offsets (relative pixels)
    arrow_offsets = generate_spread_offsets(pred.get("spread_direction", ""))

    # Render frames
    frames_out = []
    for idx in range(n):
        img = read_frame(selected[idx])
        if img is None:
            continue

        # Resize to output resolution
        fh, fw = img.shape[:2]
        scale = min(W / fw, H / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
        y_off = (H - nh) // 2
        x_off = (W - nw) // 2
        canvas[y_off:y_off + nh, x_off:x_off + nw] = resized

        # Map tracked position to canvas coordinates
        tx, ty = tracked_positions[idx]
        canvas_x = x_off + tx * scale
        canvas_y = y_off + ty * scale

        # Scale arrow offsets
        scaled_arrows = [(int(dx * scale), int(dy * scale)) for dx, dy in arrow_offsets]

        # Draw
        phase = idx * 0.15
        canvas = draw_origin_on_frame(canvas, canvas_x, canvas_y, scaled_arrows, phase)
        canvas = draw_hud(canvas, scene_id, gt_class, pred, idx, n)

        frames_out.append(canvas)

    return frames_out


def main():
    print("=" * 60)
    print("FireTrace: Tracked Origin Video Generator")
    print(f"  Output: {DEMO_PATH}")
    print(f"  Resolution: {W}x{H} @ {FPS}fps")
    print(f"  Tracking: Lucas-Kanade Optical Flow")
    print("=" * 60)

    if not RESULTS_PATH.exists():
        print(f"  ERROR: {RESULTS_PATH} not found")
        return

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    results = data.get("results", [])

    DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(DEMO_PATH), fourcc, FPS, (W, H))
    total_frames = 0

    def write(frames, label=""):
        nonlocal total_frames
        for f in frames:
            writer.write(f)
        total_frames += len(frames)
        if label:
            print(f"  {label}: {len(frames)} frames ({len(frames)/FPS:.1f}s)")

    # === INTRO ===
    write(create_title_card(
        "FireTrace",
        "Fire Origin Tracing with Cosmos-Reason2",
        "Optical Flow Tracked | NVIDIA Cosmos Cookoff 2026", 4
    ), "Intro")

    # === SCENE VIDEOS WITH TRACKED ORIGIN ===
    # Only use the first two AIHub scenes (0087 FLAME, 0096 SMOKE)
    DEMO_SCENES = {"0087", "0096"}
    for r in results:
        scene_id = r["scene_id"]
        if scene_id not in DEMO_SCENES:
            continue
        ev = r.get("evaluation", {})
        pred = r.get("prediction", {}) or {}
        gt_class = r["gt_class"]

        if ev.get("valid") and pred:
            print(f"\n  Scene {scene_id} ({gt_class}) - building tracked origin video...")
            scene_frames = build_tracked_scene_video(scene_id, gt_class, pred, max_frames=180, video_path=r.get("video_path"))
            if scene_frames:
                # 0087: only first 3 seconds (90 frames) of the 6s video
                if scene_id == "0087":
                    scene_frames = scene_frames[:90]
                write(scene_frames, f"Scene {scene_id} ({gt_class})")

                # Save individual scene video
                scene_path = OUTPUT_DIR / f"firetrace_{scene_id}.mp4"
                sw = cv2.VideoWriter(str(scene_path), fourcc, FPS, (W, H))
                for f in scene_frames:
                    sw.write(f)
                sw.release()
                print(f"    Saved: {scene_path}")
        else:
            print(f"\n  Scene {scene_id} ({gt_class}) - normal (no origin)")
            write(create_title_card(
                f"Scene {scene_id} - No Fire",
                "Model correctly identifies normal scene", "", 3
            ), f"Scene {scene_id} (NORMAL)")

    # === OUTRO ===
    write(create_title_card(
        "FireTrace", "Physics-Aware Fire Origin Tracing",
        "Built with NVIDIA Cosmos-Reason2", 3
    ), "Outro")

    writer.release()

    duration = total_frames / FPS
    size_mb = DEMO_PATH.stat().st_size / (1024 * 1024) if DEMO_PATH.exists() else 0
    print(f"\n{'='*60}")
    print(f"  Demo: {DEMO_PATH}")
    print(f"  Duration: {duration:.1f}s | Frames: {total_frames} | Size: {size_mb:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
