"""FireTrace: Fire origin visualization on representative frames.

Loads the middle frame from each fire scene, draws the predicted origin
marker and spread arrows, then saves annotated images.

Uses np.fromfile + cv2.imdecode for Korean/Unicode path support.
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

# Drawing constants
ORIGIN_COLOR = (0, 0, 255)       # Red (BGR)
ARROW_COLOR = (0, 200, 255)      # Yellow
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

SUBDIRS = {"FL": "불꽃", "SM": "연기", "NONE": "정상"}
CLASS_TO_CODE = {"FLAME": "FL", "SMOKE": "SM", "NORMAL": "NONE"}


def imread_unicode(path):
    """Read image from Unicode path."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """Write image to Unicode path."""
    ext = Path(path).suffix
    result, buf = cv2.imencode(ext, img)
    if result:
        buf.tofile(str(path))
        return True
    return False


def get_middle_frame(scene_id, gt_class, video_path=None):
    """Find and load the middle frame from a scene's JPG directory or video file."""
    # Try JPG directory first (AIHub dataset)
    class_code = CLASS_TO_CODE.get(gt_class, "NONE")
    subdir_name = SUBDIRS.get(class_code, "정상")
    scene_dir = RAW_BASE / subdir_name / scene_id / "JPG"

    if scene_dir.exists():
        frames = sorted(scene_dir.glob("*.jpg"))
        if frames:
            mid_idx = len(frames) // 2
            img = imread_unicode(frames[mid_idx])
            return img, frames[mid_idx]

    # Try video file (new dataset)
    vp = None
    if video_path and Path(video_path).exists():
        vp = Path(video_path)
    else:
        # Search common locations
        for sub in ["preprocessed", "samples/part1", "samples/part2", "samples/part3", ""]:
            for ext in [".mp4", "_prep.mp4"]:
                candidate = NEW_DATA_DIR / sub / f"{scene_id}{ext}"
                if candidate.exists():
                    vp = candidate
                    break
            if vp:
                break

    if vp and vp.exists():
        cap = cv2.VideoCapture(str(vp))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid = total // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame, vp
    return None, None


def get_frame_sequence(scene_id, gt_class, count=5, video_path=None):
    """Get evenly-spaced frames for temporal sequence visualization."""
    # Try JPG directory first (AIHub dataset)
    class_code = CLASS_TO_CODE.get(gt_class, "NONE")
    subdir_name = SUBDIRS.get(class_code, "정상")
    scene_dir = RAW_BASE / subdir_name / scene_id / "JPG"

    if scene_dir.exists():
        frames = sorted(scene_dir.glob("*.jpg"))
        if frames:
            step = max(1, len(frames) // count)
            selected = frames[::step][:count]
            images = []
            for fp in selected:
                img = imread_unicode(fp)
                if img is not None:
                    images.append((img, fp.name))
            return images

    # Try video file (new dataset)
    vp = None
    if video_path and Path(video_path).exists():
        vp = Path(video_path)
    else:
        for sub in ["preprocessed", "samples/part1", "samples/part2", "samples/part3", ""]:
            for ext in [".mp4", "_prep.mp4"]:
                candidate = NEW_DATA_DIR / sub / f"{scene_id}{ext}"
                if candidate.exists():
                    vp = candidate
                    break
            if vp:
                break

    if vp and vp.exists():
        cap = cv2.VideoCapture(str(vp))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // count)
        images = []
        for i in range(count):
            fnum = min(i * step, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ret, frame = cap.read()
            if ret:
                images.append((frame, f"frame_{fnum:05d}"))
        cap.release()
        return images

    return []


def draw_origin_marker(img, origin_x, origin_y, radius=30):
    """Draw a crosshair-style origin marker."""
    h, w = img.shape[:2]
    cx = int(origin_x * w)
    cy = int(origin_y * h)

    # Outer glow rings
    for r, alpha in [(radius + 20, 0.15), (radius + 10, 0.25), (radius, 0.4)]:
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), r, ORIGIN_COLOR, 3)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Solid inner circle
    cv2.circle(img, (cx, cy), radius // 2, ORIGIN_COLOR, -1)
    cv2.circle(img, (cx, cy), radius // 2, (255, 255, 255), 2)

    # Crosshair lines
    line_len = radius + 15
    cv2.line(img, (cx - line_len, cy), (cx - radius // 2 - 4, cy), ORIGIN_COLOR, 2)
    cv2.line(img, (cx + radius // 2 + 4, cy), (cx + line_len, cy), ORIGIN_COLOR, 2)
    cv2.line(img, (cx, cy - line_len), (cx, cy - radius // 2 - 4), ORIGIN_COLOR, 2)
    cv2.line(img, (cx, cy + radius // 2 + 4), (cx, cy + line_len), ORIGIN_COLOR, 2)

    # "ORIGIN" label
    label = "ORIGIN"
    (tw, th), baseline = cv2.getTextSize(label, FONT, 0.7, 2)
    lx = cx - tw // 2
    ly = cy - radius - 15
    cv2.rectangle(img, (lx - 4, ly - th - 4), (lx + tw + 4, ly + baseline + 4), TEXT_BG, -1)
    cv2.putText(img, label, (lx, ly), FONT, 0.7, ORIGIN_COLOR, 2, cv2.LINE_AA)

    return img


def draw_spread_arrows(img, arrows):
    """Draw spread direction arrows on the image."""
    h, w = img.shape[:2]
    for arrow in arrows:
        fx = int(arrow.get("from_x", 0) * w)
        fy = int(arrow.get("from_y", 0) * h)
        tx = int(arrow.get("to_x", 0) * w)
        ty = int(arrow.get("to_y", 0) * h)
        cv2.arrowedLine(img, (fx, fy), (tx, ty), ARROW_COLOR, 3, tipLength=0.15)
    return img


def draw_info_panel(img, scene_id, gt_class, prediction, evaluation):
    """Draw an info panel at the bottom of the image."""
    h, w = img.shape[:2]
    panel_h = 100
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    y_base = h - panel_h + 25
    left = 20

    pred_class = prediction.get("classification", "?")
    severity = prediction.get("severity", "?")
    urgency = prediction.get("urgency", "?")
    origin_text = prediction.get("fire_origin", "N/A")
    if len(origin_text) > 50:
        origin_text = origin_text[:47] + "..."

    cv2.putText(img, f"Scene {scene_id}  |  GT: {gt_class}  |  Pred: {pred_class}",
                (left, y_base), FONT, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, f"Severity: {severity}  |  Urgency: {urgency}  |  Origin: {origin_text}",
                (left, y_base + 30), FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # FireTrace badge
    badge = "FireTrace"
    (bw, bh), _ = cv2.getTextSize(badge, FONT, 0.6, 2)
    bx = w - bw - 30
    by = y_base
    cv2.rectangle(img, (bx - 8, by - bh - 6), (bx + bw + 8, by + 6), (22, 115, 249), -1)
    cv2.putText(img, badge, (bx, by), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return img


def generate_fallback_arrows(origin_x, origin_y, spread_direction):
    """Generate spread arrows from text when model doesn't provide coordinates."""
    arrows = []
    sd = str(spread_direction).lower()

    if "up" in sd:
        arrows.append({"from_x": origin_x, "from_y": origin_y,
                        "to_x": origin_x, "to_y": max(0.0, origin_y - 0.25)})
    if "outward" in sd:
        arrows.append({"from_x": origin_x, "from_y": origin_y,
                        "to_x": min(1.0, origin_x + 0.2), "to_y": max(0.0, origin_y - 0.15)})
        arrows.append({"from_x": origin_x, "from_y": origin_y,
                        "to_x": max(0.0, origin_x - 0.2), "to_y": max(0.0, origin_y - 0.15)})
    if "left" in sd:
        arrows.append({"from_x": origin_x, "from_y": origin_y,
                        "to_x": max(0.0, origin_x - 0.25), "to_y": origin_y})
    if "right" in sd:
        arrows.append({"from_x": origin_x, "from_y": origin_y,
                        "to_x": min(1.0, origin_x + 0.25), "to_y": origin_y})

    if not arrows:
        arrows.append({"from_x": origin_x, "from_y": origin_y,
                        "to_x": origin_x, "to_y": max(0.0, origin_y - 0.2)})

    return arrows


def fallback_origin_from_text(origin_text):
    """Estimate origin coordinates from text description."""
    text = str(origin_text).lower()
    x, y = 0.5, 0.5

    if "left" in text:
        x = 0.25
    elif "right" in text:
        x = 0.75
    if "top" in text or "upper" in text:
        y = 0.25
    elif "bottom" in text or "lower" in text or "base" in text:
        y = 0.75
    if "center" in text or "middle" in text:
        if "left" not in text and "right" not in text:
            x = 0.5
        if "top" not in text and "bottom" not in text:
            y = 0.5

    return x, y


def create_temporal_strip(scene_id, gt_class, origin_x, origin_y, output_path, video_path=None):
    """Create a temporal sequence strip showing fire progression."""
    seq = get_frame_sequence(scene_id, gt_class, count=5, video_path=video_path)
    if not seq:
        return None

    # Resize each to 384x216 and arrange horizontally
    thumb_w, thumb_h = 384, 216
    strip_w = thumb_w * len(seq) + 10 * (len(seq) - 1)
    strip = np.full((thumb_h + 40, strip_w, 3), (10, 10, 15), dtype=np.uint8)

    for i, (img, fname) in enumerate(seq):
        resized = cv2.resize(img, (thumb_w, thumb_h))

        # Draw small origin marker on each frame
        mcx = int(origin_x * thumb_w)
        mcy = int(origin_y * thumb_h)
        cv2.circle(resized, (mcx, mcy), 8, ORIGIN_COLOR, 2)

        x_off = i * (thumb_w + 10)
        strip[0:thumb_h, x_off:x_off + thumb_w] = resized

        # Frame label
        label = fname.split("_")[-1].replace(".jpg", "")
        cv2.putText(strip, f"Frame {label}", (x_off + 5, thumb_h + 25),
                    FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Arrow between frames
    for i in range(len(seq) - 1):
        x1 = (i + 1) * (thumb_w + 10) - 8
        x2 = x1 + 6
        y_mid = thumb_h // 2
        cv2.arrowedLine(strip, (x1, y_mid), (x2, y_mid), (100, 100, 100), 2, tipLength=0.5)

    imwrite_unicode(output_path, strip)
    return output_path


def main():
    print("=" * 60)
    print("FireTrace: Fire Origin Visualization")
    print("=" * 60)

    if not RESULTS_PATH.exists():
        print(f"  ERROR: Results not found: {RESULTS_PATH}")
        return

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = []

    for r in data.get("results", []):
        ev = r.get("evaluation", {})
        if not ev.get("valid"):
            continue

        scene_id = r["scene_id"]
        gt_class = r["gt_class"]
        prediction = r.get("prediction", {})

        print(f"\n  Processing scene {scene_id} (GT: {gt_class})...")

        # Load middle frame
        img, frame_path = get_middle_frame(scene_id, gt_class, video_path=r.get("video_path"))
        if img is None:
            print(f"    SKIP: Could not load frame")
            continue

        print(f"    Frame: {Path(frame_path).name} ({img.shape[1]}x{img.shape[0]})")

        # Get origin coordinates
        origin_x = prediction.get("origin_x")
        origin_y = prediction.get("origin_y")

        if origin_x is None or origin_y is None:
            origin_text = prediction.get("fire_origin", "")
            origin_x, origin_y = fallback_origin_from_text(origin_text)
            print(f"    Origin (text fallback): ({origin_x:.2f}, {origin_y:.2f})")
        else:
            print(f"    Origin (model coords): ({origin_x:.2f}, {origin_y:.2f})")

        # Draw origin marker
        img = draw_origin_marker(img, origin_x, origin_y)

        # Get spread arrows
        arrows = prediction.get("spread_arrows", [])
        if not arrows:
            spread_dir = prediction.get("spread_direction", "")
            arrows = generate_fallback_arrows(origin_x, origin_y, spread_dir)
            print(f"    Arrows (fallback): {len(arrows)}")
        else:
            print(f"    Arrows (model): {len(arrows)}")

        img = draw_spread_arrows(img, arrows)
        img = draw_info_panel(img, scene_id, gt_class, prediction, ev)

        # Save origin image
        output_path = OUTPUT_DIR / f"origin_{scene_id}.jpg"
        imwrite_unicode(output_path, img)
        print(f"    Saved: {output_path}")
        generated.append(str(output_path))

        # Save temporal strip
        strip_path = OUTPUT_DIR / f"temporal_{scene_id}.jpg"
        result = create_temporal_strip(scene_id, gt_class, origin_x, origin_y, strip_path, video_path=r.get("video_path"))
        if result:
            print(f"    Temporal strip: {strip_path}")
            generated.append(str(strip_path))

    print(f"\n{'='*60}")
    print(f"  Generated {len(generated)} image(s)")
    for p in generated:
        print(f"    {p}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
