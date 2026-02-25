"""FireTrace: FiftyOne-based fire origin tracing demo.

Creates a FiftyOne dataset with fire scene frames annotated with:
- Fire origin keypoint
- Spread direction polylines
- Classification labels
- Physics reasoning as sample fields

Can launch FiftyOne app for interactive exploration.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import fiftyone as fo
    import fiftyone.types as fot
except ImportError:
    print("ERROR: fiftyone is not installed. Run: pip install fiftyone")
    sys.exit(1)

BASE_DIR = Path(r"D:\Program Files\isaacsim\cookbook\firetrace")
DATA_DIR = Path(r"D:\Program Files\isaacsim\cookbook\data\fire_detection_aihub\Sample")
RAW_BASE = DATA_DIR / "01.원천데이터" / "화재현상"
RESULTS_PATH = BASE_DIR / "reports" / "results_combined.json"
ORIGIN_DIR = BASE_DIR / "reports"
VIDEO_DIR = BASE_DIR / "data" / "fire_videos"

SUBDIRS = {"FL": "불꽃", "SM": "연기", "NONE": "정상"}
CLASS_TO_CODE = {"FLAME": "FL", "SMOKE": "SM", "NORMAL": "NONE"}


def imread_unicode(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = Path(path).suffix
    result, buf = cv2.imencode(ext, img)
    if result:
        buf.tofile(str(path))
        return True
    return False


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


def generate_fallback_arrows(origin_x, origin_y, spread_direction):
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


def export_representative_frames(results_data=None):
    """Export middle frames from each scene as standalone images for FiftyOne."""
    export_dir = BASE_DIR / "data" / "fire_frames"
    export_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    # AIHub JPG directories
    for gt_class, class_code in CLASS_TO_CODE.items():
        subdir = SUBDIRS.get(class_code, "정상")
        class_dir = RAW_BASE / subdir
        if not class_dir.exists():
            continue
        for scene_dir in sorted(class_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            jpg_dir = scene_dir / "JPG"
            if not jpg_dir.exists():
                continue
            frames = sorted(jpg_dir.glob("*.jpg"))
            if not frames:
                continue

            scene_id = scene_dir.name
            indices = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]
            scene_frames = []
            for idx in indices:
                src = frames[min(idx, len(frames) - 1)]
                dst = export_dir / f"{scene_id}_{gt_class}_f{idx:04d}.jpg"
                if not dst.exists():
                    img = imread_unicode(src)
                    if img is not None:
                        imwrite_unicode(dst, img)
                if dst.exists():
                    scene_frames.append((dst, idx))

            exported[scene_id] = {
                "gt_class": gt_class,
                "frames": scene_frames,
                "total_frames": len(frames),
            }

    # New dataset: extract frames from video files
    if results_data:
        for r in results_data.get("results", []):
            scene_id = r["scene_id"]
            if scene_id in exported:
                continue
            video_path = Path(r.get("video_path", ""))
            if not video_path.exists():
                continue
            gt_class = r.get("gt_class", "UNKNOWN")

            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                continue

            indices = [0, total // 4, total // 2, 3 * total // 4, total - 1]
            scene_frames = []
            for idx in indices:
                dst = export_dir / f"{scene_id}_{gt_class}_f{idx:04d}.jpg"
                if not dst.exists():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        imwrite_unicode(dst, frame)
                if dst.exists():
                    scene_frames.append((dst, idx))

            cap.release()
            if scene_frames:
                exported[scene_id] = {
                    "gt_class": gt_class,
                    "frames": scene_frames,
                    "total_frames": total,
                }

    return exported


def build_firetrace_dataset():
    """Build FiftyOne dataset for FireTrace fire origin analysis."""

    print("  Loading results...")
    if not RESULTS_PATH.exists():
        print(f"    ERROR: {RESULTS_PATH} not found")
        return None

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    results_by_scene = {r["scene_id"]: r for r in data.get("results", [])}

    print("  Exporting representative frames...")
    exported = export_representative_frames(results_data=data)

    samples = []
    for scene_id, info in exported.items():
        gt_class = info["gt_class"]
        r = results_by_scene.get(scene_id, {})
        ev = r.get("evaluation", {})
        pred = r.get("prediction", {}) or {}
        reasoning = r.get("reasoning", "")

        for frame_path, frame_idx in info["frames"]:
            sample = fo.Sample(filepath=str(frame_path))

            # Metadata
            sample["scene_id"] = scene_id
            sample["frame_index"] = frame_idx
            sample["total_frames"] = info["total_frames"]

            # Ground truth
            sample["ground_truth"] = fo.Classification(label=gt_class)

            # Prediction
            if ev.get("valid"):
                pred_label = ev.get("pred_class", "UNKNOWN")
                confidence = pred.get("confidence", 0.5)
                sample["prediction"] = fo.Classification(
                    label=pred_label,
                    confidence=confidence,
                )

                # Fire origin as keypoint
                origin_x = pred.get("origin_x")
                origin_y = pred.get("origin_y")
                if origin_x is None or origin_y is None:
                    origin_text = pred.get("fire_origin", "")
                    if origin_text and origin_text != "NONE":
                        origin_x, origin_y = fallback_origin_from_text(origin_text)

                if origin_x is not None and origin_y is not None:
                    sample["fire_origin"] = fo.Keypoint(
                        label="ORIGIN",
                        points=[(origin_x, origin_y)],
                    )

                # Spread arrows as polylines
                arrows = pred.get("spread_arrows", [])
                if not arrows:
                    spread_dir = pred.get("spread_direction", "")
                    if origin_x is not None and spread_dir:
                        arrows = generate_fallback_arrows(origin_x, origin_y, spread_dir)

                if arrows:
                    polylines = []
                    for arrow in arrows:
                        polylines.append(
                            fo.Polyline(
                                label="SPREAD",
                                points=[[(arrow["from_x"], arrow["from_y"]),
                                         (arrow["to_x"], arrow["to_y"])]],
                                closed=False,
                            )
                        )
                    sample["spread_direction"] = fo.Polylines(polylines=polylines)

                # Text fields
                sample["fire_origin_text"] = pred.get("fire_origin", "NONE")
                sample["severity"] = pred.get("severity", "NONE")
                sample["fire_stage"] = pred.get("fire_stage", "NONE")
                sample["urgency"] = pred.get("urgency", "NONE")
                sample["convection"] = pred.get("convection_pattern", "NONE")
                sample["temporal_progression"] = pred.get("temporal_progression", "")[:500]
                sample["physics_reasoning"] = pred.get("physics_reasoning", "")[:500]

                # Evaluation results
                sample["hazard_correct"] = ev.get("hazard_correct", False)
                sample["urgency_correct"] = ev.get("urgency_appropriate", False)

                # Tags
                if ev.get("hazard_correct"):
                    sample.tags.append("hazard_detected")
                else:
                    sample.tags.append("hazard_missed")

                if gt_class == "FLAME":
                    sample.tags.append("flame")
                elif gt_class == "SMOKE":
                    sample.tags.append("smoke")
                else:
                    sample.tags.append("normal")

                # Middle frame tag (for origin visualization)
                if frame_idx == info["total_frames"] // 2:
                    sample.tags.append("representative")
            else:
                sample["prediction"] = fo.Classification(label="NO_PREDICTION")
                sample.tags.append("no_prediction")
                if gt_class == "NORMAL":
                    sample.tags.append("normal")

            # Reasoning (only on middle frame to avoid duplication)
            if frame_idx == info["total_frames"] // 2 and reasoning:
                sample["model_reasoning"] = reasoning[:2000]

            samples.append(sample)

    # Create dataset
    dataset_name = "firetrace_origin"
    dataset = fo.Dataset(dataset_name, overwrite=True)
    dataset.add_samples(samples)
    dataset.persistent = True

    # Add dataset description
    dataset.description = (
        "FireTrace: Fire Origin Tracing with Cosmos-Reason2. "
        "Each sample is a frame from fire surveillance footage. "
        "The 'fire_origin' keypoint shows the predicted ignition point. "
        "The 'spread_direction' polylines show predicted fire/smoke spread. "
        "Filter by tags: flame, smoke, normal, representative, hazard_detected."
    )

    print(f"    Dataset '{dataset_name}' created with {len(dataset)} samples")
    return dataset


def build_firetrace_video_dataset():
    """Build FiftyOne dataset with video samples for temporal analysis."""
    if not RESULTS_PATH.exists():
        return None

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for r in data.get("results", []):
        ev = r.get("evaluation", {})
        pred = r.get("prediction", {}) or {}
        scene_id = r["scene_id"]
        video_path = Path(r.get("video_path", ""))

        if not video_path.exists():
            continue

        sample = fo.Sample(filepath=str(video_path))
        sample["scene_id"] = scene_id
        sample["gt_class"] = r.get("gt_class", "UNKNOWN")
        sample["ground_truth"] = fo.Classification(label=r.get("gt_class", "UNKNOWN"))

        if ev.get("valid"):
            sample["prediction"] = fo.Classification(
                label=ev.get("pred_class", "UNKNOWN"),
                confidence=pred.get("confidence", 0.5),
            )
            sample["fire_origin_text"] = pred.get("fire_origin", "NONE")
            sample["severity"] = pred.get("severity", "NONE")
            sample["fire_stage"] = pred.get("fire_stage", "NONE")
            sample["urgency"] = pred.get("urgency", "NONE")
            sample["spread_direction_text"] = pred.get("spread_direction", "NONE")
            sample["temporal_progression"] = pred.get("temporal_progression", "")[:500]
            sample["physics_reasoning"] = pred.get("physics_reasoning", "")[:500]
            sample["hazard_correct"] = ev.get("hazard_correct", False)

            if ev.get("hazard_correct"):
                sample.tags.append("hazard_detected")
            else:
                sample.tags.append("hazard_missed")
        else:
            sample.tags.append("no_prediction")

        sample.tags.append(r.get("gt_class", "UNKNOWN").lower())
        sample["elapsed_sec"] = r.get("elapsed_sec", 0)
        sample["model_reasoning"] = r.get("reasoning", "")[:2000]

        samples.append(sample)

    dataset = fo.Dataset("firetrace_videos", overwrite=True)
    dataset.add_samples(samples)
    dataset.persistent = True
    dataset.description = (
        "FireTrace video-level results. Each sample is a full scene video "
        "with Cosmos-Reason2 fire detection predictions and physics reasoning."
    )

    print(f"    Dataset 'firetrace_videos' created with {len(dataset)} samples")
    return dataset


def build_origin_overlay_dataset():
    """Build dataset from pre-generated origin visualization images."""
    origin_images = sorted(ORIGIN_DIR.glob("origin_*.jpg"))
    if not origin_images:
        print("    No origin images found. Run visualize_origin.py first.")
        return None

    samples = []
    for img_path in origin_images:
        scene_id = img_path.stem.replace("origin_", "")
        sample = fo.Sample(filepath=str(img_path))
        sample["scene_id"] = scene_id
        sample.tags.append("origin_overlay")
        samples.append(sample)

    # Add temporal strips
    temporal_images = sorted(ORIGIN_DIR.glob("temporal_*.jpg"))
    for img_path in temporal_images:
        scene_id = img_path.stem.replace("temporal_", "")
        sample = fo.Sample(filepath=str(img_path))
        sample["scene_id"] = scene_id
        sample.tags.append("temporal_strip")
        samples.append(sample)

    dataset = fo.Dataset("firetrace_overlays", overwrite=True)
    dataset.add_samples(samples)
    dataset.persistent = True
    dataset.description = (
        "FireTrace annotated images: origin markers and temporal progression strips."
    )

    print(f"    Dataset 'firetrace_overlays' created with {len(dataset)} samples")
    return dataset


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FireTrace FiftyOne Demo")
    parser.add_argument("--launch", action="store_true", help="Launch FiftyOne app")
    parser.add_argument("--port", type=int, default=5151)
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "frames", "videos", "overlays"],
                        help="Which dataset(s) to build")
    args = parser.parse_args()

    print("=" * 60)
    print("FireTrace: FiftyOne Dataset Builder")
    print("=" * 60)

    datasets = {}

    if args.dataset in ("all", "frames"):
        print("\n  Building frame-level dataset (with origin keypoints)...")
        ds = build_firetrace_dataset()
        if ds:
            datasets["firetrace_origin"] = ds

    if args.dataset in ("all", "videos"):
        print("\n  Building video-level dataset...")
        ds = build_firetrace_video_dataset()
        if ds:
            datasets["firetrace_videos"] = ds

    if args.dataset in ("all", "overlays"):
        print("\n  Building origin overlay dataset...")
        ds = build_origin_overlay_dataset()
        if ds:
            datasets["firetrace_overlays"] = ds

    # Summary
    print(f"\n  Available FireTrace datasets:")
    for name in fo.list_datasets():
        if "firetrace" in name:
            ds = fo.load_dataset(name)
            print(f"    - {name}: {len(ds)} samples")

    if args.launch and datasets:
        # Launch with the frame-level dataset (most interesting)
        primary = datasets.get("firetrace_origin") or list(datasets.values())[0]
        print(f"\n  Launching FiftyOne app on port {args.port}...")
        print(f"  Open http://localhost:{args.port} in your browser")
        session = fo.launch_app(primary, port=args.port)
        session.wait()
    else:
        print(f"\n  To view: python pipelines/firetrace_fiftyone.py --launch")
        print(f"  Or in Python:")
        print(f"    import fiftyone as fo")
        print(f"    ds = fo.load_dataset('firetrace_origin')")
        print(f"    session = fo.launch_app(ds)")

    print("\nDone.")


if __name__ == "__main__":
    main()
