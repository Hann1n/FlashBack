"""FlashBack: Inference on new fire detection datasets.

Processes videos from:
1. archive.zip - 3 sample videos with per-frame bounding box annotations (markup.json)
2. archive (1).zip - 5 FID/FOD surveillance videos

Uses Cosmos-Reason2 for fire origin tracing and compares
predicted origin against ground truth bounding box centers.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from inference import Reason2Model, parse_json_response

MAX_VIDEO_DURATION = 30  # seconds - clip longer videos to this duration
TARGET_WIDTH = 640       # resize for faster processing

BASE_DIR = Path(r"D:\Program Files\isaacsim\cookbook\firetrace")
DATASET_DIR = BASE_DIR / "data" / "fire_dataset"
RESULTS_DIR = BASE_DIR / "reports"
MARKUP_PATH = DATASET_DIR / "markup.json"

# Same prompts as fire_detection.py
SYSTEM_PROMPT = """You are an expert fire safety analyst with deep knowledge of combustion physics.
You can identify fire hazards from video surveillance footage:
- Flame detection: color temperature, shape dynamics, flickering patterns over time
- Smoke detection: density changes, color (white=steam/early, grey=organic, black=petroleum), dispersion patterns
- Fire stage assessment: incipient, growth, fully developed, decay - tracked over time
- False positive awareness: distinguish fire from reflections, sunset, steam, fog
- Fire origin analysis: trace flame spread direction back to ignition point using temporal cues
You understand that flames follow convection physics (upward spread), smoke rises and
accumulates at ceiling level, and fire intensity correlates with fuel load and ventilation.
By analyzing the video temporally, you can estimate the fire origin point from how
flames and smoke spread over successive frames."""

DETECTION_PROMPT = """Watch this surveillance video carefully and analyze it for fire and smoke detection.

Pay attention to how the scene evolves over time - fire spreads, smoke accumulates, and
these temporal dynamics reveal the fire's origin and severity.

Determine:
1. **Classification**: Is this video showing flame, smoke, or a normal (no fire) scene?
2. **Fire Indicators**: What visual evidence of fire/smoke do you see across the frames?
3. **Temporal Evolution**: How does the fire/smoke change over time in the video?
4. **Fire Origin**: Based on the spread pattern across frames, estimate where the fire originated.
   Express the origin as BOTH a text description AND normalized (x, y) coordinates where
   (0.0, 0.0) = top-left corner and (1.0, 1.0) = bottom-right corner of the frame.
5. **Spread Arrows**: Provide 1-3 arrows showing fire/smoke spread direction as (from_x, from_y) -> (to_x, to_y) coordinate pairs.
6. **Urgency**: How urgently does this require emergency response?

OUTPUT FORMAT (JSON only, no extra text):
{{
  "classification": "FLAME|SMOKE|NORMAL",
  "fire_detected": true/false,
  "smoke_detected": true/false,
  "severity": "NONE|LOW|MEDIUM|HIGH|CRITICAL",
  "confidence": 0.0-1.0,
  "fire_stage": "NONE|INCIPIENT|GROWTH|FULLY_DEVELOPED|DECAY",
  "fire_origin": "estimated location/region where fire originated (e.g. bottom-left, center, near equipment) or NONE",
  "origin_x": 0.0-1.0,
  "origin_y": 0.0-1.0,
  "spread_arrows": [
    {{"from_x": 0.0-1.0, "from_y": 0.0-1.0, "to_x": 0.0-1.0, "to_y": 0.0-1.0}}
  ],
  "spread_direction": "NONE|UP|LEFT|RIGHT|OUTWARD|MULTI_DIRECTIONAL",
  "convection_pattern": "describe observed heat/smoke flow pattern over time or NONE",
  "temporal_progression": "describe how fire/smoke evolved across the video frames",
  "urgency": "NONE|MONITOR|EVACUATE|IMMEDIATE_RESPONSE",
  "visual_observations": "describe what fire/smoke indicators you observe",
  "physics_reasoning": "explain the combustion physics behind your assessment, including how temporal spread patterns reveal the fire origin point"
}}"""


def load_markup():
    """Load markup.json and compute ground truth info per video."""
    if not MARKUP_PATH.exists():
        return {}

    with open(MARKUP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_info = {}
    for video_name, frames in data.items():
        # Determine GT class from first annotated object
        fire_frames = []
        smoke_frames = []
        all_objects = []

        for frame in frames:
            for obj in frame.get("objects", []):
                all_objects.append(obj)
                if obj["class"] == "fire":
                    fire_frames.append(frame)
                elif obj["class"] == "smoke":
                    smoke_frames.append(frame)

        # Determine primary class
        if fire_frames:
            gt_class = "FLAME"
        elif smoke_frames:
            gt_class = "SMOKE"
        else:
            gt_class = "NORMAL"

        # Compute GT origin from first fire/smoke detection
        gt_origin_x, gt_origin_y = None, None
        first_detection_frames = fire_frames or smoke_frames
        if first_detection_frames:
            first_frame = first_detection_frames[0]
            w = first_frame.get("width", 1280)
            h = first_frame.get("height", 720)
            first_obj = first_frame["objects"][0]
            # Center of first bounding box = GT origin
            cx = (first_obj["x1"] + first_obj["x2"]) / 2.0
            cy = (first_obj["y1"] + first_obj["y2"]) / 2.0
            gt_origin_x = round(cx / w, 4)
            gt_origin_y = round(cy / h, 4)

        gt_info[video_name] = {
            "gt_class": gt_class,
            "total_frames": len(frames),
            "fire_frame_count": len(fire_frames),
            "smoke_frame_count": len(smoke_frames),
            "total_annotations": len(all_objects),
            "gt_origin_x": gt_origin_x,
            "gt_origin_y": gt_origin_y,
        }

    return gt_info


def preprocess_video(src_path, dst_dir, max_duration=MAX_VIDEO_DURATION, target_w=TARGET_WIDTH):
    """Clip and resize video for faster VLM inference.

    Returns path to preprocessed video (or original if short enough).
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / f"{Path(src_path).stem}_prep.mp4"

    if dst_path.exists():
        return dst_path

    cap = cv2.VideoCapture(str(src_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / max(fps, 1)
    cap.release()

    # Build ffmpeg command: clip to max_duration, resize to target_w maintaining aspect ratio
    scale_filter = f"scale={target_w}:-2"
    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-t", str(max_duration),
        "-vf", scale_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",  # no audio
        str(dst_path),
    ]

    print(f"      Preprocessing: {duration:.0f}s -> {min(duration, max_duration):.0f}s, {w}x{h} -> {target_w}x...")
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        print(f"      ffmpeg error: {result.stderr.decode()[-200:]}")
        return src_path  # fallback to original

    return dst_path


def collect_videos():
    """Collect all video files to process."""
    videos = []
    prep_dir = DATASET_DIR / "preprocessed"

    # From archive.zip samples
    for part_dir in sorted(DATASET_DIR.glob("samples/part*")):
        for mp4 in sorted(part_dir.glob("*.mp4")):
            video_name = mp4.stem  # e.g. bucket11, roomfire41, printer31
            prep_path = preprocess_video(mp4, prep_dir)
            videos.append({
                "path": prep_path,
                "original_path": mp4,
                "name": video_name,
                "source": "archive_samples",
            })

    # From archive (1).zip - FID/FOD videos
    for mp4 in sorted(DATASET_DIR.glob("*.mp4")):
        video_name = mp4.stem
        # Determine GT class from filename
        # FID = Fire Indoor Detection, Y = fire present
        # FOD = Fire Outdoor Detection, N = no fire
        if video_name.startswith("FID") and "_Y_" in video_name:
            gt_class_hint = "FLAME"
        elif video_name.startswith("FOD") and "_N_" in video_name:
            gt_class_hint = "NORMAL"
        else:
            gt_class_hint = "UNKNOWN"

        prep_path = preprocess_video(mp4, prep_dir)
        videos.append({
            "path": prep_path,
            "original_path": mp4,
            "name": video_name,
            "source": "fid_fod",
            "gt_class_hint": gt_class_hint,
        })

    return videos


def evaluate_prediction(predicted, gt_class, gt_origin_x=None, gt_origin_y=None):
    """Evaluate prediction against ground truth."""
    if not predicted:
        return {"valid": False}

    gt_has_fire = gt_class == "FLAME"
    gt_has_smoke = gt_class == "SMOKE"

    pred_class = predicted.get("classification", "UNKNOWN").upper()
    pred_fire = predicted.get("fire_detected", False)
    pred_smoke = predicted.get("smoke_detected", False)
    pred_severity = predicted.get("severity", "NONE")
    pred_origin = predicted.get("fire_origin", "NONE")
    pred_origin_x = predicted.get("origin_x", None)
    pred_origin_y = predicted.get("origin_y", None)
    pred_spread_arrows = predicted.get("spread_arrows", [])
    pred_spread = predicted.get("spread_direction", "NONE")
    pred_convection = predicted.get("convection_pattern", "NONE")
    pred_temporal = predicted.get("temporal_progression", "")
    pred_urgency = predicted.get("urgency", "NONE")

    # Classification accuracy
    class_correct = pred_class == gt_class

    # Binary hazard detection
    gt_hazard = gt_class != "NORMAL"
    pred_hazard = pred_fire or pred_smoke
    hazard_correct = pred_hazard == gt_hazard

    # Severity check
    if gt_class == "NORMAL":
        severity_correct = pred_severity in ("NONE", "LOW")
    else:
        severity_correct = pred_severity not in ("NONE",)

    # Fire origin reasoning
    has_origin = pred_origin not in ("NONE", "", None) if gt_hazard else True
    has_spread = pred_spread not in ("NONE", "", None) if gt_hazard else True
    has_temporal = len(str(pred_temporal)) > 10 if gt_hazard else True

    # Urgency
    if gt_class == "FLAME":
        urgency_appropriate = pred_urgency in ("EVACUATE", "IMMEDIATE_RESPONSE")
    elif gt_class == "SMOKE":
        urgency_appropriate = pred_urgency in ("MONITOR", "EVACUATE", "IMMEDIATE_RESPONSE")
    else:
        urgency_appropriate = pred_urgency in ("NONE", "MONITOR")

    # Coordinate validation
    has_coordinates = (
        pred_origin_x is not None and pred_origin_y is not None
        and isinstance(pred_origin_x, (int, float))
        and isinstance(pred_origin_y, (int, float))
        and 0.0 <= pred_origin_x <= 1.0
        and 0.0 <= pred_origin_y <= 1.0
    ) if gt_hazard else True

    # Origin distance (if both GT and pred coordinates exist)
    origin_distance = None
    if (gt_origin_x is not None and gt_origin_y is not None
            and pred_origin_x is not None and pred_origin_y is not None):
        origin_distance = round(
            ((pred_origin_x - gt_origin_x) ** 2 + (pred_origin_y - gt_origin_y) ** 2) ** 0.5, 4
        )

    return {
        "valid": True,
        "gt_class": gt_class,
        "pred_class": pred_class,
        "pred_fire": pred_fire,
        "pred_smoke": pred_smoke,
        "pred_severity": pred_severity,
        "pred_fire_origin": pred_origin,
        "pred_origin_x": pred_origin_x,
        "pred_origin_y": pred_origin_y,
        "pred_spread_arrows": pred_spread_arrows,
        "pred_spread_direction": pred_spread,
        "pred_convection_pattern": pred_convection,
        "pred_temporal_progression": pred_temporal,
        "pred_urgency": pred_urgency,
        "class_correct": class_correct,
        "hazard_correct": hazard_correct,
        "severity_correct": severity_correct,
        "has_origin": has_origin,
        "has_coordinates": has_coordinates,
        "has_spread": has_spread,
        "has_temporal": has_temporal,
        "urgency_appropriate": urgency_appropriate,
        "origin_distance": origin_distance,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FlashBack: New Dataset Inference")
    parser.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos that already have results")
    args = parser.parse_args()

    print("=" * 60)
    print("FlashBack: New Dataset Inference")
    print(f"  Model: {args.model}")
    print(f"  Inference FPS: {args.fps}")
    print("=" * 60)

    # Load ground truth from markup.json
    markup_gt = load_markup()
    print(f"\n  Markup annotations: {len(markup_gt)} videos")
    for name, info in list(markup_gt.items())[:5]:
        print(f"    {name}: {info['gt_class']}, "
              f"fire={info['fire_frame_count']}, smoke={info['smoke_frame_count']}, "
              f"origin=({info['gt_origin_x']}, {info['gt_origin_y']})")
    if len(markup_gt) > 5:
        print(f"    ... and {len(markup_gt) - 5} more")

    # Collect videos
    videos = collect_videos()
    print(f"\n  Videos to process: {len(videos)}")
    for v in videos:
        size_mb = v["path"].stat().st_size / (1024 * 1024)
        print(f"    {v['name']} ({v['source']}, {size_mb:.1f} MB)")

    # Check existing results
    output_path = RESULTS_DIR / "results_new_dataset.json"
    existing_results = []
    existing_scenes = set()
    if args.skip_existing and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            existing_results = existing_data.get("results", [])
            existing_scenes = {r["scene_id"] for r in existing_results}
        print(f"  Existing results: {len(existing_scenes)} scenes")

    # Load model
    model = Reason2Model(args.model)

    results = list(existing_results)
    total_start = time.time()

    for i, video_info in enumerate(videos):
        video_path = video_info["path"]
        video_name = video_info["name"]

        if video_name in existing_scenes:
            print(f"\n  [{i+1}/{len(videos)}] {video_name} - SKIPPED (existing)")
            continue

        # Determine GT class
        if video_name in markup_gt:
            gt_class = markup_gt[video_name]["gt_class"]
            gt_origin_x = markup_gt[video_name]["gt_origin_x"]
            gt_origin_y = markup_gt[video_name]["gt_origin_y"]
        else:
            gt_class = video_info.get("gt_class_hint", "UNKNOWN")
            gt_origin_x = None
            gt_origin_y = None

        print(f"\n  [{i+1}/{len(videos)}] {video_name} (GT: {gt_class})")

        try:
            result = model.infer(
                video_path=video_path,
                user_prompt=DETECTION_PROMPT,
                system_prompt=SYSTEM_PROMPT,
                fps=args.fps,
                max_tokens=args.max_tokens,
                temperature=0.6,
                use_reasoning=True,
            )

            parsed = parse_json_response(result["answer"])
            if parsed is None:
                parsed = parse_json_response(result["raw_output"])

            eval_result = evaluate_prediction(parsed, gt_class, gt_origin_x, gt_origin_y)

            if eval_result.get("valid"):
                pred_cls = eval_result["pred_class"]
                haz_ok = "OK" if eval_result["hazard_correct"] else "MISS"
                origin_dist = eval_result.get("origin_distance")
                origin_str = f"dist={origin_dist:.3f}" if origin_dist is not None else "no GT"
                print(f"    Pred: {pred_cls}, Hazard: [{haz_ok}], Origin: {origin_str}")
                print(f"    Time: {result['elapsed_sec']:.1f}s")
            else:
                print(f"    SKIP (no valid prediction)")

            results.append({
                "scene_id": video_name,
                "gt_class": gt_class,
                "gt_origin_x": gt_origin_x,
                "gt_origin_y": gt_origin_y,
                "video_path": str(video_path),
                "source": video_info["source"],
                "prediction": parsed,
                "reasoning": result["reasoning"],
                "evaluation": eval_result,
                "elapsed_sec": result["elapsed_sec"],
            })

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "scene_id": video_name,
                "gt_class": gt_class,
                "video_path": str(video_path),
                "source": video_info["source"],
                "prediction": None,
                "reasoning": "",
                "evaluation": {"valid": False, "error": str(e)},
                "elapsed_sec": 0,
            })

        # Save intermediate results after each video
        _save_results(results, output_path, args.model, time.time() - total_start)

    total_elapsed = time.time() - total_start

    # Final save
    _save_results(results, output_path, args.model, total_elapsed)

    # Print summary
    valid = [r for r in results if r["evaluation"].get("valid")]
    hazard_ok = sum(1 for r in valid if r["evaluation"]["hazard_correct"])
    origin_ok = sum(1 for r in valid if r["evaluation"]["has_origin"])
    coord_ok = sum(1 for r in valid if r["evaluation"]["has_coordinates"])
    origin_dists = [r["evaluation"]["origin_distance"] for r in valid
                    if r["evaluation"].get("origin_distance") is not None]

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"  Total videos: {len(results)}")
    print(f"  Valid predictions: {len(valid)}")
    print(f"  Hazard detection: {hazard_ok}/{len(valid)}")
    print(f"  Origin reasoning: {origin_ok}/{len(valid)}")
    print(f"  Has coordinates: {coord_ok}/{len(valid)}")
    if origin_dists:
        print(f"  Origin distance (mean): {np.mean(origin_dists):.4f}")
        print(f"  Origin distance (min):  {min(origin_dists):.4f}")
        print(f"  Origin distance (max):  {max(origin_dists):.4f}")
    print(f"  Total time: {total_elapsed:.0f}s")
    print(f"  Results: {output_path}")

    model.unload()

    # Also merge with existing S5 results
    merge_all_results()

    print("\nDone.")


def _save_results(results, output_path, model_name, elapsed):
    """Save intermediate results."""
    valid = [r for r in results if r["evaluation"].get("valid")]
    hazard_ok = sum(1 for r in valid if r["evaluation"]["hazard_correct"])
    origin_ok = sum(1 for r in valid if r["evaluation"]["has_origin"])
    coord_ok = sum(1 for r in valid if r["evaluation"]["has_coordinates"])
    temporal_ok = sum(1 for r in valid if r["evaluation"]["has_temporal"])
    urgency_ok = sum(1 for r in valid if r["evaluation"]["urgency_appropriate"])
    class_ok = sum(1 for r in valid if r["evaluation"]["class_correct"])
    severity_ok = sum(1 for r in valid if r["evaluation"]["severity_correct"])
    spread_ok = sum(1 for r in valid if r["evaluation"]["has_spread"])

    n = max(len(valid), 1)

    output_data = {
        "scenario": "fire_detection_new",
        "model": model_name,
        "total_scenes": len(results),
        "total_elapsed_sec": round(elapsed, 2),
        "metrics": {
            "total": len(valid),
            "classification_accuracy": round(class_ok / n, 4),
            "hazard_detection_accuracy": round(hazard_ok / n, 4),
            "severity_accuracy": round(severity_ok / n, 4),
            "fire_origin_reasoning_rate": round(origin_ok / n, 4),
            "coordinate_accuracy_rate": round(coord_ok / n, 4),
            "spread_direction_rate": round(spread_ok / n, 4),
            "temporal_reasoning_rate": round(temporal_ok / n, 4),
            "urgency_accuracy": round(urgency_ok / n, 4),
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def merge_all_results():
    """Merge original S5 results with new dataset results into a combined file."""
    combined_results = []

    # Load original S5 results
    s5_path = RESULTS_DIR / "results_cosmos_reason2_2b.json"
    if s5_path.exists():
        with open(s5_path, "r", encoding="utf-8") as f:
            s5_data = json.load(f)
        for r in s5_data.get("results", []):
            r["source"] = "aihub_fire"
            combined_results.append(r)
        print(f"  Merged {len(s5_data.get('results', []))} original S5 results")

    # Load new dataset results
    new_path = RESULTS_DIR / "results_new_dataset.json"
    if new_path.exists():
        with open(new_path, "r", encoding="utf-8") as f:
            new_data = json.load(f)
        for r in new_data.get("results", []):
            combined_results.append(r)
        print(f"  Merged {len(new_data.get('results', []))} new dataset results")

    # Compute combined metrics
    valid = [r for r in combined_results if r.get("evaluation", {}).get("valid")]
    n = max(len(valid), 1)
    hazard_ok = sum(1 for r in valid if r["evaluation"].get("hazard_correct"))
    origin_ok = sum(1 for r in valid if r["evaluation"].get("has_origin"))
    temporal_ok = sum(1 for r in valid if r["evaluation"].get("has_temporal"))
    urgency_ok = sum(1 for r in valid if r["evaluation"].get("urgency_appropriate"))
    class_ok = sum(1 for r in valid if r["evaluation"].get("class_correct"))
    severity_ok = sum(1 for r in valid if r["evaluation"].get("severity_correct"))
    spread_ok = sum(1 for r in valid if r["evaluation"].get("has_spread"))

    combined_data = {
        "scenario": "firetrace_combined",
        "model": "nvidia/Cosmos-Reason2-2B",
        "total_scenes": len(combined_results),
        "metrics": {
            "total": len(valid),
            "classification_accuracy": round(class_ok / n, 4),
            "hazard_detection_accuracy": round(hazard_ok / n, 4),
            "severity_accuracy": round(severity_ok / n, 4),
            "fire_origin_reasoning_rate": round(origin_ok / n, 4),
            "spread_direction_rate": round(spread_ok / n, 4),
            "temporal_reasoning_rate": round(temporal_ok / n, 4),
            "urgency_accuracy": round(urgency_ok / n, 4),
        },
        "results": combined_results,
    }

    combined_path = RESULTS_DIR / "results_combined.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    print(f"  Combined results: {combined_path} ({len(combined_results)} total)")


if __name__ == "__main__":
    main()
