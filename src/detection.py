"""Scenario 5: Fire and Smoke Detection (Video-based)

Converts sequential fire/smoke/normal image frames into videos,
then asks Cosmos-Reason2 to analyze the temporal evolution of fire.

Key insight: Tests the model's understanding of combustion physics —
flame propagation patterns over time, smoke dispersion dynamics,
heat convection indicators, fire origin point estimation,
and distinguishing real fire from visually similar phenomena.

Data: fire_detection_aihub — 3 scenes x 360 frames each (flame, smoke, normal).
      Frames are stitched into videos for temporal reasoning.
"""

import cv2
import json
import numpy as np
import tempfile
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference import Reason2Model, parse_json_response


BASE_DIR = Path(r"D:\Program Files\isaacsim\cookbook\firetrace")
DATA_DIR = Path(r"D:\Program Files\isaacsim\cookbook\data\fire_detection_aihub\Sample")
RESULTS_DIR = BASE_DIR / "reports"
VIDEO_DIR = BASE_DIR / "data" / "fire_videos"

RAW_BASE = DATA_DIR / "01.원천데이터" / "화재현상"
LABEL_BASE = DATA_DIR / "02.라벨링데이터" / "화재현상"


SYSTEM_PROMPT = """You are an expert fire safety analyst with deep knowledge of combustion physics.
You can identify fire hazards from video surveillance footage:
- Flame detection: color temperature, shape dynamics, flickering patterns over time
- Smoke detection: density changes, color (white=steam/early, grey=organic, black=petroleum), dispersion patterns
- Fire stage assessment: incipient, growth, fully developed, decay — tracked over time
- False positive awareness: distinguish fire from reflections, sunset, steam, fog
- Fire origin analysis: trace flame spread direction back to ignition point using temporal cues
You understand that flames follow convection physics (upward spread), smoke rises and
accumulates at ceiling level, and fire intensity correlates with fuel load and ventilation.
By analyzing the video temporally, you can estimate the fire origin point from how
flames and smoke spread over successive frames."""

DETECTION_PROMPT = """Watch this surveillance video carefully and analyze it for fire and smoke detection.

Pay attention to how the scene evolves over time — fire spreads, smoke accumulates, and
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


# Map dataset class codes to our labels
CLASS_MAP = {
    "FL": "FLAME",
    "SM": "SMOKE",
    "NONE": "NORMAL",
}

SUBDIRS = {
    "FL": "불꽃",
    "SM": "연기",
    "NONE": "정상",
}


def frames_to_video(frame_dir, output_path, fps=30):
    """Convert sequential JPG frames into an MP4 video.

    Args:
        frame_dir: Directory containing numbered JPG frames.
        output_path: Path for output MP4 file.
        fps: Frames per second for the video.

    Returns:
        Path to created video file, or None on failure.
    """
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        return None

    def _imread_unicode(path):
        """Read image from Unicode path (cv2.imread fails on Korean paths)."""
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Read first frame to get dimensions
    first = _imread_unicode(frames[0])
    if first is None:
        return None
    h, w = first.shape[:2]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame_path in frames:
        img = _imread_unicode(frame_path)
        if img is not None:
            writer.write(img)

    writer.release()
    return output_path if output_path.exists() else None


def get_scene_data():
    """Collect scene-level data: one video per class.

    Returns list of (scene_dir, class_code, scene_id) tuples.
    """
    scenes = []
    for class_code, subdir_name in SUBDIRS.items():
        raw_class_dir = RAW_BASE / subdir_name
        if not raw_class_dir.exists():
            continue

        for scene_dir in sorted(raw_class_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            jpg_dir = scene_dir / "JPG"
            if not jpg_dir.exists():
                continue

            scene_id = scene_dir.name
            n_frames = len(list(jpg_dir.glob("*.jpg")))
            if n_frames == 0:
                continue

            scenes.append((jpg_dir, class_code, scene_id, n_frames))

    return scenes


def evaluate_prediction(predicted, gt_class_code):
    """Evaluate prediction against ground truth."""
    if not predicted:
        return {"valid": False}

    gt_label = CLASS_MAP[gt_class_code]
    gt_has_fire = gt_class_code == "FL"
    gt_has_smoke = gt_class_code == "SM"

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
    class_correct = pred_class == gt_label

    # Fire detection
    fire_correct = pred_fire == gt_has_fire

    # Smoke detection
    if gt_class_code == "SM":
        smoke_correct = pred_smoke
    elif gt_class_code == "NONE":
        smoke_correct = not pred_smoke
    else:
        smoke_correct = True

    # Severity check
    if gt_class_code == "NONE":
        severity_correct = pred_severity in ("NONE", "LOW")
    else:
        severity_correct = pred_severity not in ("NONE",)

    # Binary hazard detection
    gt_hazard = gt_class_code != "NONE"
    pred_hazard = pred_fire or pred_smoke
    hazard_correct = pred_hazard == gt_hazard

    # Fire origin reasoning
    has_origin = pred_origin not in ("NONE", "", None) if gt_hazard else True
    # Spread direction
    has_spread = pred_spread not in ("NONE", "", None) if gt_hazard else True
    # Temporal reasoning (video-specific: should describe progression)
    has_temporal = len(str(pred_temporal)) > 10 if gt_hazard else True
    # Urgency
    if gt_class_code == "FL":
        urgency_appropriate = pred_urgency in ("EVACUATE", "IMMEDIATE_RESPONSE")
    elif gt_class_code == "SM":
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

    return {
        "valid": True,
        "gt_class": gt_label,
        "gt_has_fire": gt_has_fire,
        "gt_has_smoke": gt_has_smoke,
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
        "fire_correct": fire_correct,
        "smoke_correct": smoke_correct,
        "severity_correct": severity_correct,
        "hazard_correct": hazard_correct,
        "has_origin": has_origin,
        "has_coordinates": has_coordinates,
        "has_spread": has_spread,
        "has_temporal": has_temporal,
        "urgency_appropriate": urgency_appropriate,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PhysicsGuard Scenario 5: Fire and Smoke Detection")
    parser.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--video-fps", type=int, default=30,
                        help="FPS for creating videos from frames (default 30)")
    parser.add_argument("--infer-fps", type=int, default=1,
                        help="FPS for sampling frames during inference (default 1)")
    args = parser.parse_args()

    print("=" * 60)
    print("PhysicsGuard Scenario 5: Fire and Smoke Detection (Video)")
    print(f"  Model: {args.model}")
    print(f"  Video FPS: {args.video_fps}, Inference FPS: {args.infer_fps}")
    print("=" * 60)

    # Step 1: Collect scenes
    scenes = get_scene_data()
    print(f"  Scenes found: {len(scenes)}")
    for jpg_dir, cls, sid, n in scenes:
        print(f"    {sid} ({CLASS_MAP[cls]}): {n} frames")

    # Step 2: Create videos from frames
    print(f"\n  Creating videos from frames...")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video_scenes = []

    for jpg_dir, class_code, scene_id, n_frames in scenes:
        video_path = VIDEO_DIR / f"{scene_id}_{class_code}.mp4"
        if not video_path.exists():
            print(f"    {scene_id}: {n_frames} frames -> {video_path.name} ...", end=" ")
            result_path = frames_to_video(jpg_dir, video_path, fps=args.video_fps)
            if result_path:
                print("OK")
            else:
                print("FAILED")
                continue
        else:
            print(f"    {scene_id}: {video_path.name} (cached)")

        video_scenes.append((video_path, class_code, scene_id, n_frames))

    print(f"  Videos ready: {len(video_scenes)}")

    # Step 3: Run inference
    model = Reason2Model(args.model)

    results = []
    total_start = time.time()

    for i, (video_path, gt_class, scene_id, n_frames) in enumerate(video_scenes):
        duration_sec = n_frames / args.video_fps
        print(f"\n  [{i+1}/{len(video_scenes)}] {scene_id} (GT: {CLASS_MAP[gt_class]}, {n_frames} frames, {duration_sec:.0f}s)")

        result = model.infer(
            video_path=video_path,
            user_prompt=DETECTION_PROMPT,
            system_prompt=SYSTEM_PROMPT,
            fps=args.infer_fps,
            max_tokens=1024,
            temperature=0.6,
            use_reasoning=True,
        )

        parsed = parse_json_response(result["answer"])
        if parsed is None:
            parsed = parse_json_response(result["raw_output"])

        eval_result = evaluate_prediction(parsed, gt_class)

        if eval_result.get("valid"):
            pred_cls = eval_result["pred_class"]
            cls_ok = "OK" if eval_result["class_correct"] else "MISS"
            haz_ok = "OK" if eval_result["hazard_correct"] else "MISS"
            origin = eval_result.get("pred_fire_origin", "N/A")
            urgency = eval_result.get("pred_urgency", "N/A")
            print(f"    Pred: {pred_cls}, Class: [{cls_ok}], Hazard: [{haz_ok}]")
            print(f"    Origin: {origin}, Urgency: {urgency} ({result['elapsed_sec']:.1f}s)")
        else:
            print(f"    SKIP (no valid prediction)")

        results.append({
            "scene_id": scene_id,
            "gt_class": CLASS_MAP[gt_class],
            "n_frames": n_frames,
            "video_path": str(video_path),
            "prediction": parsed,
            "reasoning": result["reasoning"],
            "evaluation": eval_result,
            "elapsed_sec": result["elapsed_sec"],
        })

    total_elapsed = time.time() - total_start

    # Metrics
    valid = [r for r in results if r["evaluation"].get("valid")]
    class_ok = sum(1 for r in valid if r["evaluation"]["class_correct"])
    fire_ok = sum(1 for r in valid if r["evaluation"]["fire_correct"])
    smoke_ok = sum(1 for r in valid if r["evaluation"]["smoke_correct"])
    hazard_ok = sum(1 for r in valid if r["evaluation"]["hazard_correct"])
    severity_ok = sum(1 for r in valid if r["evaluation"]["severity_correct"])
    origin_ok = sum(1 for r in valid if r["evaluation"]["has_origin"])
    spread_ok = sum(1 for r in valid if r["evaluation"]["has_spread"])
    temporal_ok = sum(1 for r in valid if r["evaluation"]["has_temporal"])
    urgency_ok = sum(1 for r in valid if r["evaluation"]["urgency_appropriate"])

    # Per-class breakdown
    per_class_stats = {}
    for cls_name in ["FLAME", "SMOKE", "NORMAL"]:
        cls_valid = [r for r in valid if r["gt_class"] == cls_name]
        cls_correct = sum(1 for r in cls_valid if r["evaluation"]["class_correct"])
        per_class_stats[cls_name] = {
            "total": len(cls_valid),
            "correct": cls_correct,
            "accuracy": round(cls_correct / max(len(cls_valid), 1), 4),
        }

    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"  Total scenes: {len(valid)}")
    print(f"  Classification accuracy:   {class_ok}/{len(valid)} = {class_ok/max(len(valid),1):.1%}")
    print(f"  Hazard detection accuracy:  {hazard_ok}/{len(valid)} = {hazard_ok/max(len(valid),1):.1%}")
    print(f"  Fire detection accuracy:    {fire_ok}/{len(valid)} = {fire_ok/max(len(valid),1):.1%}")
    print(f"  Smoke detection accuracy:   {smoke_ok}/{len(valid)} = {smoke_ok/max(len(valid),1):.1%}")
    print(f"  Severity accuracy:          {severity_ok}/{len(valid)} = {severity_ok/max(len(valid),1):.1%}")
    print(f"  Fire origin reasoning:      {origin_ok}/{len(valid)} = {origin_ok/max(len(valid),1):.1%}")
    print(f"  Spread direction provided:  {spread_ok}/{len(valid)} = {spread_ok/max(len(valid),1):.1%}")
    print(f"  Temporal reasoning:         {temporal_ok}/{len(valid)} = {temporal_ok/max(len(valid),1):.1%}")
    print(f"  Urgency accuracy:           {urgency_ok}/{len(valid)} = {urgency_ok/max(len(valid),1):.1%}")
    print(f"\n  Per-class breakdown:")
    for cls_name, stats in per_class_stats.items():
        print(f"    {cls_name:8s}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1%}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = Path(args.model).name.replace("-", "_").lower()
    output_path = args.output or str(RESULTS_DIR / f"results_{model_tag}.json")

    output_data = {
        "scenario": "fire_detection",
        "model": args.model,
        "total_scenes": len(results),
        "total_elapsed_sec": round(total_elapsed, 2),
        "metrics": {
            "total": len(valid),
            "classification_accuracy": round(class_ok / max(len(valid), 1), 4),
            "hazard_detection_accuracy": round(hazard_ok / max(len(valid), 1), 4),
            "fire_detection_accuracy": round(fire_ok / max(len(valid), 1), 4),
            "smoke_detection_accuracy": round(smoke_ok / max(len(valid), 1), 4),
            "severity_accuracy": round(severity_ok / max(len(valid), 1), 4),
            "fire_origin_reasoning_rate": round(origin_ok / max(len(valid), 1), 4),
            "spread_direction_rate": round(spread_ok / max(len(valid), 1), 4),
            "temporal_reasoning_rate": round(temporal_ok / max(len(valid), 1), 4),
            "urgency_accuracy": round(urgency_ok / max(len(valid), 1), 4),
            "per_class": per_class_stats,
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {output_path}")

    model.unload()
    print("\nDone.")


if __name__ == "__main__":
    main()
