"""FireTrace: Complete pipeline runner.

Runs all FireTrace steps:
1. Visualize fire origin on representative frames
2. Generate FireTrace dashboard (HTML)
3. Build FiftyOne datasets
4. (Optional) Launch FiftyOne app
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(r"D:\Program Files\isaacsim\cookbook\firetrace")


def run_step(script, description):
    print(f"\n{'='*60}")
    print(f"  Step: {description}")
    print(f"  Script: {script}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, str(BASE_DIR / script)],
        cwd=str(BASE_DIR),
    )

    if result.returncode != 0:
        print(f"\n  WARNING: {script} exited with code {result.returncode}")
        return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FireTrace Pipeline")
    parser.add_argument("--launch", action="store_true", help="Launch FiftyOne after building")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip Cosmos inference (use existing results)")
    args = parser.parse_args()

    print("=" * 60)
    print("FireTrace: Fire Origin Tracing Pipeline")
    print("=" * 60)

    steps = []

    if not args.skip_inference:
        results_path = BASE_DIR / "reports" / "results_combined.json"
        if results_path.exists():
            print(f"\n  Existing results found: {results_path}")
            print(f"  Using --skip-inference (results already available)")
        else:
            print(f"\n  No existing results. Run fire_detection.py separately first.")

    steps.append(("visualize_origin.py", "Fire origin visualization"))
    steps.append(("firetrace_dashboard.py", "Dashboard generation"))
    steps.append(("firetrace_fiftyone.py", "FiftyOne dataset building"))

    results = {}
    for script, desc in steps:
        ok = run_step(script, desc)
        results[script] = "OK" if ok else "FAILED"

    print(f"\n{'='*60}")
    print("FireTrace Pipeline Summary")
    print(f"{'='*60}")
    for script, status in results.items():
        print(f"  {script:35s} {status}")

    print(f"\nOutputs:")
    print(f"  Origin images:  reports/origin_*.jpg")
    print(f"  Temporal strips: reports/temporal_*.jpg")
    print(f"  Dashboard:      reports/firetrace_dashboard.html")
    print(f"  FiftyOne:       firetrace_origin, firetrace_videos, firetrace_overlays")

    if args.launch:
        print(f"\n  Launching FiftyOne app...")
        run_step("firetrace_fiftyone.py --launch", "FiftyOne app")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
