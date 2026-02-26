"""FlashBack: Complete pipeline runner.

Runs all FlashBack steps:
1. Visualize fire origin on representative frames
2. Generate FlashBack dashboard (HTML)
3. Build FiftyOne datasets
4. (Optional) Launch FiftyOne app
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def run_step(module, description):
    print(f"\n{'='*60}")
    print(f"  Step: {description}")
    print(f"  Module: {module}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=str(BASE_DIR),
    )

    if result.returncode != 0:
        print(f"\n  WARNING: {module} exited with code {result.returncode}")
        return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FlashBack Pipeline")
    parser.add_argument("--launch", action="store_true", help="Launch FiftyOne after building")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip Cosmos inference (use existing results)")
    args = parser.parse_args()

    print("=" * 60)
    print("FlashBack: Fire Origin Tracing Pipeline")
    print("=" * 60)

    steps = []

    if not args.skip_inference:
        results_path = BASE_DIR / "reports" / "results_combined.json"
        if results_path.exists():
            print(f"\n  Existing results found: {results_path}")
            print(f"  Using --skip-inference (results already available)")
        else:
            print(f"\n  No existing results. Run 'python -m src.core.detection' first.")

    steps.append(("src.core.visualize", "Fire origin visualization"))
    steps.append(("src.ui.dashboard", "Dashboard generation"))
    steps.append(("src.ui.fiftyone_builder", "FiftyOne dataset building"))

    results = {}
    for module, desc in steps:
        ok = run_step(module, desc)
        results[module] = "OK" if ok else "FAILED"

    print(f"\n{'='*60}")
    print("FlashBack Pipeline Summary")
    print(f"{'='*60}")
    for module, status in results.items():
        print(f"  {module:35s} {status}")

    print(f"\nOutputs:")
    print(f"  Origin images:  reports/origin_*.jpg")
    print(f"  Temporal strips: reports/temporal_*.jpg")
    print(f"  Dashboard:      reports/firetrace_dashboard.html")

    if args.launch:
        print(f"\n  Launching FiftyOne app...")
        run_step("src.ui.fiftyone_builder", "FiftyOne app")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
