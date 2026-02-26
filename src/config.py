"""FlashBack: Centralized path configuration.

All paths are relative to the project root (the directory containing run.py).
"""

from pathlib import Path

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core directories
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEMO_DIR = PROJECT_ROOT / "demo"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Data subdirectories
FIRE_VIDEOS_DIR = DATA_DIR / "fire_videos"
FIRE_DATASET_DIR = DATA_DIR / "fire_dataset"
FIRE_FRAMES_DIR = DATA_DIR / "fire_frames"

# Results files
RESULTS_COMBINED = REPORTS_DIR / "results_combined.json"
RESULTS_2B = REPORTS_DIR / "results_cosmos_reason2_2b.json"
RESULTS_NEW = REPORTS_DIR / "results_new_dataset.json"
DASHBOARD_HTML = REPORTS_DIR / "firetrace_dashboard.html"

# Demo output
DEMO_VIDEO = DEMO_DIR / "demo.mp4"

# Markup annotations (new dataset)
MARKUP_PATH = FIRE_DATASET_DIR / "markup.json"

# Dataset class mappings
CLASS_MAP = {"FL": "FLAME", "SM": "SMOKE", "NONE": "NORMAL"}
SUBDIRS = {"FL": "flame", "SM": "smoke", "NONE": "normal"}
CLASS_TO_CODE = {"FLAME": "FL", "SMOKE": "SM", "NORMAL": "NONE"}
