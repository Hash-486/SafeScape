"""Single switchable root for all derived data (manifests, windows, feature cache).

Window files are named by manifest row index and the feature cache is keyed on that
filename, so a manifest rebuilt over a different set of source files would silently pair
new audio with stale cached features. Rather than risk that, a new data revision gets a
whole new tree:

    SAFESCAPE_PROC_DIR=data/processed_v2 python scripts/03_preprocess_audio.py

Unset, everything resolves to data/processed exactly as before.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

PROC_DIR = Path(os.environ.get("SAFESCAPE_PROC_DIR") or (ROOT / "data" / "processed"))
if not PROC_DIR.is_absolute():
    PROC_DIR = ROOT / PROC_DIR

MANIFEST = PROC_DIR / "manifest.csv"
WINDOWS_MANIFEST = PROC_DIR / "windows_manifest.csv"
WINDOWS_DIR = PROC_DIR / "windows"
FEATURES_DIR = PROC_DIR / "features"
