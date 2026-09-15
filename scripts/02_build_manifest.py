"""Map every downloaded source audio file to one of the 5 target classes and write
data/processed/manifest.csv (one row per SOURCE file, not yet windowed/split).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT_CSV = ROOT / "data" / "processed" / "manifest.csv"
RNG_SEED = 42

# ESC-50 category -> target class. Everything else in ESC-50 becomes ambience.
ESC50_HAZARD_MAP = {
    "glass_breaking": "glass_break",
    "car_horn": "horn_skid",
    "siren": "alarm",
    "clock_alarm": "alarm",
}


def load_esc50():
    meta_path = RAW / "esc50" / "ESC-50-master" / "meta" / "esc50.csv"
    audio_dir = RAW / "esc50" / "ESC-50-master" / "audio"
    df = pd.read_csv(meta_path)
    rows = []
    for _, r in df.iterrows():
        cls = ESC50_HAZARD_MAP.get(r["category"], "ambience")
        rows.append({
            "filepath": str(audio_dir / r["filename"]),
            "target_class": cls,
            "source_dataset": "esc50",
            "orig_category": r["category"],
        })
    return pd.DataFrame(rows)


def load_scream_kaggle(dataset_dirname="scream_kaggle_1"):
    """Scans the Kaggle scream-detection dataset. Handles either a flat/labelled-folder
    layout (folder name containing 'scream'/'positive'/'1' -> distress_call, else ambience)
    or a metadata CSV if the dataset ships one. Falls back to treating all audio as
    distress_call if no clear negative-class signal is found (dataset is scream-focused)."""
    ds_dir = RAW / dataset_dirname
    if not ds_dir.exists():
        return pd.DataFrame(columns=["filepath", "target_class", "source_dataset", "orig_category"])

    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = [p for p in ds_dir.rglob("*") if p.suffix.lower() in audio_exts]

    rows = []
    for p in audio_files:
        parts_lower = [part.lower() for part in p.parts]
        joined = " ".join(parts_lower)
        if any(neg in joined for neg in ["non_scream", "non-scream", "negative", "not_scream", "no_scream"]):
            cls = "ambience"
        elif any(pos in joined for pos in ["scream", "positive", "distress"]):
            cls = "distress_call"
        else:
            # dataset is scream-detection-focused; default unlabelled audio to distress_call
            cls = "distress_call"
        rows.append({
            "filepath": str(p),
            "target_class": cls,
            "source_dataset": dataset_dirname,
            "orig_category": p.parent.name,
        })
    return pd.DataFrame(rows)


def load_ravdess_supplement(max_clips=None):
    """Distress-adjacent supplement only (emotional speech, not literal screaming).
    Emotion code 05=angry, 06=fearful from filename field 3."""
    ds_dir = RAW / "ravdess"
    if not ds_dir.exists():
        return pd.DataFrame(columns=["filepath", "target_class", "source_dataset", "orig_category"])
    rows = []
    for p in ds_dir.rglob("*.wav"):
        fields = p.stem.split("-")
        if len(fields) < 3:
            continue
        emotion = fields[2]
        if emotion in ("05", "06"):
            rows.append({
                "filepath": str(p),
                "target_class": "distress_call",
                "source_dataset": "ravdess_supplement",
                "orig_category": f"emotion_{emotion}",
            })
    df = pd.DataFrame(rows)
    if max_clips and len(df) > max_clips:
        df = df.sample(n=max_clips, random_state=RNG_SEED)
    return df


def cap_ambience(df, max_ratio=1.75):
    hazard_counts = df[df["target_class"] != "ambience"]["target_class"].value_counts()
    if hazard_counts.empty:
        return df
    cap = int(hazard_counts.max() * max_ratio)
    amb = df[df["target_class"] == "ambience"]
    other = df[df["target_class"] != "ambience"]
    if len(amb) > cap:
        # sample evenly across ambience source categories for diversity
        amb = amb.groupby("orig_category", group_keys=False).apply(
            lambda g: g.sample(n=max(1, int(cap * len(g) / len(amb))), random_state=RNG_SEED)
        )
        if len(amb) > cap:
            amb = amb.sample(n=cap, random_state=RNG_SEED)
    return pd.concat([other, amb], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ambience-max-ratio", type=float, default=1.75)
    ap.add_argument("--ravdess-max-clips", type=int, default=200)
    ap.add_argument("--use-ravdess-supplement", action="store_true", default=True)
    args = ap.parse_args()

    parts = [load_esc50(), load_scream_kaggle("scream_kaggle_1"),
             load_scream_kaggle("scream_kaggle_2"), load_scream_kaggle("scream_kaggle_3")]
    if args.use_ravdess_supplement:
        parts.append(load_ravdess_supplement(max_clips=args.ravdess_max_clips))

    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df = df.drop_duplicates(subset=["filepath"]).reset_index(drop=True)

    print("raw counts before ambience cap:\n", df["target_class"].value_counts())
    df = cap_ambience(df, max_ratio=args.ambience_max_ratio)
    print("\nfinal counts after ambience cap:\n", df["target_class"].value_counts())

    for cls in ["distress_call", "glass_break", "horn_skid", "alarm", "ambience"]:
        assert (df["target_class"] == cls).sum() > 0, f"class {cls} has ZERO source files"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(df)} source files)")


if __name__ == "__main__":
    main()
