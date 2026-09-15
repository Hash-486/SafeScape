"""Map every downloaded source audio file to one of the 5 target classes and write
data/processed/manifest.csv (one row per SOURCE file, not yet windowed/split).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.paths import ROOT, MANIFEST as OUT_CSV

RAW = ROOT / "data" / "raw"
RNG_SEED = 42

# ESC-50 category -> target class. Everything else in ESC-50 becomes ambience.
ESC50_HAZARD_MAP = {
    "glass_breaking": "glass_break",
    "car_horn": "horn_skid",
    "siren": "alarm",
    "clock_alarm": "alarm",
}

# UrbanSound8K class -> target class. ESC-50 caps every category at 40 clips, which
# starves horn_skid/alarm; UrbanSound8K carries 429 car_horn and 929 siren clips, and its
# remaining categories are precisely the urban hard negatives that were being misread as
# horn_skid, so they go in as ambience to sharpen the hazard/normal boundary.
US8K_MAP = {
    "car_horn": "horn_skid",
    "siren": "alarm",
    "air_conditioner": "ambience",
    "children_playing": "ambience",
    "dog_bark": "ambience",
    "drilling": "ambience",
    "engine_idling": "ambience",
    "jackhammer": "ambience",
    "street_music": "ambience",
}
# gun_shot is deliberately dropped, not mapped to ambience: it is a genuine hazard with no
# class in this 5-way taxonomy, and teaching a safety model that gunfire is "normal" is a
# worse failure than simply not training on it.
US8K_EXCLUDE = {"gun_shot"}


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


# UrbanSound8K encodes the label in the filename: <fsID>-<classID>-<occurrence>-<slice>.wav
US8K_CLASS_BY_ID = {
    0: "air_conditioner", 1: "car_horn", 2: "children_playing", 3: "dog_bark",
    4: "drilling", 5: "engine_idling", 6: "gun_shot", 7: "jackhammer",
    8: "siren", 9: "street_music",
}


def load_urbansound8k(ambience_per_category=90):
    """UrbanSound8K (Zenodo 1203745): 8732 labelled 4 s urban clips across 10 classes.

    Labels come from the FILENAME, not metadata/UrbanSound8K.csv. In the distributed
    tar the metadata directory sorts after audio/, so a partially-extracted archive has
    thousands of usable clips but no CSV; parsing the classID field makes the loader work
    on whatever folds actually landed.

    Hazard classes are taken whole (they are the scarce ones). Ambience-mapped categories
    are subsampled: cap_ambience() discards almost all of them downstream anyway, so
    windowing and feature-extracting the surplus would be pure wasted compute.
    """
    base = RAW / "urbansound8k"
    if not base.exists():
        return pd.DataFrame(columns=["filepath", "target_class", "source_dataset", "orig_category"])

    found = {}
    for p in base.rglob("*.wav"):
        fields = p.stem.split("-")
        if len(fields) < 4:
            continue
        try:
            cat = US8K_CLASS_BY_ID[int(fields[1])]
        except (ValueError, KeyError):
            continue
        if cat in US8K_EXCLUDE or cat not in US8K_MAP:
            continue
        found.setdefault(cat, []).append(p)

    rng = np.random.RandomState(RNG_SEED)
    rows = []
    for cat, paths in found.items():
        paths = sorted(paths)
        if US8K_MAP[cat] == "ambience" and len(paths) > ambience_per_category:
            idx = rng.choice(len(paths), size=ambience_per_category, replace=False)
            paths = [paths[i] for i in sorted(idx)]
        for p in paths:
            rows.append({
                "filepath": str(p),
                "target_class": US8K_MAP[cat],
                "source_dataset": "urbansound8k",
                "orig_category": f"us8k_{cat}",
            })
    return pd.DataFrame(rows)


def load_glass_extra(max_seconds=12.0):
    """Extra glass-breaking clips — the one class UrbanSound8K cannot help with.

    data/raw/glass_datasec          : DataSEC (Zenodo 15340689), CC BY 4.0
    data/raw/glass_break_freesound  : Freesound CC0 preview clips (id = freesound.org/s/<id>)
    data/raw/glass_tut              : TUT Rare Sound Events 2017 (Zenodo 401395)

    Long files are DROPPED rather than used. Most DataSEC entries are 60.5 s continuous
    recordings, and some TUT recordings run to ~100 s with a handful of break events
    separated by ambience. Windowed at 1 s/50% a 60 s file yields ~119 windows of which
    only a few contain glass, so admitting them would inject thousands of mislabelled
    ambience windows straight into the scarcest class — actively worse than having less
    data. Per-event onset times exist only in sibling .yaml files the partial extraction
    did not pull, so a duration filter is the honest way to keep single-event clips.
    """
    import soundfile as sf

    rows = []
    for dirname, source in [("glass_datasec", "datasec"),
                            ("glass_break_freesound", "freesound_cc0"),
                            ("glass_tut", "tut_rare_sound")]:
        d = RAW / dirname
        if not d.exists():
            continue
        files = sorted(list(d.glob("*.wav")) + list(d.glob("*.mp3")))
        for p in files:
            try:
                info = sf.info(str(p))
                if info.duration > max_seconds:
                    continue
            except Exception:
                continue
            rows.append({
                "filepath": str(p),
                "target_class": "glass_break",
                "source_dataset": source,
                "orig_category": f"{source}_glassbreak",
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

    parts = [load_esc50(), load_urbansound8k(), load_glass_extra(),
             load_scream_kaggle("scream_kaggle_1"),
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
