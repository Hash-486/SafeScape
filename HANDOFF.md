# SafeScape — session handoff (2026-09-15, 20:35 IST)

Temporary working note for continuing in a fresh Claude Code session. Delete once the
review is submitted.

## HOW TO RESTART SO I DON'T ASK FOR PERMISSIONS

The reason you were getting prompts: permission **mode** is per-session and resets on
restart — it is never saved to settings. Your previous sessions were running in a
permissive mode that this one didn't inherit.

Restart with:

```
claude --dangerously-skip-permissions
```

(or start normally and press **shift+tab** until the mode indicator reads bypass /
accept-edits). A scoped project allowlist was also written to
`.claude/settings.local.json` — it covers routine commands here and explicitly denies
`rm -rf`, `git push`, `git reset --hard`, `git clean`.

Then paste: *"Read HANDOFF.md and continue where you left off."*

## HARD DEADLINE

Review presentation due **~01:35 IST tonight (2026-09-16)**.

## STATE: WHAT IS ALREADY DONE AND MUST NOT BE BROKEN

These are the guaranteed deliverables. **Do not overwrite them.** If the improvement
work below fails or runs out of time, this is what gets submitted, and it is complete.

- `docs/SafeScape_Review_Presentation.pptx` — 18 slides, covers all 12 rubric items,
  every slide visually verified by PNG export. Built by `scripts/14_build_presentation.py`.
- `models/exported/` — champion bundle (quantized log-mel CRNN) + `calibration.json`.
- `models/checkpoints/logmel_crnn_best.pt` — champion weights (hidden_size=**128**, not
  the class default 64 — pass `--hidden-size 128` to `10_evaluate.py`).
- `reports/eval_results.md` — all results incl. the calibration section.
- Working demo server, verified end-to-end.

Current champion numbers (held-out test, 2799 windows):
accuracy **84.8%**, macro-F1 **0.642**, distress_call recall **85.6%**,
glass_break P/R **0.283 / 0.895**, horn_skid P/R **0.246 / 0.638**, alarm P/R 0.694 / 0.670.

## STILL-OPEN PLACEHOLDERS IN THE DECK (need the user)

In `scripts/14_build_presentation.py`, top of file: `TEAM` (names + reg. numbers),
`GROUP_NO`, `COURSE`, `DEMO_URL`. Edit and re-run the script to regenerate.
Also unresolved: rubric item 10 asks for an IEEE DataPort URL, but the datasets actually
used are on GitHub/Kaggle/Zenodo. The slide flags this honestly.

## IN-FLIGHT WORK: IMPROVING THE WEAK CLASSES

Goal from the user: "address and improve weaker classes and avoid the normal" —
i.e. fix `glass_break` (P 0.28) and `horn_skid` (P 0.25), and stop the ambience/"normal"
class bleeding into them.

**Root cause (confirmed):** ESC-50 caps every category at 40 clips. So `glass_break`
gets 40 source clips and `horn_skid` gets 40, against 1,334 for ambience. The models
over-fire on the thin classes because they never learn a tight boundary against ambience.

### STATUS UPDATE (21:55) — data acquired, pipeline ready to run

New data is on disk and the loaders are verified against it:

| source | clips | goes to |
|---|---|---|
| UrbanSound8K (partial: folds 1,2,3,10 + part of 4) | 167 car_horn / 435 siren / 630 other | horn_skid / alarm / ambience hard negatives |
| DataSEC glass (Zenodo 15340689) | 40 usable of 110 | glass_break |
| Freesound CC0 glass previews | 33+ usable | glass_break |

Resulting source-clip counts vs. before: **horn_skid 40 → 207 (5×), alarm 80 → 515 (6.4×),
glass_break 40 → 113+ (≈3×)**, plus ~630 urban hard negatives to sharpen the
hazard/ambience boundary.

The UrbanSound8K download was deliberately **stopped at 43%** and the truncated archive
partially extracted (3,865 wavs). Waiting for the remaining folds would have cost ~50 more
minutes for roughly 2× the car_horn/siren counts, which was not worth the schedule risk.
The loader reads labels from the FILENAME (`<fsID>-<classID>-<occ>-<slice>.wav`) rather
than `metadata/UrbanSound8K.csv`, precisely because the metadata directory sorts after
audio/ in the tar and never arrived. To get the rest later, resume with `curl -C -` and
re-extract; the loader will pick up whatever is present with no code change.

**Run the whole rebuild with one command:**

```
bash scripts/run_v2_pipeline.sh          # logs to reports/v2_pipeline.log
```

It is fully isolated: `SAFESCAPE_PROC_DIR=data/processed_v2`, checkpoint
`models/checkpoints/logmel_crnn_v2.pt`, calibration `models/exported/calibration_v2.json`.
The v1 champion, its manifests and its feature cache are never touched.

### 1. UrbanSound8K download — reference (already done, see status above)

```
ls -la data/raw/urbansound8k/UrbanSound8K.tar.gz    # target size: 6,023,741,708 bytes
```

At handoff it was at ~770 MB of 6.0 GB, downloading ~1.1 MB/s (ETA ~21:55 IST).
**The curl was launched from this session's shell and may have died on restart.**
Resume (curl `-C -` continues a partial file, it will not restart from zero):

```
curl -L -C - --retry 15 --retry-all-errors --retry-delay 3 \
  -o data/raw/urbansound8k/UrbanSound8K.tar.gz \
  "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1"
```

Then: `tar -xzf data/raw/urbansound8k/UrbanSound8K.tar.gz -C data/raw/urbansound8k/`
Layout after extract: `UrbanSound8K/audio/fold1..fold10/*.wav` plus
`UrbanSound8K/metadata/UrbanSound8K.csv` (columns include `slice_file_name`, `fold`,
`classID`, `class`).

### 2. Why UrbanSound8K is the right fix

It attacks both halves of the problem at once:

| UrbanSound8K class | clips | map to | why |
|---|---|---|---|
| `car_horn` | 429 | **horn_skid** | 10× more than the 40 we have |
| `siren` | 929 | **alarm** | 23× more than ESC-50's siren |
| `air_conditioner`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `jackhammer`, `street_music` | ~6,000 | **ambience** | these are exactly the hard negatives currently misread as horn_skid — this is the "avoid the normal" half |
| `gun_shot` | 374 | **EXCLUDE** | it is a real hazard with no class in our 5-way taxonomy. Labelling it ambience would teach a safety model that gunfire is normal. Dropping it is the defensible choice — worth saying out loud on the results slide. |

Implement as a `load_urbansound8k()` function in `scripts/02_build_manifest.py`,
following the existing `load_esc50()` pattern (returns a DataFrame with columns
`filepath, target_class, source_dataset, orig_category`), then add it to the `parts`
list in `main()`. Note `cap_ambience(max_ratio=1.75)` will automatically raise the
ambience cap as the hazard classes grow, so the ratio stays sane.

### 3. glass_break still needs a separate source

UrbanSound8K has **no** glass-breaking class, so `glass_break` stays at 40 clips unless
another source is found. A research agent was searching for a downloadable glass-break
dataset (FSD50K "Glass" class, Kaggle, Freesound, Hugging Face) when this session ended —
**that result was lost, re-run the search if needed.** Fallback if no dataset is
obtainable in time: waveform-level augmentation (`scripts/utils/augment.py` already has
`pitch_shift`, `time_stretch`, `add_noise`, `mixup`, currently unused by the
precomputed-feature training path, which only does `spec_augment` + feature noise).

Also worth trying regardless of new data: a **WeightedRandomSampler** for balanced
batches. Training currently relies on class-weighted loss only
(`class_weights()` in `scripts/utils/dataset.py`) — the window *counts* stay imbalanced.

### 4. Pipeline to re-run after new data lands — NON-DESTRUCTIVELY

Write to new paths, keep the current champion intact, and only swap in if the numbers
actually improve:

`06_train.py` already accepts `--ckpt` and `--log-csv`, so the retrain is
non-destructive with no code change — just point it at v2 paths:

```
python scripts/02_build_manifest.py                      # now includes UrbanSound8K
python scripts/03_preprocess_audio.py                    # re-window + re-split
python scripts/04_extract_features.py                    # recompute log-mel cache
python scripts/06_train.py --arch logmel_crnn --hidden-size 128 --epochs 25 \
    --lr 8.56e-4 --batch-size 16 --dropout 0.269 \
    --ckpt models/checkpoints/logmel_crnn_v2.pt \
    --log-csv reports/logmel_crnn_v2_train_log.csv
python scripts/10_evaluate.py --arch logmel_crnn --hidden-size 128 \
    --ckpt models/checkpoints/logmel_crnn_v2.pt --tag logmel_crnn_v2 --cpu-only
python scripts/12_calibrate_logits.py --objective safety \
    --ckpt models/checkpoints/logmel_crnn_v2.pt --out models/exported/calibration_v2.json
```

### 5. DO NOT add a balanced sampler on top of the class-weighted loss

I considered a `WeightedRandomSampler` and then talked myself out of it — recording the
reasoning so it isn't re-attempted blindly.

The weak classes' problem is **low precision, not low recall** (glass_break R is already
0.895, P is 0.283). That low precision is *caused by* over-correction toward the rare
classes: `class_weights()` already reweights the loss by 1/n_c. Stacking a balanced
sampler on top would reweight the rare classes a second time, pushing recall even higher
and precision even lower — the opposite of what is wanted here.

The correct fix is more real data, which UrbanSound8K supplies. Because `class_weights()`
derives from the train split at run time, the weights **auto-relax** as the counts
rebalance:

| class | source clips before | after UrbanSound8K |
|---|---|---|
| horn_skid | 40 | ~469 |
| alarm | 80 | ~1,009 |
| glass_break | 40 | **still 40** ← the remaining outlier |

So horn_skid and alarm precision should improve for free. `glass_break` is the one class
UrbanSound8K cannot help, which is why finding a glass-break source (§3) matters most.

**Guard rail — BACKUPS ALREADY MADE.** `06_train.py` checkpoints to
`models/checkpoints/logmel_crnn_best.pt` and `02/03_*.py` overwrite `data/processed/`,
all of which back the current champion. Safe copies now exist:

| backup | restores |
|---|---|
| `models/checkpoints/logmel_crnn_best_v1.pt` | champion weights |
| `models/exported_v1/` | champion bundle + calibration.json |
| `data/processed/manifest_v1.csv` | source-file manifest |
| `data/processed/windows_manifest_v1.csv` | windowed manifest + splits |

If v2 turns out worse, restore by copying these back over the originals. Note the
feature cache (`data/processed/features/`) is NOT backed up — it is large and fully
regenerable by re-running `04_extract_features.py`.

Compare against the baseline table above. Only if v2 wins: re-export, re-quantize,
re-calibrate, update `reports/eval_results.md`, and re-run
`scripts/14_build_presentation.py` to refresh the results slides.

## SCRIPTS ADDED THIS SESSION

- `scripts/12_calibrate_logits.py` — safety-constrained post-hoc logit calibration
- `scripts/13_make_diagrams.py` — system / CRNN / comparison diagrams
- `scripts/14_build_presentation.py` — builds the whole 18-slide deck
- `scripts/15_make_ui_screenshots.py` — headless-Edge UI screenshots
- `--calibration` flag added to `scripts/10_evaluate.py`
- calibration loading added to `server/inference.py`

## USEFUL ENVIRONMENT NOTES

- Python: `.venv/Scripts/python.exe` (torch 2.11.0+cu128, python-pptx and python-docx
  installed this session).
- The Claude-in-Chrome extension is NOT connected. UI screenshots were taken with
  headless Edge at `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`.
- Slides can be rendered to PNG for visual checking via PowerPoint COM automation —
  see the `export.ps1` approach (`$pres.Export($dir, "PNG", 1600, 900)`).
- Demo server: `.venv/Scripts/python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8124`
  (`Predictor` is a lazy singleton — restart the server to pick up a new `calibration.json`).
- A git repo exists (`origin` = github.com/Hash-486/SafeScape). Nothing from this session
  has been committed. Do not push without asking.
