# Data acquisition log

Commands actually run to populate `data/raw/` (data-provenance record for the submission).

## ESC-50 (glass_breaking, siren, car_horn, clock_alarm + ~43 ambience categories)
```
curl -L --retry 15 --retry-all-errors --retry-delay 2 --speed-time 30 --speed-limit 1000 \
  -o data/raw/esc50.zip https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master
# unzipped to data/raw/esc50/ESC-50-master/{audio,meta/esc50.csv}
```
Source: https://github.com/karolpiczak/ESC-50 (CC BY-NC 3.0). 2000 clips, 50 categories, 40 clips/category, 5s each.

## Kaggle: Human Screaming Detection Dataset (primary distress_call source)
Auth: Kaggle CLI 2.2.4 token-based auth (`~/.kaggle/access_token`), not the legacy `kaggle.json` username+key format.
```
kaggle datasets download -d whats2000/human-screaming-detection-dataset -p data/raw/scream_kaggle_1 --unzip
```

## RAVDESS speech (Zenodo, no login) — distress-adjacent supplement, NOT literal screaming
```
curl -L --retry 10 --retry-all-errors --retry-delay 2 \
  -o data/raw/ravdess/Audio_Speech_Actors_01-24.zip \
  "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
```
Source: https://zenodo.org/record/1188976 (CC BY-NC-SA 4.0). Filename encodes
`Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor`; emotion codes
05=angry, 06=fearful used as a distress-adjacent supplement only — this is *emotional
speech*, not literal screaming, and is flagged as such in the eval report if it ends up
load-bearing rather than purely supplementary.

## UrbanSound8K — not used
ESC-50 alone already covers all 4 hazard categories directly (glass_breaking, siren,
car_horn, clock_alarm) plus ample ambience diversity, so UrbanSound8K was skipped to
save time under the deadline. Noted as a possible future-work data-augmentation source.
