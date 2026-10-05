# Review 3 — UI integrated with every model, results compared (6 marks)

| file | what | marks |
|---|---|---|
| `SafeScape_Review3.pptx` | 10-slide deck: integration, dashboard, UI testing, functional tests, comparison, limitations | presentation |
| `SafeScape_Case_Study_Report.docx` / `.pdf` | 20-page case study report | 6 |
| the app itself (below) | working models with UI integrated | 4 |

## Live demo

From the repo root:

```
.venv/Scripts/python.exe run_demo.py
```

It checks all three model bundles, starts the server on port 8124, and verifies one
held-out clip per class. Then, on the laptop:

1. `http://127.0.0.1:8124/compare` — pick a test clip, upload a file, or record 2 s:
   all three models answer side by side with probabilities and latency.
2. **Run all test clips** — 20 held-out clips through every model; expect
   MFCC-CNN 15/20, CRNN 17/20, Transformer 17/20.
3. Scroll to the offline metrics table and confusion matrices.
4. `http://127.0.0.1:8124/` — the mobile app (CRNN), Settings → *Compare all models*.

The microphone only works on `localhost` (browsers block it over plain HTTP on the LAN);
upload and test clips work anywhere.

## Tests

```
.venv/Scripts/python.exe server/test_multi_model.py     # all models, held-out clips + edge inputs
.venv/Scripts/python.exe server/test_clip_position.py   # hazard position in clip (1 known ambience failure, see report §8)
.venv/Scripts/python.exe server/test_client.py          # against the running server
```
