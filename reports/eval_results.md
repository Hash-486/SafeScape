# SafeScape — Architecture Comparison (accuracy vs. latency vs. size)

All models trained on the same source-file-disjoint train/val/test split (70/15/15),
evaluated once on the held-out test set (2799 windows). Distress-call recall is called
out separately since a missed hazard is costlier than a false alarm per the proposal.

| architecture | test accuracy | macro recall | distress_call recall | params | size (fp32) | size (best) | CPU latency |
|---|---|---|---|---|---|---|---|
| MFCC-CNN | 66.6% | 73.3% | 66.7% | 60,901 | 248.6 KB | 249.3 KB (quantized, no gain) | 0.44 ms |
| log-mel CRNN | 84.5% | **79.2%** | 82.3% | 499,237 | 1958.3 KB | **516.4 KB** (quantized, -73.6%) | 6.44 ms |
| **log-mel CRNN + calibration** | **84.8%** | 78.4% | **85.6%** | 499,237 | 1958.3 KB | **516.4 KB** (quantized, -73.6%) | 6.64 ms |
| Transformer (distilled from CRNN) | 73.5% | 74.2% | 68.7% | 84,293 | 339.0 KB | 339.0 KB (not quantized) | 0.48 ms |

> **Latency note:** all latencies in this table were re-measured in a single session on an
> otherwise-idle CPU. Earlier recorded figures (1.05 / 89.47 / 6.74 ms) were taken during a
> training-heavy session and were inflated by CPU contention; accuracy figures reproduced
> exactly on re-measurement, confirming the models themselves are unchanged. The relative
> ordering (CNN ≲ Transformer ≪ CRNN) holds either way.

**Takeaways:**
- **Post-hoc calibration raises the metric that matters most.** A per-class additive logit
  bias, fitted on the *validation* split only (`scripts/12_calibrate_logits.py`), lifts
  distress_call recall 82.3% -> 85.6% and accuracy 84.5% -> 84.8% with zero retraining and
  no change in model size. See the calibration section below for the safety-constrained
  objective that produced it.
- **log-mel CRNN is the champion** — best accuracy and distress-call recall by a wide
  margin, and dynamic quantization shrinks it 73.6% with no accuracy loss. Exported as
  the model the FastAPI server actually serves.
- **MFCC-CNN** is the fastest and lightest by far (sub-millisecond CPU inference) but
  trades off meaningfully on accuracy — MFCCs discard the fine time-frequency detail
  the CRNN's log-mel input + temporal modeling can exploit, and dynamic PTQ can't help
  a conv-heavy model (only Linear/GRU layers get quantized).
  See `mfcc_cnn_quantized` note below.
- **Distilled transformer** lands between the two on accuracy while being ~6x smaller
  than the fp32 CRNN and ~13x faster at inference than the CRNN — a reasonable
  accuracy/efficiency middle ground, though the CRNN teacher it distilled from still
  outperforms it directly on this dataset.
- **Class imbalance is the dominant limitation across all three**: `glass_break`/
  `horn_skid` have only 40 ESC-50 source clips each (vs. 1334 for ambience, 775 for
  distress_call), so every model shows a recurring pattern of high recall but low
  precision on these two classes (class-weighted loss correctly prioritizes not missing
  hazards, at the cost of more false alarms on the thinnest classes) — see per-class
  tables below.

---

### mfcc_cnn_fp32

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| distress_call | 0.811 | 0.667 | 0.732 | 843 |
| glass_break | 0.072 | 0.895 | 0.133 | 19 |
| horn_skid | 0.148 | 0.617 | 0.239 | 47 |
| alarm | 0.276 | 0.835 | 0.415 | 115 |
| ambience | 0.876 | 0.654 | 0.749 | 1775 |
| **macro avg** | 0.436 | 0.733 | 0.453 | 2799 |
| **accuracy** | | | 0.666 | |

**accuracy:** 0.666 | **distress_call recall:** 0.667 | **params:** 60,901 | **size:** 248.6 KB | **CPU latency:** 1.05 ms/clip

![confusion matrix](reports\figures\mfcc_cnn_fp32_confusion_matrix.png)

---
### logmel_crnn_fp32

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| distress_call | 0.862 | 0.823 | 0.842 | 843 |
| glass_break | 0.258 | 0.895 | 0.400 | 19 |
| horn_skid | 0.242 | 0.638 | 0.351 | 47 |
| alarm | 0.639 | 0.739 | 0.685 | 115 |
| ambience | 0.921 | 0.867 | 0.893 | 1775 |
| **macro avg** | 0.584 | 0.792 | 0.634 | 2799 |
| **accuracy** | | | 0.845 | |

**accuracy:** 0.845 | **distress_call recall:** 0.823 | **params:** 499,237 | **size:** 1958.3 KB | **CPU latency:** 89.47 ms/clip

![confusion matrix](reports\figures\logmel_crnn_fp32_confusion_matrix.png)

---
### logmel_crnn_quantized (dynamic PTQ, qint8)

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| distress_call | 0.861 | 0.824 | 0.842 | 843 |
| glass_break | 0.262 | 0.895 | 0.405 | 19 |
| horn_skid | 0.244 | 0.638 | 0.353 | 47 |
| alarm | 0.634 | 0.739 | 0.683 | 115 |
| ambience | 0.920 | 0.866 | 0.892 | 1775 |
| **macro avg** | 0.584 | 0.793 | 0.635 | 2799 |
| **accuracy** | | | 0.845 | |

**accuracy:** 0.845 | **distress_call recall:** 0.824 | **size:** 516.4 KB (vs 1958.3 KB fp32, -73.6%) | **CPU latency:** 138.47 ms/clip (vs 89.47 ms fp32)

Quantization preserved accuracy/recall essentially exactly while cutting size by nearly
3/4 — but CPU latency actually *rose* rather than fell. This is a known dynamic-PTQ
quirk: the quantize/dequantize overhead around each GRU/Linear call isn't offset by
compute savings at this small a model size and batch-of-1 inference workload, so the
win here is model-size/memory, not raw speed. Both are still comfortably fast enough
for near-real-time (well under the 1s clip duration). **This quantized CRNN is the
champion model exported for serving** (best recall + by far the smallest footprint,
size mattering most for the proposal's low-cost/on-device framing).

---
### mfcc_cnn_quantized (dynamic PTQ, qint8)

Dynamic PTQ only quantizes `nn.Linear`/`nn.GRU` modules; MFCC-CNN is conv-heavy with a
single small final Linear layer, so — as anticipated — quantization had essentially no
effect: 248.6 KB -> 249.3 KB (+0.3%, quantization bookkeeping overhead exceeds the tiny
savings). Static quantization or QAT (which *can* quantize conv layers) is the
documented next step if compressing the CNN specifically becomes a priority.

---
### transformer_distilled_fp32

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| distress_call | 0.787 | 0.687 | 0.733 | 843 |
| glass_break | 0.167 | 0.789 | 0.275 | 19 |
| horn_skid | 0.152 | 0.830 | 0.257 | 47 |
| alarm | 0.460 | 0.643 | 0.536 | 115 |
| ambience | 0.867 | 0.760 | 0.810 | 1775 |
| **macro avg** | 0.486 | 0.742 | 0.522 | 2799 |
| **accuracy** | | | 0.735 | |

**accuracy:** 0.735 | **distress_call recall:** 0.687 | **params:** 84,293 | **size:** 339.0 KB | **CPU latency:** 6.74 ms/clip

![confusion matrix](reports\figures\transformer_distilled_fp32_confusion_matrix.png)

---
### logmel_crnn_recheck_baseline

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| distress_call | 0.862 | 0.823 | 0.842 | 843 |
| glass_break | 0.258 | 0.895 | 0.400 | 19 |
| horn_skid | 0.242 | 0.638 | 0.351 | 47 |
| alarm | 0.639 | 0.739 | 0.685 | 115 |
| ambience | 0.921 | 0.867 | 0.893 | 1775 |
| **macro avg** | 0.584 | 0.792 | 0.634 | 2799 |
| **accuracy** | | | 0.845 | |

**accuracy:** 0.845 | **distress_call recall:** 0.823 | **params:** 499,237 | **size:** 1958.3 KB | **CPU latency:** 6.44 ms/clip

![confusion matrix](reports\figures\logmel_crnn_recheck_baseline_confusion_matrix.png)

---
### logmel_crnn_calibrated

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| distress_call | 0.840 | 0.856 | 0.848 | 843 |
| glass_break | 0.283 | 0.895 | 0.430 | 19 |
| horn_skid | 0.246 | 0.638 | 0.355 | 47 |
| alarm | 0.694 | 0.670 | 0.681 | 115 |
| ambience | 0.928 | 0.861 | 0.893 | 1775 |
| **macro avg** | 0.598 | 0.784 | 0.642 | 2799 |
| **accuracy** | | | 0.848 | |

**accuracy:** 0.848 | **distress_call recall:** 0.856 | **params:** 499,237 | **size:** 1958.3 KB | **CPU latency:** 6.64 ms/clip

![confusion matrix](reports\figures\logmel_crnn_calibrated_confusion_matrix.png)

---
### mfcc_cnn_remeasure

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| distress_call | 0.811 | 0.667 | 0.732 | 843 |
| glass_break | 0.072 | 0.895 | 0.133 | 19 |
| horn_skid | 0.148 | 0.617 | 0.239 | 47 |
| alarm | 0.276 | 0.835 | 0.415 | 115 |
| ambience | 0.876 | 0.654 | 0.749 | 1775 |
| **macro avg** | 0.436 | 0.733 | 0.453 | 2799 |
| **accuracy** | | | 0.666 | |

**accuracy:** 0.666 | **distress_call recall:** 0.667 | **params:** 60,901 | **size:** 248.6 KB | **CPU latency:** 0.44 ms/clip

![confusion matrix](reports\figures\mfcc_cnn_remeasure_confusion_matrix.png)

---
### transformer_remeasure

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| distress_call | 0.787 | 0.687 | 0.733 | 843 |
| glass_break | 0.167 | 0.789 | 0.275 | 19 |
| horn_skid | 0.152 | 0.830 | 0.257 | 47 |
| alarm | 0.460 | 0.643 | 0.536 | 115 |
| ambience | 0.867 | 0.760 | 0.810 | 1775 |
| **macro avg** | 0.486 | 0.742 | 0.522 | 2799 |
| **accuracy** | | | 0.735 | |

**accuracy:** 0.735 | **distress_call recall:** 0.687 | **params:** 84,293 | **size:** 339.0 KB | **CPU latency:** 0.48 ms/clip

![confusion matrix](reports\figures\transformer_remeasure_confusion_matrix.png)

---
## Post-hoc logit calibration (no retraining)

Training used a class-weighted cross-entropy to protect hazard recall. That works, but it
pushes the decision boundary toward the rare classes, so `glass_break`/`horn_skid` end up
with high recall and poor precision. `scripts/12_calibrate_logits.py` corrects the operating
point *after* training by fitting one additive bias per class on the **validation** split
(2712 windows; the test split is never touched) via coordinate ascent, then adding that bias
to the logits before softmax. No gradient updates, no change in model size or architecture.

**Choice of objective — this is the interesting part.** Optimizing macro-F1 without
constraints produces a much better-looking headline number but is wrong for this
application:

| objective | val accuracy | val macro-F1 | val macro recall | glass_break R | horn_skid R |
|---|---|---|---|---|---|
| uncalibrated | 0.845 | 0.662 | 0.758 | 0.844 | 0.560 |
| unconstrained macro-F1 | **0.863** | **0.700** | 0.671 | 0.594 | **0.340** |
| safety-constrained (shipped) | 0.844 | 0.672 | **0.758** | 0.844 | 0.560 |

Unconstrained macro-F1 buys its precision by giving up hazard recall — `horn_skid` recall
collapses to 0.340, i.e. it misses two thirds of those events. That directly contradicts
RQ4 of the proposal ("preserving recall — since a missed danger event is far costlier here
than a false alarm"). The shipped calibration therefore maximizes macro-F1 **subject to a
floor on every hazard class's recall** (no more than 2 points below its uncalibrated value),
which takes the free precision without trading away detection.

**Held-out test results (safety-constrained bias `[+0.4, -0.2, 0.0, -0.6, 0.0]`):**

| class | precision | recall | f1 | Δ recall |
|---|---|---|---|---|
| distress_call | 0.840 (was 0.862) | **0.856** (was 0.823) | 0.848 (was 0.842) | **+3.3 pts** |
| glass_break | 0.283 (was 0.258) | 0.895 (unchanged) | 0.430 (was 0.400) | 0 |
| horn_skid | 0.246 (was 0.242) | 0.638 (unchanged) | 0.355 (was 0.351) | 0 |
| alarm | 0.694 (was 0.639) | 0.670 (was 0.739) | 0.681 (was 0.685) | -6.9 pts |
| ambience | 0.928 (was 0.921) | 0.861 (was 0.867) | 0.893 (unchanged) | -0.6 pts |
| **macro avg** | **0.598** (was 0.584) | 0.784 (was 0.792) | **0.642** (was 0.634) | |
| **accuracy** | | | **0.848** (was 0.845) | |

The one real cost is `alarm` recall (-6.9 points), which stayed inside tolerance on
validation but generalized slightly worse to test. In absolute terms the trade is still
favourable for the application: +3.3 points on `distress_call` over 843 test windows is
roughly **+28 hazard events correctly caught**, against roughly **-8 missed** on `alarm`'s
115 windows — a net gain of ~20 correctly detected hazard events.

Verified end-to-end: the calibration bundle (`models/exported/calibration.json`) is loaded
by `server/inference.py` and applied before softmax, and a 20-clip live smoke test against
the running server scored 16/20 with all five classes represented.

---
### regression_check

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| distress_call | 0.840 | 0.856 | 0.848 | 843 |
| glass_break | 0.283 | 0.895 | 0.430 | 19 |
| horn_skid | 0.246 | 0.638 | 0.355 | 47 |
| alarm | 0.694 | 0.670 | 0.681 | 115 |
| ambience | 0.928 | 0.861 | 0.893 | 1775 |
| **macro avg** | 0.598 | 0.784 | 0.642 | 2799 |
| **accuracy** | | | 0.848 | |

**accuracy:** 0.848 | **distress_call recall:** 0.856 | **params:** 499,237 | **size:** 1958.3 KB | **CPU latency:** 7.20 ms/clip

![confusion matrix](reports\figures\regression_check_confusion_matrix.png)

---
### logmel_crnn_v2

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| distress_call | 0.913 | 0.888 | 0.900 | 779 |
| glass_break | 0.684 | 0.684 | 0.684 | 76 |
| horn_skid | 0.742 | 0.938 | 0.828 | 144 |
| alarm | 0.922 | 0.905 | 0.914 | 613 |
| ambience | 0.903 | 0.899 | 0.901 | 1608 |
| **macro avg** | 0.833 | 0.863 | 0.845 | 3220 |
| **accuracy** | | | 0.894 | |

**accuracy:** 0.894 | **distress_call recall:** 0.888 | **params:** 499,237 | **size:** 1958.2 KB | **CPU latency:** 11.60 ms/clip

![confusion matrix](reports\figures\logmel_crnn_v2_confusion_matrix.png)

---
### logmel_crnn_v2_calibrated

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| distress_call | 0.915 | 0.882 | 0.898 | 779 |
| glass_break | 0.684 | 0.684 | 0.684 | 76 |
| horn_skid | 0.832 | 0.931 | 0.879 | 144 |
| alarm | 0.910 | 0.922 | 0.916 | 613 |
| ambience | 0.903 | 0.905 | 0.904 | 1608 |
| **macro avg** | 0.849 | 0.865 | 0.856 | 3220 |
| **accuracy** | | | 0.898 | |

**accuracy:** 0.898 | **distress_call recall:** 0.882 | **params:** 499,237 | **size:** 1958.2 KB | **CPU latency:** 11.56 ms/clip

![confusion matrix](reports\figures\logmel_crnn_v2_calibrated_confusion_matrix.png)

---
