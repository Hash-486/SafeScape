# SafeScape — Architecture Comparison (accuracy vs. latency vs. size)

All models trained on the same source-file-disjoint train/val/test split (70/15/15),
evaluated once on the held-out test set (2799 windows). Distress-call recall is called
out separately since a missed hazard is costlier than a false alarm per the proposal.

| architecture | test accuracy | macro recall | distress_call recall | params | size (fp32) | size (best) | CPU latency |
|---|---|---|---|---|---|---|---|
| MFCC-CNN | 66.6% | 73.3% | 66.7% | 60,901 | 248.6 KB | 249.3 KB (quantized, no gain) | 1.05 ms |
| log-mel CRNN | **84.5%** | 79.2% | **82.4%** (82.3% fp32) | 499,237 | 1958.3 KB | **516.4 KB** (quantized, -73.6%) | 89–138 ms |
| Transformer (distilled from CRNN) | 73.5% | 74.2% | 68.7% | 84,293 | 339.0 KB | 339.0 KB (not quantized) | 6.74 ms |

**Takeaways:**
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
