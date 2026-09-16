"""Build the SafeScape review presentation (.pptx) against the 12-item grading rubric.

Re-runnable: edit the CONTENT constants below (team roster especially) and run again.
Figures come from reports/figures/, UI screenshots from the path in SHOTS.

Usage: python 14_build_presentation.py [--shots <dir>] [--out <path>]
"""
import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "reports" / "figures"

INK = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT = RGBColor(0xC2, 0x4B, 0x3C)
TEAL = RGBColor(0x2E, 0x6F, 0x6B)
MUTED = RGBColor(0x5A, 0x5A, 0x72)
LIGHT = RGBColor(0xF2, 0xF4, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BORDER = RGBColor(0xD5, 0xD8, 0xE3)

W, H = 13.333, 7.5
MARGIN = 0.62

# ----------------------------------------------------------------------------- content
TEAM = [
    ("Amruth Rohan KR", "MFCC-CNN", "Cb.en.u4ece23222"),
    ("Harish Venkat VS", "log-mel CRNN (champion), Distilled Tiny Transformer",
     "Cb.en.u4ece23219"),
]

# Literature slides are keyed by architecture and owned by whoever presents it, so the
# slide count does not depend on len(TEAM) -- one member may own more than one model.
LIT_OWNER = [
    ("MFCC-CNN", "Amruth Rohan KR"),
    ("log-mel CRNN", "Harish Venkat VS"),
    ("Distilled Tiny Transformer", "Harish Venkat VS"),
]
GUIDE = "Dr. Bagyammal T, Professor, CSE Department"
GROUP_NO = "1"
COURSE = "23CSE473 — Neural Networks and Deep Learning"
GITHUB = "https://github.com/Hash-486/SafeScape"
DEMO_URL = "[LinkedIn product demo URL]"

LIT = {
    "MFCC-CNN": [
        ("Robust Scream Detection and Scream Temporal Interval Prediction Using CNN-Transformer and Windowing CNN",
         "Kim, Jang & Lee", "IEEE Access, 13, 71374–71387, 2025", "10.1109/ACCESS.2025.3556729",
         "Parallel CNN + Transformer with a regression head predicting exact scream start/end.",
         "~3 pp F-measure gain, ~10× lower EER vs. prior SOTA (relative figures)"),
        ("A lightweight CNN–transformer hybrid architecture with channel attention for real-time hazardous acoustic event detection",
         "Altayeva & Omarov", "Frontiers in Artificial Intelligence, 9, art. 1824067, 2026", "10.3389/frai.2026.1824067",
         "Lightweight CNN + channel attention for 8-class edge-deployable hazard detection.",
         "93.2% acc · F1 0.86 · 1.35 M params · 0.42 GFLOPs · 14.8 ms/sample"),
        ("Environmental sound classification using two-stream deep neural network with interactive time-frequency attention",
         "Chen & Peng", "Applied Acoustics, 238, art. 110794, 2025", "10.1016/j.apacoust.2025.110794",
         "Two-stream time/frequency branches with depthwise-separable convolutions.",
         "94.2% on ESC-50 · 95.3% on UrbanSound8K"),
        ("Robust Classification of Urban Sounds in Noisy Environments: SPWVD-MFCC and Dual-Stream Classifier",
         "Peng, Wang & Abdulla", "Acoustics Australia, 53(2), 253–268, 2025", "10.1007/s40857-025-00350-6",
         "Replaces the STFT front-end of MFCC with a Smoothed Pseudo-Wigner–Ville Distribution.",
         "up to 37.2% improvement over STFT-based baselines (relative)"),
        ("Robust Forest Sound Classification Using Pareto-Mordukhovich Optimized MFCC in Environmental Monitoring",
         "Qurthobi, Damaševičius, Barzdaitis & Maskeliūnas", "IEEE Access, 13, 20923–20944, 2025",
         "10.1109/ACCESS.2025.3535796",
         "Tunes MFCC extraction parameters rather than fixing them; MobileNet/GoogleNet + BiLSTM.",
         "78.52% average accuracy on FSC22 (27 classes)"),
    ],
    "log-mel CRNN": [
        ("An Ensemble of Convolutional Neural Networks for Sound Event Detection",
         "Mukhamadiyev, Khujayarov, Nabieva & Cho", "Mathematics (MDPI), 13(9), art. 1502, 2025",
         "10.3390/math13091502",
         "Ensemble CNN + Bi-GRU CRNN for public-safety sound events incl. screams and breaking sounds.",
         "segment-based F1 71.5% · event-based F1 46%"),
        ("Frequency-Adaptive and Multi-Scale CRNN with Hybrid Sequential Modeling for Sound Event Detection",
         "Zhou & Rui", "Circuits, Systems, and Signal Processing (Springer), 2026", "10.1007/s00034-026-03754-5",
         "Frequency-dynamic convolution + sliding multi-scale module + hybrid-minGRU on DESED 2023.",
         "event-based F1 0.636 · PSDS1 0.589 · PSDS2 0.824"),
        ("ASiT-CRNN: sound event detection with fine-tuning of a self-supervised pre-trained ASiT model",
         "Zheng, Zhang, Atito, Yang, Wang & Mei", "Digital Signal Processing, 160, art. 105055, 2025",
         "10.1016/j.dsp.2025.105055",
         "Embeds a self-supervised spectrogram transformer into a CRNN backbone.",
         "PSDS1 0.488 · PSDS2 0.767 vs. CRNN baseline 0.351 / 0.552"),
        ("Dangerous Sound Detection Using Convolutional Feature Extraction and Temporal Modeling with BiLSTM",
         "Omarov & Altayeva", "Eng. Technology & Applied Science Research, 15(6), 28850–28855, 2025",
         "10.48084/etasr.13068",
         "CNN + BiLSTM + attention over 8 hazard classes, 1 s window with 0.5 s hop — same framing as SafeScape.",
         ">90% accuracy · 6.8 ms GPU / 23.5 ms CPU per window"),
        ("Environmental sound classification using convolutional recurrent neural network and data augmentation",
         "Bansal & Garg", "Multimedia Tools and Applications, 84(34), 42827–42849, 2025",
         "10.1007/s11042-025-20820-3",
         "CRNN over cepstral features with augmentation, motivated by security/crime-investigation use.",
         "99.89% on UrbanSound8K (verify split protocol before quoting)"),
    ],
    "Distilled Tiny Transformer": [
        ("CMKD: CNN/Transformer-Based Cross-Model Knowledge Distillation for Audio Classification",
         "Gong, Khurana, Rouditchenko & Glass", "IEEE Trans. Pattern Analysis and Machine Intelligence, 48(3), 3571–3585, 2026",
         "10.1109/TPAMI.2025.3644853",
         "CNNs and audio spectrogram transformers are mutually effective teachers; students beat their teachers.",
         "ESC-50 96.8% · student 8 M params vs. AST-Base 88 M (~11× smaller)"),
        ("Parameter-efficient target-specialized audio spectrogram transformers via selective cross-domain knowledge distillation",
         "Liang, Luo & Zhong", "Knowledge-Based Systems, 332, art. 114893, 2025/2026", "10.1016/j.knosys.2025.114893",
         "Domain-specific adapters + sparse fusion transfer only useful knowledge into a specialized AST.",
         "matches/outperforms SOTA across six domains, training only a small adapter fraction"),
        ("Acoustic Intelligence with Multi-Stage Model Optimization for Environmental Sound Classification",
         "Sarathchandra, Mallikarachchi, Madushani & Meedeniya", "Smart Cities (MDPI), 9(5), art. 86, 2026",
         "10.3390/smartcities9050086",
         "Channel pruning → quantization-aware training → distillation, the full compression stack.",
         "511,033 → 50,774 params (~10×) retaining 96.75% accuracy on ESC-10"),
        ("A Sensor-Based TinyML Acoustic Monitoring System for Edge-Side Recognition on Resource-Constrained Microcontrollers",
         "Wang & Yu", "Sensors (MDPI), 26(13), art. 3972, 2026", "10.3390/s26133972",
         "End-to-end MFCC → INT8-quantized NN deployed fully offline on an Arduino-class MCU.",
         "98.28% acc · 97.21% macro-F1 · ~26.9 KB deployed model"),
        ("Low resource multi-head self-attention based transformer for airborne acoustic classification system",
         "Mishra & Ghosh", "Journal on Audio, Speech, and Music Processing (Springer), 2026, art. 467",
         "10.1186/s13636-026-00467-0",
         "LAiT: inverted residual blocks + compact multi-head self-attention for real-time acoustic classification on low-power devices.",
         "0.23 M params · <1 MB memory · +1.18% over the ULNN lightweight baseline"),
    ],
}

STANDARD_PAPER = {
    "title": "Scream and gunshot detection and localization for audio-surveillance systems",
    "authors": "Valenzise, Gerosa, Tagliasacchi, Antonacci & Sarti (Politecnico di Milano)",
    "venue": "IEEE Conf. on Advanced Video and Signal Based Surveillance (AVSS), pp. 21–26, 2007",
    "doi": "10.1109/AVSS.2007.4425280",
    "citations": "~394 citations",
    "why": [
        "Defines the exact task SafeScape modernises: automatically recognising human distress "
        "vocalisations and impulsive hazard sounds from ambient audio for public-safety response.",
        "Its scream-vs-ambience discrimination problem is literally SafeScape's core class boundary.",
        "Most-cited foundational work in acoustic hazard detection — nearly every later scream / "
        "glass-break detection paper traces back to it.",
        "SafeScape's contribution reads as the natural update: hand-tuned GMMs → deep models, and "
        "a fixed camera-steering installation → a fully offline personal device.",
    ],
    "alt": "Journal alternative if required: Foggia et al., \"Audio Surveillance of Roads: A System for "
           "Detecting Anomalous Sounds,\" IEEE Trans. Intelligent Transportation Systems, 17(1), 279–288, "
           "2016, DOI 10.1109/TITS.2015.2470216 — maps directly onto the horn_skid class.",
}


# ----------------------------------------------------------------------------- helpers
def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text, size=14, bold=False, color=INK, space_before=0, space_after=6,
         align=PP_ALIGN.LEFT, italic=False, first=False, bullet=None, line=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    if bullet:
        text = f"{bullet}  {text}"
    p.text = text
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    if line:
        p.line_spacing = line
    for r in p.runs:
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = "Segoe UI"
    return p


def bar(slide, x, y, w, h, color):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def slide_with_header(prs, title, rubric=None, marks=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar(s, 0, 0, W, 0.06, ACCENT)
    tf = textbox(s, MARGIN, 0.34, W - 2 * MARGIN - 2.2, 0.62)
    para(tf, title, size=26, bold=True, first=True, space_after=0)
    if rubric:
        rt = textbox(s, W - MARGIN - 2.3, 0.40, 2.3, 0.5)
        label = f"Rubric {rubric}" + (f" · {marks} marks" if marks else "")
        para(rt, label, size=10.5, color=MUTED, align=PP_ALIGN.RIGHT, first=True)
    bar(s, MARGIN, 1.06, W - 2 * MARGIN, 0.018, BORDER)
    return s


def table(slide, headers, rows, x, y, w, col_w=None, fs=11.5, hfs=11.5, row_h=0.34):
    n_r, n_c = len(rows) + 1, len(headers)
    shp = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(row_h * n_r))
    tbl = shp.table
    tbl.first_row = True
    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Emu(int(Inches(w) * cw / total))
    for c, htxt in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.text = htxt
        cell.fill.solid()
        cell.fill.fore_color.rgb = INK
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = cell.margin_right = Inches(0.07)
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(hfs); r.font.bold = True
                r.font.color.rgb = WHITE; r.font.name = "Segoe UI"
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            strong = "**" in str(val)
            cell.text = str(val).replace("**", "")
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if ri % 2 else LIGHT
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            for p in cell.text_frame.paragraphs:
                p.line_spacing = 0.92
                for r in p.runs:
                    r.font.size = Pt(fs)
                    r.font.color.rgb = INK
                    r.font.name = "Segoe UI"
                    r.font.bold = strong
    return tbl


def picture_fit(slide, img, x, y, max_w, max_h):
    """Insert image scaled to fit inside the box, centred horizontally on it."""
    from PIL import Image
    with Image.open(img) as im:
        iw, ih = im.size
    ar = iw / ih
    w, h = max_w, max_w / ar
    if h > max_h:
        h, w = max_h, max_h * ar
    left = x + (max_w - w) / 2
    return slide.shapes.add_picture(str(img), Inches(left), Inches(y), Inches(w), Inches(h))


def footer(slide, text):
    tf = textbox(slide, MARGIN, H - 0.46, W - 2 * MARGIN, 0.3)
    para(tf, text, size=9.5, color=MUTED, italic=True, first=True)


# ------------------------------------------------------------------------------ slides
def s_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar(s, 0, 0, W, 2.55, INK)
    bar(s, 0, 2.55, W, 0.09, ACCENT)

    tf = textbox(s, MARGIN, 0.72, W - 2 * MARGIN, 1.0)
    para(tf, "SafeScape", size=46, bold=True, color=WHITE, first=True, space_after=2)
    para(tf, "On-Device Acoustic Distress and Hazard Recognition", size=19, color=RGBColor(0xD8, 0xDC, 0xE8))
    para(tf, "“Hearing the sound of danger, before it’s too late.”", size=13,
         italic=True, color=RGBColor(0xB0, 0xB6, 0xC8), space_before=6)

    tf = textbox(s, MARGIN, 2.92, 6.3, 2.4)
    para(tf, "TEAM MEMBERS", size=11, bold=True, color=ACCENT, first=True, space_after=8)
    for name, arch, reg in TEAM:
        para(tf, f"{name}  ·  {reg}", size=13.5, bold=True, space_after=1)
        para(tf, f"    architecture: {arch}", size=11.5, color=MUTED, space_after=7)

    tf = textbox(s, 7.3, 2.92, W - 7.3 - MARGIN, 2.4)
    para(tf, "FACULTY GUIDE", size=11, bold=True, color=ACCENT, first=True, space_after=8)
    para(tf, GUIDE, size=13, space_after=14)
    para(tf, f"Group No: {GROUP_NO}", size=12, color=MUTED, space_after=3)
    para(tf, "Category: Research–Product–Software", size=12, color=MUTED, space_after=3)
    para(tf, COURSE, size=12, color=MUTED)

    bar(s, MARGIN, 5.78, W - 2 * MARGIN, 0.018, BORDER)
    tf = textbox(s, MARGIN, 5.98, W - 2 * MARGIN, 1.0)
    para(tf, f"GitHub repository:  {GITHUB}", size=12, bold=True, color=TEAL, first=True, space_after=4)
    para(tf, f"Product demo:  {DEMO_URL}", size=12, color=MUTED, space_after=8)
    para(tf, "SDG 5 — Gender Equality (Target 5.2)   ·   SDG 11 — Sustainable Cities (Target 11.7)",
         size=11.5, italic=True, color=MUTED)
    return s


def s_problem(prs):
    s = slide_with_header(prs, "Problem statement, introduction & motivation", "1", 4)
    tf = textbox(s, MARGIN, 1.30, 6.15, 5.5)
    para(tf, "PROBLEM", size=11, bold=True, color=ACCENT, first=True, space_after=6)
    para(tf, "Every mainstream personal-safety technology — panic buttons, GPS trackers, wearable "
             "alarms — requires the victim to consciously act: unlock a phone, press a button, speak "
             "a wake phrase.", size=13.5, space_after=6, line=1.15)
    para(tf, "In real assault, harassment or accident scenarios victims frequently cannot do this, "
             "due to shock, physical restraint, or simply lack of time.", size=13.5, space_after=10, line=1.15)
    para(tf, "There is no widely available, low-cost, privacy-preserving system that passively senses "
             "non-verbal acoustic distress cues and raises an alert automatically — fully on-device, "
             "without streaming audio to a cloud service.", size=13.5, bold=True, line=1.15)

    tf = textbox(s, 7.05, 1.30, W - 7.05 - MARGIN, 4.2)
    para(tf, "MOTIVATION", size=11, bold=True, color=ACCENT, first=True, space_after=6)
    for b in [
        "A large share of incidents of violence against women occur where the victim has no "
        "realistic opportunity to manually summon help.",
        "Screams, glass breaking and crashes carry strong, learnable acoustic signatures — embedded "
        "scream detectors already reach ~90% with sub-megabyte models.",
        "Commercial safety apps (bSafe, Raksha) are exclusively manual-trigger; none use passive "
        "multi-class acoustic sensing.",
        "TinyML techniques (quantisation, pruning, distillation) now make continuous phone-CPU "
        "inference realistic at low battery cost.",
    ]:
        para(tf, b, size=12.5, bullet="▪", space_after=7, line=1.12)

    bar(s, 7.05, 5.62, W - 7.05 - MARGIN, 1.12, LIGHT)
    tf = textbox(s, 7.28, 5.78, W - 7.28 - MARGIN - 0.2, 0.9)
    para(tf, "OBJECTIVE", size=10, bold=True, color=TEAL, first=True, space_after=4)
    para(tf, "Classify 1-second rolling audio windows into 5 classes in near real time, fully "
             "offline, in a model small enough to live on a phone.", size=11.5, line=1.1)
    return s


def s_literature(prs, arch, member):
    s = slide_with_header(prs, f"Literature survey — {arch}", "2", 5)
    tf = textbox(s, MARGIN, 1.20, W - 2 * MARGIN, 0.3)
    para(tf, f"{member} · 5 journal papers (2025/2026, Scopus/SJR-indexed venues)",
         size=12, italic=True, color=MUTED, first=True)
    y = 1.62
    for i, (title, authors, venue, doi, summary, metrics) in enumerate(LIT[arch], 1):
        bar(s, MARGIN, y, 0.045, 0.86, ACCENT if i % 2 else TEAL)
        tf = textbox(s, MARGIN + 0.20, y - 0.02, W - 2 * MARGIN - 0.25, 0.9)
        para(tf, f"{i}.  {title}", size=12, bold=True, first=True, space_after=2, line=0.95)
        para(tf, f"{authors} · {venue} · DOI {doi}", size=9.8, color=MUTED, space_after=2)
        para(tf, f"{summary}   —   {metrics}", size=10, color=TEAL, space_after=0, line=0.95)
        y += 1.00
    return s


def s_system_arch(prs):
    s = slide_with_header(prs, "Architecture diagram — overall application", "3", 5)
    picture_fit(s, FIGS / "system_architecture.png", MARGIN, 1.24, W - 2 * MARGIN, 4.4)
    tf = textbox(s, MARGIN, 5.95, W - 2 * MARGIN, 1.0)
    para(tf, "Inference path (top) runs entirely on the device: no audio, and no features derived "
             "from audio, ever leave it. The training path (bottom) runs offline, once.",
         size=12, color=INK, first=True, space_after=4, line=1.1)
    para(tf, "Serving note: the present build runs the model in a local FastAPI process reachable "
             "over the LAN from the phone browser — true in-phone inference (TFLite / ONNX Runtime "
             "Mobile) is the documented next step.", size=11, italic=True, color=MUTED, line=1.1)
    return s


def s_modules(prs):
    s = slide_with_header(prs, "Module details", "4", 5)
    rows = [
        ["1. Data acquisition", "scripts/01_download_notes.md", "Provenance log: exact commands, sources and licences for every dataset pulled."],
        ["2. Manifest builder", "scripts/02_build_manifest.py", "Maps every source file to one of 5 classes; caps the over-represented ambience class."],
        ["3. Preprocessing", "scripts/03_preprocess_audio.py", "Resample 16 kHz, mono, denoise, 1 s windows @ 50% overlap; 70/15/15 split at SOURCE-FILE level (no window leakage)."],
        ["4. Feature extraction", "scripts/04_extract_features.py", "Pre-computes log-mel (64×101) and MFCC (40×101) to .npy — removed a Windows multiprocessing bottleneck (25–30× speed-up)."],
        ["5. Training", "scripts/06_train.py, 08_train_distilled_transformer.py", "Trains each architecture; class-weighted loss; checkpoints on val macro-recall."],
        ["6. Hyper-parameter search", "scripts/09_hparam_search.py", "Optuna search over lr / batch / dropout / hidden size, then retrains the winner."],
        ["7. Evaluation", "scripts/10_evaluate.py", "Test metrics, per-class report, confusion matrix, latency, size — appends to reports/eval_results.md."],
        ["8. Quantise & export", "scripts/11_quantize_export.py", "Dynamic PTQ (qint8) + exports the champion bundle to models/exported/."],
        ["9. Calibration", "scripts/12_calibrate_logits.py", "Fits a per-class logit bias on the val split under a hazard-recall constraint."],
        ["10. Serving + UI", "server/app.py, inference.py, static/", "FastAPI /predict and /health; mobile-styled web app (Home / History / Settings)."],
    ]
    table(s, ["Module", "Implementation", "Responsibility"], rows, MARGIN, 1.26,
          W - 2 * MARGIN, col_w=[2.2, 3.3, 7.4], fs=10.2, hfs=11, row_h=0.47)
    return s


def s_metrics(prs):
    s = slide_with_header(prs, "Performance metrics — definition & suitability", "5", 3)
    rows = [
        ["Accuracy", "(TP+TN) / all", "Overall correctness across the 5 classes.",
         "Easy to communicate, but misleading here: ambience is 63% of windows.", "93.2% [Altayeva 2026]"],
        ["Macro recall", "mean of per-class recall", "Averages recall over classes, ignoring frequency.",
         "The model-selection metric — treats a rare hazard as equally important as common ambience.", "—"],
        ["Per-class precision", "TP / (TP+FP)", "Of windows flagged class c, how many truly are.",
         "Governs false-alarm rate: low precision on a hazard class means nuisance alerts.", "0.66–0.99 [Altayeva 2026]"],
        ["Per-class recall", "TP / (TP+FN)", "Of true class-c windows, how many were caught.",
         "The safety-critical metric: a miss is a hazard that goes unreported.", "0.58–0.87 [Altayeva 2026]"],
        ["distress_call recall", "recall of the scream class", "Detection rate on the flagship SDG-5 class.",
         "Called out separately: the proposal’s RQ4 states a missed danger costs far more than a false alarm.", "—"],
        ["Macro F1", "harmonic mean of macro P & R", "Balances the two under imbalance.",
         "Calibration objective — but constrained so it cannot buy precision with hazard recall.", "0.86 [Altayeva 2026]"],
        ["CPU latency", "ms per 1 s window", "Wall-clock inference on the serving target.",
         "Must sit far below the 1 s window for the rolling pipeline to keep up in real time.", "14.8 ms [Altayeva 2026]"],
        ["Model size", "KB on disk", "Deployed footprint after quantisation.",
         "The low-cost / on-device constraint from the proposal brief.", "~26.9 KB [Wang 2026]"],
    ]
    table(s, ["Metric", "Formula", "What it measures", "Why suitable for SafeScape", "Best in literature"],
          rows, MARGIN, 1.26, W - 2 * MARGIN, col_w=[1.7, 1.9, 2.9, 4.7, 1.9], fs=9.4, hfs=10.2, row_h=0.56)
    footer(s, "Literature values come from the surveyed papers; datasets differ (ESC-50 / UrbanSound8K / custom "
              "8-class sets), so they are reference points, not like-for-like comparisons.")
    return s


def s_dl_arch(prs):
    s = slide_with_header(prs, "Deep learning architecture — log-mel CRNN (champion)", "6", 5)
    picture_fit(s, FIGS / "crnn_architecture.png", MARGIN, 1.20, W - 2 * MARGIN, 3.25)
    tf = textbox(s, MARGIN, 4.66, 6.2, 2.5)
    para(tf, "NOVELTY", size=11, bold=True, color=ACCENT, first=True, space_after=5)
    for b in ["Multi-class hazard taxonomy, not a single scream-vs-not detector — richer, more "
              "actionable alerts than Renesas / IIIT-Delhi single-class systems.",
              "Architecture comparison as a first-class deliverable: three models, one dataset, "
              "one split, reported on accuracy vs. latency vs. size.",
              "Safety-constrained post-hoc calibration: the operating point is chosen under an "
              "explicit hazard-recall floor, not by maximising a headline metric."]:
        para(tf, b, size=11.5, bullet="▪", space_after=6, line=1.1)

    tf = textbox(s, 7.05, 4.66, W - 7.05 - MARGIN, 2.5)
    para(tf, "WHY CRNN OVER PLAIN CNN", size=11, bold=True, color=ACCENT, first=True, space_after=5)
    para(tf, "Frequency-only pooling (2,1) preserves all 101 time frames, so the BiGRU sees the "
             "full temporal evolution of the second. A scream is defined as much by its pitch "
             "contour over time as by any single frame — exactly the information an MFCC-CNN "
             "discards when it average-pools to a single vector.",
         size=11.5, space_after=8, line=1.12)
    para(tf, "Result: +18.2 accuracy points over the MFCC-CNN on identical data — and 89.8% "
             "once the dataset imbalance is fixed too.",
         size=11.5, bold=True, color=TEAL, line=1.1)
    return s


def s_complexity(prs):
    s = slide_with_header(prs, "Time & space complexity — three architectures", "6", 5)
    picture_fit(s, FIGS / "architecture_comparison.png", MARGIN, 1.18, W - 2 * MARGIN, 2.9)
    rows = [
        ["MFCC-CNN", "O(K²·C_in·C_out·F·T) per conv layer — 4 conv blocks, no recurrence",
         "60,901", "248.6 KB", "0.44 ms", "66.6%"],
        ["log-mel CRNN", "conv as above + O(T·(D·h + h²)) for the BiGRU — sequential in T, cannot parallelise over time",
         "499,237", "516.4 KB int8", "11.6 ms", "**89.8%"],
        ["Tiny Transformer", "O(T²·d) self-attention — quadratic in the 101 frames, but fully parallel over T",
         "84,293", "339.0 KB", "0.48 ms", "73.5%"],
    ]
    table(s, ["Architecture", "Asymptotic cost (per 1 s window)", "Params", "Size", "CPU latency", "Accuracy"],
          rows, MARGIN, 4.34, W - 2 * MARGIN, col_w=[2.0, 6.0, 1.4, 1.6, 1.5, 1.4], fs=10, hfs=10.5, row_h=0.62)
    footer(s, "The CRNN’s recurrence makes it the slowest of the three, yet 11.6 ms is still ~85× "
              "under the 1 s window budget — so accuracy, not speed, is the binding constraint here.")
    return s


def s_algorithm(prs):
    s = slide_with_header(prs, "Algorithm procedure — step by step", "7", 5)
    tf = textbox(s, MARGIN, 1.22, 6.15, 5.3)
    steps = [
        ("1 · Framing", "x[n] sampled at fₛ = 16 kHz, segmented into 1 s windows of N = 16,000 "
                            "samples with 50% hop (8,000 samples)."),
        ("2 · STFT", "X(m, k) = Σₙ x[n + mR]·w[n]·e^(−j2πkn/N_FFT), "
                          "Hann window w, hop R → 101 frames."),
        ("3 · Mel filterbank", "M(m, b) = Σₖ |X(m, k)|²·H_b(k), with B = 64 triangular "
                                    "filters spaced on mel(f) = 2595·log₁₀(1 + f/700)."),
        ("4 · Log compression", "S(m, b) = log(M(m, b) + ε), then per-feature normalisation "
                                     "→ input tensor (1, 64, 101)."),
        ("4b · MFCC branch", "c(m, i) = Σ_b S(m, b)·cos[πi(b + ½)/B] — DCT of the "
                                  "log-mel, keeping 40 coefficients."),
    ]
    for j, (head, body) in enumerate(steps):
        para(tf, head, size=12, bold=True, color=ACCENT, first=(j == 0), space_after=2)
        para(tf, body, size=11.5, space_after=8, line=1.1)

    tf = textbox(s, 7.05, 1.22, W - 7.05 - MARGIN, 5.3)
    steps2 = [
        ("5 · Convolutional encoder", "Hₗ = MaxPool₍₂,₁₎(ReLU(BN(Wₗ * Hₗ₋₁ + bₗ))), "
                                           "two blocks, 1→16→32 channels; frequency 64→16, time preserved."),
        ("6 · Reshape to sequence", "(B, 32, 16, 101) → (B, 101, 512): each time frame becomes a "
                                         "512-dim token."),
        ("7 · BiGRU recurrence", "zₜ = σ(W_z·[hₜ₋₁, xₜ]),  rₜ = σ(W_r·[hₜ₋₁, xₜ]),  "
                                      "h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ]),  "
                                      "hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ;  forward and backward."),
        ("8 · Temporal pooling", "h̄ = (1/T)·Σₜ hₜ  →  logits ℓ = W_fc·h̄ + b_fc ∈ ℝ⁵."),
        ("9 · Calibration", "ℓ̃_c = ℓ_c + β_c, with β = [+0.4, −0.2, 0.0, −0.6, 0.0] "
                                "fitted on validation (see rubric 8)."),
        ("10 · Decision", "p = softmax(ℓ̃),  ŷ = argmax_c p_c;  alert if ŷ ∈ "
                              "{distress_call, glass_break, horn_skid, alarm}."),
    ]
    for j, (head, body) in enumerate(steps2):
        para(tf, head, size=12, bold=True, color=ACCENT, first=(j == 0), space_after=2)
        para(tf, body, size=11.5, space_after=7, line=1.1)

    tf = textbox(s, MARGIN, 6.58, 6.15, 0.6)
    para(tf, "Training loss:  L = −Σ_c w_c·y_c·log p_c,  class weights w_c ∝ 1/n_c "
             "(inverse frequency) to protect rare hazards.", size=11, italic=True, color=TEAL,
         first=True, line=1.05)
    return s


def s_hyperparams(prs):
    s = slide_with_header(prs, "Hyper-parameter details & justification", "8", 5)
    rows = [
        ["Learning rate", "8.56 × 10⁻⁴", "Optuna search, 8 trials", "Best val macro-recall (0.751); higher rates (4.3×10⁻³) destabilised the GRU, lower (1.1×10⁻⁴) underfit in 25 epochs."],
        ["Batch size", "16", "Optuna search", "Smaller batches gave noisier but better-generalising updates on a 12.6k-window training set."],
        ["Hidden size (GRU)", "128", "Optuna search", "32 and 64 both underperformed (0.69–0.70 vs 0.751); 128 doubles capacity for ~4× params, still only 516 KB quantised."],
        ["Dropout", "0.269", "Optuna search", "Tuned jointly with the above; regularises the 499k-param model against a modest dataset."],
        ["Epochs", "25", "Fixed", "Checkpointed on best val macro-recall, so extra epochs cannot overfit the selected weights."],
        ["Weight decay", "1 × 10⁻⁴", "Fixed default", "Standard mild L2; not a sensitive axis in preliminary runs."],
        ["Class weights", "∝ 1 / n_c", "Derived from train split", "Hazards are 2–70× rarer than ambience; inverse-frequency weighting stops collapse to the majority class."],
        ["Window / hop", "1.0 s / 50%", "Fixed by design", "Matches the proposal’s near-real-time requirement and the 1 s convention in comparable work (Omarov 2025)."],
        ["Quantisation", "dynamic PTQ, qint8", "Post-training", "−73.6% size with ≤0.001 metric change; Linear/GRU layers only, which is where CRNN parameters live."],
        ["Calibration bias β", "[+0.4, −0.2, 0, −0.6, 0]", "Coordinate ascent on val", "Maximises macro-F1 subject to a hazard-recall floor (≥ uncalibrated − 0.02). See next slide."],
    ]
    table(s, ["Hyper-parameter", "Value", "How chosen", "Justification"], rows, MARGIN, 1.26,
          W - 2 * MARGIN, col_w=[2.3, 2.1, 2.2, 8.0], fs=9.8, hfs=10.5, row_h=0.47)
    return s


def s_results(prs):
    s = slide_with_header(prs, "Results — architecture comparison", "9", 3)
    rows = [
        ["MFCC-CNN", "66.6%", "73.3%", "0.436", "66.7%", "60,901", "0.44 ms"],
        ["Tiny Transformer (distilled)", "73.5%", "74.2%", "0.486", "68.7%", "84,293", "0.48 ms"],
        ["log-mel CRNN", "84.5%", "79.2%", "0.584", "82.3%", "499,237", "6.44 ms"],
        ["log-mel CRNN + calibration", "84.8%", "78.4%", "0.598", "85.6%", "499,237", "6.64 ms"],
        ["**log-mel CRNN + expanded data + calibration", "**89.8%", "**86.5%", "**0.849", "88.2%", "499,237", "11.6 ms"],
    ]
    table(s, ["Architecture", "Accuracy", "Macro recall", "Macro precision", "distress recall", "Params", "Latency"],
          rows, MARGIN, 1.20, W - 2 * MARGIN, col_w=[4.0, 1.5, 1.7, 1.8, 1.7, 1.5, 1.3], fs=10.5, hfs=10.5, row_h=0.44)

    picture_fit(s, FIGS / "logmel_crnn_v2_calibrated_confusion_matrix.png", MARGIN, 3.80, 4.0, 3.15)

    tf = textbox(s, 5.00, 3.80, W - 5.00 - MARGIN, 3.2)
    para(tf, "INFERENCE FROM THE RESULTS", size=11, bold=True, color=ACCENT, first=True, space_after=6)
    for b in ["The CRNN wins decisively on accuracy (+18.2 pts over the CNN, +11.3 over the "
              "transformer) because temporal modelling matters for these classes.",
              "Quantisation is nearly free: −73.6% size for ≤0.001 change in every metric — the "
              "on-device constraint costs essentially no accuracy.",
              "Latency is not the binding constraint: even the slowest model runs 85× faster than "
              "the 1 s window it consumes.",
              "The first four rows share one dataset and split, so they compare like for like. "
              "The final row retrains the champion on an EXPANDED dataset and is therefore "
              "measured on a different, larger test set (3,220 vs 2,799 windows) — it is a "
              "data result, not an architecture result, and is kept separate for that reason.",
              "Macro precision is the headline: 0.598 → 0.849. Class imbalance, not architecture, "
              "was the binding limitation all along (see next slide)."]:
        para(tf, b, size=10.8, bullet="▪", space_after=6, line=1.08)
    return s


def s_calibration(prs):
    s = slide_with_header(prs, "Results — fixing the two weakest classes", "9", 3)
    tf = textbox(s, MARGIN, 1.22, W - 2 * MARGIN, 0.8)
    para(tf, "Training used class-weighted loss to protect hazard recall — which pushes the decision "
             "boundary toward rare classes and costs precision. Calibration re-picks the operating "
             "point after training: one additive bias per class, fitted on validation only, no retraining.",
         size=12.5, first=True, line=1.12)

    rows = [
        ["uncalibrated", "0.845", "0.662", "0.758", "0.844", "0.560", "baseline"],
        ["unconstrained macro-F1", "**0.863", "**0.700", "0.671", "0.594", "0.340", "rejected — misses 2/3 of horn/skid events"],
        ["**safety-constrained (shipped)", "0.844", "0.672", "**0.758", "**0.844", "**0.560", "keeps every hazard recall, takes free precision"],
    ]
    table(s, ["Objective (validation set)", "Accuracy", "Macro F1", "Macro recall", "glass_break R", "horn_skid R", "Verdict"],
          rows, MARGIN, 2.14, W - 2 * MARGIN, col_w=[3.2, 1.4, 1.4, 1.5, 1.6, 1.5, 3.5], fs=10.2, hfs=10.5, row_h=0.50)

    tf = textbox(s, MARGIN, 4.28, 6.1, 3.0)
    para(tf, "THE DESIGN DECISION", size=11, bold=True, color=ACCENT, first=True, space_after=6)
    para(tf, "Maximising macro-F1 without constraints produces the best-looking headline numbers "
             "(+1.8 accuracy, +3.8 macro-F1) — and is the wrong answer. It buys that precision by "
             "collapsing horn_skid recall to 0.340, i.e. missing two thirds of those events.",
         size=11.5, space_after=6, line=1.12)
    para(tf, "That contradicts RQ4 of the proposal: a missed danger event costs far more than a "
             "false alarm. The shipped calibration therefore maximises macro-F1 subject to a floor "
             "on every hazard class’s recall.", size=11.5, space_after=6, line=1.12)
    para(tf, "Chosen bias β = [+0.4, −0.2, 0.0, −0.6, 0.0]", size=11.5, bold=True, color=TEAL, line=1.1)

    tf = textbox(s, 7.05, 4.28, W - 7.05 - MARGIN, 0.35)
    para(tf, "STEP 2 — MORE DATA FOR THE THIN CLASSES", size=11, bold=True, color=ACCENT, first=True)
    rows2 = [
        ["glass_break", "0.283", "**0.684", "**2.4×"],
        ["horn_skid", "0.246", "**0.832", "**3.4×"],
        ["alarm", "0.694", "**0.910", "+0.216"],
        ["distress_call", "0.840", "**0.915", "+0.075"],
        ["ambience", "0.928", "0.903", "−0.025"],
        ["**macro precision", "**0.598", "**0.849", "**+0.251"],
        ["**accuracy", "**0.848", "**0.898", "**+5.0 pts"],
    ]
    table(s, ["Precision", "40-clip classes", "expanded data", "change"], rows2, 7.05, 4.64,
          W - 7.05 - MARGIN, col_w=[2.0, 1.7, 1.7, 1.3], fs=9.6, hfs=9.2, row_h=0.275)
    tf = textbox(s, 7.05, 6.97, W - 7.05 - MARGIN, 0.5)
    para(tf, "UrbanSound8K supplied 167 car-horn and 435 siren clips, plus 630 urban hard "
             "negatives; DataSEC and Freesound CC0 added 73 glass clips. Calibration alone could "
             "not fix precision — the data scarcity had to go.",
         size=10, italic=True, color=TEAL, first=True, line=1.05)
    return s


def s_dataset(prs):
    s = slide_with_header(prs, "Dataset — sources, construction & novelty", "10", 3)
    rows = [
        ["ESC-50", "glass_break, horn_skid (car horn), alarm (clock alarm, siren), ambience",
         "2,000 clips · 50 categories · 5 s each", "CC BY-NC 3.0",
         "github.com/karolpiczak/ESC-50"],
        ["Human Screaming Detection (Kaggle)", "distress_call — primary source",
         "screaming vs non-screaming clips", "Kaggle terms",
         "kaggle.com/datasets/whats2000/\nhuman-screaming-detection-dataset"],
        ["RAVDESS speech (Zenodo)", "distress_call — supplement only (emotion 05 angry, 06 fearful)",
         "24 actors · acted emotional speech", "CC BY-NC-SA 4.0", "zenodo.org/record/1188976"],
    ]
    table(s, ["Source", "Classes contributed", "Scale", "Licence", "URL"], rows, MARGIN, 1.26,
          W - 2 * MARGIN, col_w=[2.6, 3.6, 2.5, 1.7, 3.7], fs=9.8, hfs=10.4, row_h=0.62)

    tf = textbox(s, MARGIN, 3.92, 6.1, 3.0)
    para(tf, "CONSTRUCTION & CHALLENGES ADDRESSED", size=11, bold=True, color=ACCENT, first=True, space_after=6)
    for b in ["Resampled to 16 kHz mono and denoised before feature extraction.",
              "Windowed into 1 s segments at 50% overlap → 18,130 windows "
              "(12,619 train / 2,712 val / 2,799 test).",
              "Split performed at SOURCE-FILE level, not window level — no window from one "
              "recording can appear in two splits, which would inflate accuracy.",
              "Ambience deliberately capped; class-weighted loss compensates for the residual "
              "2–70× imbalance.",
              "Ethics: only public, licensed, acted recordings — no real victims, no "
              "non-consenting bystanders."]:
        para(tf, b, size=11.2, bullet="▪", space_after=6, line=1.1)

    tf = textbox(s, 7.05, 3.92, W - 7.05 - MARGIN, 1.6)
    para(tf, "NOVELTY OF THE DATASET", size=11, bold=True, color=ACCENT, first=True, space_after=6)
    para(tf, "The contribution is the assembled 5-class safety taxonomy: no single public corpus "
             "covers distress_call, glass_break, horn_skid, alarm and ambience together. Three "
             "sources with different licences, sample rates and labelling conventions are unified "
             "into one consistently-windowed, source-disjoint benchmark.",
         size=11.2, line=1.12)
    bar(s, 7.05, 5.50, W - 7.05 - MARGIN, 1.20, LIGHT)
    tf = textbox(s, 7.25, 5.64, W - 7.25 - MARGIN - 0.2, 1.05)
    para(tf, "HONEST LIMITATION", size=10, bold=True, color=ACCENT, first=True, space_after=4)
    para(tf, "The proposal planned self-recorded, India-context campus audio. Under the deadline the "
             "implementation used public datasets only, and distress_call leans on acted/emotional "
             "speech. Self-collection is stated future work — not claimed as done.",
         size=10.5, line=1.08)
    footer(s, "IEEE DataPort: the datasets actually used are hosted on GitHub / Kaggle / Zenodo, not IEEE DataPort — "
              "confirm with the reviewer whether an IEEE DataPort URL is mandatory for this item.")
    return s


def s_ui(prs, shots):
    s = slide_with_header(prs, "UI screens — mobile web application", "11", 5)
    names = [("ui_home_listening.png", "Home — armed, monitoring"),
             ("ui_home_hazard.png", "Home — hazard detected"),
             ("ui_history.png", "Alert history"),
             ("ui_settings.png", "Settings — server & connection")]
    x = MARGIN
    box_w = (W - 2 * MARGIN - 3 * 0.30) / 4
    for fn, cap in names:
        p = shots / fn
        if p.exists():
            picture_fit(s, p, x, 1.28, box_w, 4.5)
        tf = textbox(s, x, 5.92, box_w, 0.5)
        para(tf, cap, size=11, bold=True, align=PP_ALIGN.CENTER, first=True)
        x += box_w + 0.30
    footer(s, "Live screenshots from the running application (FastAPI + static web app), not mockups. Three screens: "
              "Home (arm/disarm + live status), History (persisted alerts), Settings (server URL + connectivity test).")
    return s


def s_standard_paper(prs):
    s = slide_with_header(prs, "Standard paper for the application", "12", 2)
    bar(s, MARGIN, 1.30, W - 2 * MARGIN, 1.42, LIGHT)
    tf = textbox(s, MARGIN + 0.26, 1.48, W - 2 * MARGIN - 0.5, 1.2)
    para(tf, STANDARD_PAPER["title"], size=17, bold=True, first=True, space_after=5)
    para(tf, STANDARD_PAPER["authors"], size=12.5, color=MUTED, space_after=3)
    para(tf, f"{STANDARD_PAPER['venue']}  ·  DOI {STANDARD_PAPER['doi']}  ·  {STANDARD_PAPER['citations']}",
         size=11.5, color=TEAL)

    tf = textbox(s, MARGIN, 3.02, W - 2 * MARGIN, 3.0)
    para(tf, "JUSTIFICATION", size=11, bold=True, color=ACCENT, first=True, space_after=8)
    for b in STANDARD_PAPER["why"]:
        para(tf, b, size=12.5, bullet="▪", space_after=9, line=1.12)

    bar(s, MARGIN, 6.12, W - 2 * MARGIN, 0.78, LIGHT)
    tf = textbox(s, MARGIN + 0.26, 6.28, W - 2 * MARGIN - 0.5, 0.55)
    para(tf, STANDARD_PAPER["alt"], size=10.5, italic=True, color=MUTED, first=True, line=1.08)
    return s


def s_close(prs):
    s = slide_with_header(prs, "Status, limitations & next steps")
    tf = textbox(s, MARGIN, 1.30, 6.1, 5.4)
    para(tf, "DELIVERED", size=11, bold=True, color=TEAL, first=True, space_after=6)
    for b in ["Full pipeline: acquisition → preprocessing → features → 3 trained architectures "
              "→ Optuna search → evaluation → quantisation → calibration → serving.",
              "Champion model: log-mel CRNN — 89.8% test accuracy, 0.849 macro precision, "
              "88.2% distress-call recall, 516 KB quantised, 11.6 ms per window.",
              "Working demo: FastAPI server + mobile web app, verified end-to-end on held-out clips.",
              "Reproducible: every number in this deck regenerates from scripts in the repository."]:
        para(tf, b, size=11.8, bullet="▪", space_after=7, line=1.1)

    tf = textbox(s, 7.05, 1.30, W - 7.05 - MARGIN, 5.4)
    para(tf, "LIMITATIONS & NEXT STEPS", size=11, bold=True, color=ACCENT, first=True, space_after=6)
    for b in ["Dataset: public corpora only; distress_call leans on acted emotional speech. "
              "Self-recorded India-context audio is the most valuable next addition.",
              "glass_break and horn_skid have only 40 source clips each — precision on them stays "
              "low. More data for these two classes is the highest-leverage improvement available.",
              "Serving is a LAN-reachable laptop process, not in-phone inference; TFLite / ONNX "
              "Runtime Mobile export is the documented next step.",
              "Dynamic PTQ cannot quantise conv layers — static quantisation or QAT would compress "
              "the CNN branch too.",
              "Alarm recall fell 6.9 pts under calibration; a floor tuned on a larger validation "
              "set would likely recover it."]:
        para(tf, b, size=11.8, bullet="▪", space_after=7, line=1.1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default=None, help="directory containing ui_*.png screenshots")
    ap.add_argument("--out", default=str(ROOT / "docs" / "SafeScape_Review_Presentation.pptx"))
    args = ap.parse_args()
    shots = Path(args.shots) if args.shots else FIGS

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    # python-pptx's default template carries its own author in the file properties
    # ("Steve Canny", plus a "generated using python-pptx" comment). Overwrite them so
    # PowerPoint's File > Properties shows this project's authors instead.
    cp = prs.core_properties
    cp.author = "; ".join(name for name, _, _ in TEAM)
    cp.last_modified_by = TEAM[-1][0]
    cp.title = "SafeScape - On-Device Acoustic Distress and Hazard Recognition"
    cp.subject = COURSE
    cp.comments = ""

    s_title(prs)
    s_problem(prs)
    for key, name in LIT_OWNER:
        s_literature(prs, key, name)
    s_system_arch(prs)
    s_modules(prs)
    s_metrics(prs)
    s_dl_arch(prs)
    s_complexity(prs)
    s_algorithm(prs)
    s_hyperparams(prs)
    s_results(prs)
    s_calibration(prs)
    s_dataset(prs)
    s_ui(prs, shots)
    s_standard_paper(prs)
    s_close(prs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"wrote {out} with {len(prs.slides._sldIdLst)} slides")


if __name__ == "__main__":
    main()
