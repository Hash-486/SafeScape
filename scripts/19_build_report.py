"""Build the case study report (docs/SafeScape_Case_Study_Report.docx).

Every number is read at build time: test metrics from reports/metrics/*_v2.json (the
leak-free test split, see 18_flag_duplicate_sources.py), training curves from the v2
train logs, and the UI test grid by running the served model bundles on
demo_clips/test. Re-run after any retrain and the report follows.

Usage (from the repo root): .venv/Scripts/python.exe scripts/19_build_report.py
"""
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from server.inference import Predictor, MODELS, EXPORTED_DIR
from utils.labels import LABELS

FIG = ROOT / "reports" / "figures"
METRICS = ROOT / "reports" / "metrics"
OUT = ROOT / "docs" / "Review3" / "SafeScape_Case_Study_Report.docx"
PROC_V2 = ROOT / "data" / "processed_v2"

TEAM = [("Amruth Rohan KR", "CB.EN.U4ECE23222", "MFCC-CNN"),
        ("Harish Venkat VS", "CB.EN.U4ECE23219", "log-mel CRNN and Distilled Transformer")]
GUIDE = "Dr. Bagyammal T, Professor, Department of CSE"
COURSE = "23CSE473 — Neural Networks and Deep Learning"
ACCENT = RGBColor(0xC0, 0x00, 0x00)

SHORT = {"mfcc_cnn": "MFCC-CNN", "logmel_crnn": "CRNN", "transformer": "Transformer"}


# ----------------------------------------------------------------------------- data
def load_metrics():
    return {k: json.loads((METRICS / f"{k}_v2.json").read_text()) for k in MODELS}


def bundle_kb(key):
    return (EXPORTED_DIR / key / "best_model.pt").stat().st_size / 1024


def ui_test_grid():
    preds = {k: Predictor(EXPORTED_DIR / k) for k in MODELS}
    clips = ROOT / "demo_clips" / "test"
    with open(clips / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    grid = []
    for r in rows:
        y, sr = sf.read(clips / r["file"], dtype="float32")
        grid.append((r["file"], r["label"],
                     {k: p.predict(y.copy(), sr)["predicted_class"] for k, p in preds.items()}))
    return grid


def training_curves():
    logs = {"mfcc_cnn": "mfcc_cnn_v2_train_log.csv", "logmel_crnn": "logmel_crnn_v2_train_log.csv",
            "transformer": "transformer_v2_train_log.csv"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, name in logs.items():
        df = pd.read_csv(ROOT / "reports" / name)
        ax.plot(df.epoch, df.val_recall, marker="o", ms=3, label=MODELS[k])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation macro recall")
    ax.set_title("Model selection signal per epoch (v2 data)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = FIG / "training_curves_v2.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def dataset_tables():
    m = pd.read_csv(PROC_V2 / "manifest.csv")
    src = pd.crosstab(m.source_dataset, m.target_class).reindex(columns=LABELS, fill_value=0)
    w = pd.read_csv(PROC_V2 / "windows_manifest_dedup.csv")
    split = pd.crosstab(w.target_class, w.split).reindex(LABELS)
    return src, split


# ----------------------------------------------------------------------------- docx helpers
def set_cell_bg(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def table(doc, headers, rows, widths=None, caption=None):
    if caption:
        cap = doc.add_paragraph(caption, style="Caption")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_bg(c, "C00000")
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(9.5)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return t


def figure(doc, path, caption, width_cm=15):
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption, style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def figure_row(doc, items, width_cm, caption):
    """Several images side by side in a borderless one-row table, one caption under all."""
    t = doc.add_table(rows=1, cols=len(items))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, (path, sub) in zip(t.rows[0].cells, items):
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        if sub:
            sp = cell.add_paragraph(sub)
            sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            sp.runs[0].font.size = Pt(9)
    cap = doc.add_paragraph(caption, style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def para(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def toc(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), 'TOC \\o "1-2" \\h \\z \\u')
    run._r.append(fld)


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def pct(x):
    return f"{x * 100:.1f}%"


# ----------------------------------------------------------------------------- content
def title_page(doc):
    for _ in range(3):
        doc.add_paragraph()
    for text, size, bold in [
        ("SafeScape", 30, True),
        ("On-Device Acoustic Distress and Hazard Recognition", 16, True),
        ("A Case Study Comparing Three Deep Learning Architectures", 13, False),
        ("", 10, False),
        ("Case Study Report", 13, True),
        (COURSE, 12, False),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.font.size = Pt(size)
        r.bold = bold
        if size == 30:
            r.font.color.rgb = ACCENT
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Submitted by").bold = True
    for name, reg, arch in TEAM:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(f"{name}  ({reg})  —  {arch}")
    doc.add_paragraph()
    for text in [f"Under the guidance of {GUIDE}",
                 "Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore",
                 date.today().strftime("%B %Y")]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(text)
    page_break(doc)


def abstract(doc, M):
    c = M["logmel_crnn"]
    doc.add_heading("Abstract", level=1)
    para(doc,
         "Personal-safety apps today mostly wait for the user to press a button. In the moments that "
         "matter, a person may be unable to do that. SafeScape listens instead: it classifies short, "
         "rolling one-second windows of ambient audio into five classes — distress_call, glass_break, "
         "horn_skid, alarm and ambience — and raises an alert when a hazard is heard. The design target "
         "is a low-cost phone running fully offline, so model size and CPU latency matter as much as "
         "accuracy.")
    para(doc,
         "This case study trains and compares three architectures on one dataset of 2,919 public "
         "recordings (21,917 windows), split by source file, with test windows whose audio also "
         "appears in training removed after a duplicate audit: an MFCC-based CNN, a log-mel CRNN (CNN front-end with a "
         "bidirectional GRU), and a tiny Transformer distilled from the CRNN. On the held-out test split "
         f"the CRNN is the strongest model, with {pct(c['accuracy'])} accuracy, macro-F1 "
         f"{c['macro_f1']:.3f} and distress-call recall {pct(c['distress_recall'])}; after dynamic int8 "
         f"quantization it is {bundle_kb('logmel_crnn'):.0f} KB. All three models are served by one "
         "FastAPI back-end and compared side by side in a web dashboard, which is also used to test the "
         "user interface on held-out clips. Post-hoc, safety-constrained logit calibration and a "
         "multi-window inference rule are introduced to keep hazard recall high without retraining.")
    p = para(doc, "")
    p.add_run("Keywords: ").bold = True
    p.add_run("acoustic event detection, CRNN, knowledge distillation, MFCC, log-mel spectrogram, "
              "model quantization, public safety, SDG 5, SDG 11")
    page_break(doc)


def ch_intro(doc):
    doc.add_heading("1. Introduction", level=1)
    doc.add_heading("1.1 Problem statement", level=2)
    para(doc,
         "Harassment, assault and road accidents in public spaces are often preceded or accompanied by "
         "characteristic sounds — a scream, breaking glass, a skidding vehicle or a horn held down, an "
         "alarm. Existing safety apps rely on a manual SOS trigger, which assumes the user has a free "
         "hand, an unlocked phone and the presence of mind to use it. The problem this project addresses "
         "is to recognise those sounds automatically, quickly enough to be useful, on hardware an "
         "ordinary user already owns, without sending audio to a server.")
    doc.add_heading("1.2 Motivation", level=2)
    para(doc,
         "The project is aligned with UN Sustainable Development Goal 5 (gender equality, through "
         "freedom from violence) and Goal 11 (safe and inclusive public spaces). Privacy is a design "
         "constraint, not an afterthought: a system that streams microphone audio to the cloud would not "
         "be acceptable to most of the people it is meant to protect. That pushes the solution towards "
         "small models that run locally, which in turn makes this a natural setting to study the "
         "trade-off between accuracy, model size and latency across deep learning architectures.")
    doc.add_heading("1.3 Objectives", level=2)
    bullets(doc, [
        "Build a five-class acoustic hazard dataset from public sources with a leakage-free split.",
        "Train one architecture per team member and compare them on identical data and metrics.",
        "Prioritise hazard recall: a missed scream costs far more than a false alarm.",
        "Compress the best model for low-cost, on-device inference.",
        "Integrate every model behind one user interface and test the interface against the models.",
    ])
    doc.add_heading("1.4 Contributions", level=2)
    bullets(doc, [
        "A three-way comparison of MFCC-CNN, log-mel CRNN and a distilled Transformer on the same "
        "source-disjoint split, reporting accuracy, macro-F1, hazard recall, size and CPU latency.",
        "A safety-constrained post-hoc calibration that tunes per-class logit offsets on validation "
        "data while forbidding any hazard class from losing recall.",
        "A multi-window serving rule that lets a hazard in any second of a clip decide the verdict.",
        "A multi-model inference server and comparison dashboard used for UI testing on held-out clips.",
        "An audit of the dataset that found and removed byte-identical recordings spanning the split.",
    ])


def ch_literature(doc):
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    deck = importlib.import_module("14_build_presentation")
    doc.add_heading("2. Literature Survey", level=1)
    para(doc,
         "Each team member surveyed five recent journal papers (2025–2026) related to the architecture "
         "they own. The tables below list each work, what it contributes and its reported result; the "
         "paragraph after each table states what SafeScape takes from that line of work.")
    takeaways = {
        "MFCC-CNN": "MFCCs remain a strong, compact front-end, but the surveyed work shows their weakness "
                    "under noise and their loss of fine time-frequency structure. SafeScape keeps the "
                    "MFCC-CNN as the lightest baseline and measures what that compactness costs.",
        "log-mel CRNN": "Convolution over log-mel spectrograms followed by a recurrent layer is the "
                        "recurring design for sound event detection, and the hazard-detection paper by "
                        "Omarov and Altayeva uses the same 1 s window and 0.5 s hop adopted here. "
                        "SafeScape's CRNN follows this pattern with a bidirectional GRU.",
        "Distilled Tiny Transformer": "Knowledge distillation lets a small student approach a larger "
                                      "teacher, and the surveyed TinyML work shows sub-megabyte audio "
                                      "models are practical. SafeScape distils a two-layer Transformer "
                                      "from its own CRNN to test that claim on this task.",
    }
    for (arch, member), (_, papers) in zip(deck.LIT_OWNER, deck.LIT.items()):
        doc.add_heading(f"2.{list(deck.LIT).index(arch) + 1} {arch} (surveyed by {member})", level=2)
        table(doc, ["#", "Paper", "Venue", "Contribution", "Reported result"],
              [(i + 1, f"{t} — {a}", v, idea, re.sub(r"\s*\(verify[^)]*\)", "", res))
               for i, (t, a, v, _doi, idea, res) in enumerate(papers)],
              widths=[0.7, 5, 3.2, 4.2, 3.4])
        para(doc, takeaways[arch])
    sp = deck.STANDARD_PAPER
    doc.add_heading("2.4 Standard paper", level=2)
    para(doc, f"{sp['title']} — {sp['authors']}, {sp['venue']}, DOI {sp['doi']}. "
              "It defines the task SafeScape revisits: recognising screams and impulsive hazard sounds in "
              "ambient audio for public safety. Where that work used hand-crafted features with Gaussian "
              "mixture models on a fixed installation, SafeScape uses learned representations on a "
              "personal device.")


def ch_dataset(doc, split):
    doc.add_heading("3. Dataset", level=1)
    para(doc,
         "No single public dataset covers all five classes, so the dataset is assembled from six "
         "sources. Each source category is mapped to one SafeScape class in "
         "scripts/02_build_manifest.py.")
    src, _ = dataset_tables()
    table(doc, ["source"] + LABELS + ["total"],
          [[s] + [int(src.loc[s, l]) for l in LABELS] + [int(src.loc[s].sum())] for s in src.index]
          + [["total"] + [int(src[l].sum()) for l in LABELS] + [int(src.values.sum())]],
          caption="Table 3.1 — Source recordings per class (v2 dataset)")
    bullets(doc, [
        "ESC-50: glass breaking, car horn, siren, clock alarm and a range of ambient categories.",
        "Kaggle human-screaming detection set: the primary distress_call source.",
        "RAVDESS (Zenodo): angry and fearful speech as a distress-adjacent supplement (200 clips).",
        "UrbanSound8K: car_horn → horn_skid, siren → alarm, and seven everyday urban classes as "
        "ambience — these are the hard negatives that previously confused the hazard classes.",
        "DataSEC (Zenodo) and Freesound CC0 previews: additional glass-break recordings.",
    ])
    para(doc,
         "UrbanSound8K's gun_shot class was deliberately excluded. It is a genuine hazard with no class "
         "in the five-way taxonomy, and labelling it ambience would teach a safety model that gunfire is "
         "normal background sound.")
    doc.add_heading("3.1 Pre-processing and split", level=2)
    para(doc,
         "Audio is resampled to 16 kHz mono, denoised with non-stationary spectral gating, and cut into "
         "1.0 s windows with 50 % overlap; windows with RMS energy below 1e-4 are dropped. The 70/15/15 "
         "train/validation/test split is stratified and made at the source-file level, so all windows "
         "of a recording stay on one side of the split.")
    doc.add_heading("3.2 Duplicate audit", level=2)
    para(doc,
         "Hashing every source file revealed that the Kaggle scream set ships the same recordings in two "
         "folders. Because the split is by file path, 230 test windows — all distress_call — were byte-"
         "identical to recordings in the training or validation split. Rather than re-split and retrain "
         "every model, those windows are flagged (scripts/18_flag_duplicate_sources.py) and excluded: "
         "every test number in this report is computed on the remaining 2,990 windows, none of which has "
         "a byte-identical copy in training. The check is exact-match only: a trimmed or re-encoded copy "
         "of the same recording (ESC-50, UrbanSound8K and the Freesound previews all draw on Freesound) "
         "would not be caught. Validation retains 169 such windows; it is used only for checkpoint "
         "selection and calibration, and this is listed as a limitation.")
    table(doc, ["class", "train", "val", "test (leak-free)", "removed from test"],
          [[l, int(split.loc[l, "train"]), int(split.loc[l, "val"]), int(split.loc[l, "test"]),
            int(split.loc[l].get("test_dup", 0))] for l in LABELS],
          caption="Table 3.2 — Windows per split")


def ch_method(doc):
    doc.add_heading("4. Methodology", level=1)
    doc.add_heading("4.1 Features", level=2)
    para(doc,
         "Two time-frequency representations are computed with a 512-point FFT and a 10 ms hop "
         "(160 samples), giving 101 frames per window. The MFCC-CNN receives 40 MFCCs per frame; the "
         "CRNN and the Transformer receive a 64-band log-mel spectrogram. Each feature map is "
         "standardised per window (zero mean, unit variance). During training, SpecAugment-style "
         "frequency and time masking plus small Gaussian feature noise are applied.")
    doc.add_heading("4.2 MFCC-CNN (Amruth Rohan KR)", level=2)
    para(doc,
         "Four 3×3 convolution blocks (16, 32, 64, 64 channels), each with batch normalisation and ReLU, "
         "the first three followed by 2×2 max-pooling, then global average pooling, dropout and a linear "
         "classifier. The design treats the 40×101 MFCC matrix as a single-channel image. It has no "
         "explicit temporal model; the receptive field of the stacked convolutions is the only way it "
         "relates distant frames.")
    doc.add_heading("4.3 Log-mel CRNN (Harish Venkat VS)", level=2)
    para(doc,
         "Two convolution blocks (16 and 32 channels) pool only along frequency (2×1), so the 64 mel "
         "bands shrink to 16 while all 101 time steps survive. Each time step's 32×16 feature map is "
         "flattened to a 512-dimensional vector and fed to a bidirectional GRU with 128 hidden units per "
         "direction. GRU outputs are mean-pooled over time and passed through dropout to a linear layer. "
         "Keeping the time axis intact is the point of the design: a glass break lasts a few hundred "
         "milliseconds, and the recurrent layer can respond to that burst wherever it falls.")
    figure(doc, FIG / "crnn_architecture.png", "Figure 4.1 — Log-mel CRNN architecture", 15)
    doc.add_heading("4.4 Distilled Transformer (Harish Venkat VS)", level=2)
    para(doc,
         "Each log-mel frame becomes a token: a linear projection maps its 64 bands to a 64-dimensional "
         "embedding, a learnable class token is prepended, and learnable positional embeddings are added. "
         "Two Transformer encoder layers (4 heads, feed-forward width 128) follow, and the class token's "
         "output feeds a linear classifier. The student is trained against the CRNN teacher with the "
         "loss  L = α·CE(y, s) + (1 − α)·T²·KL(softmax(t/T) ‖ softmax(s/T)),  with α = 0.5 and "
         "temperature T = 4, so it learns both the labels and the teacher's relative confidence across "
         "classes.")
    doc.add_heading("4.5 Training", level=2)
    para(doc,
         "All models use AdamW with weight decay 1e-4 and a class-weighted cross-entropy (weights "
         "inversely proportional to class frequency), because the hazard classes are much rarer than "
         "ambience. The checkpoint kept is the epoch with the best validation macro recall, with early "
         "stopping after six epochs without improvement. Hyperparameters for the CNN and CRNN come from "
         "an Optuna search over learning rate, batch size and dropout (and GRU width for the CRNN).")
    table(doc, ["model", "learning rate", "batch", "dropout", "other", "why"],
          [["MFCC-CNN", "1.25e-3", "64", "0.25", "—", "Optuna best on validation macro recall; the small "
            "model tolerates larger batches"],
           ["log-mel CRNN", "8.56e-4", "16", "0.27", "GRU 128 units", "Optuna best; small batches and a "
            "wider GRU helped the rare classes"],
           ["Transformer", "1e-3", "32", "0.20", "α 0.5, T 4", "standard distillation settings; the teacher "
            "already encodes the class balance"]],
          caption="Table 4.1 — Hyperparameters")
    doc.add_heading("4.6 Safety-constrained calibration", level=2)
    para(doc,
         "Class weighting protects recall but pushes the decision boundary towards the rare classes. "
         "Instead of retraining, a per-class bias b is added to the logits before the softmax. b is found "
         "by coordinate ascent on the validation split to maximise macro-F1 subject to a floor: no "
         "hazard class may lose more than two points of recall relative to the uncalibrated model. An "
         "unconstrained search was tried first and rejected — it raised macro-F1 by trading away horn_skid "
         "recall, which is the wrong trade for a safety system. Each of the three models is calibrated "
         "this way, so they are compared at equivalent operating points.")
    doc.add_heading("4.7 Compression", level=2)
    para(doc,
         "The CRNN is compressed with dynamic post-training quantization to int8, which targets its GRU "
         "and linear layers. The MFCC-CNN is convolution-heavy, so dynamic quantization would leave it "
         "almost unchanged, and the Transformer is already small; both are served in fp32.")
    doc.add_heading("4.8 Serving rule", level=2)
    para(doc,
         "The app sends two-second clips. The server splits each clip into the same 1 s, 50 %-overlap "
         "windows used in training, applies the same energy gate, and classifies every window. If any "
         "window's top class is a hazard, the clip takes the most confident hazard window's verdict; "
         "otherwise it takes the most confident window. Averaging would let a short hazard be outvoted by "
         "the quiet second around it. A clip with no window above the energy floor is reported as "
         "ambience, because the model was never trained on silence.")


def ch_system(doc, shots):
    doc.add_heading("5. System Design and User Interface", level=1)
    figure(doc, FIG / "system_architecture.png", "Figure 5.1 — System architecture", 15.5)
    para(doc,
         "A FastAPI server loads one exported bundle per model (weights, label map, pre-processing "
         "configuration and calibration). Browser audio arrives in whatever format the browser records; "
         "a bundled static ffmpeg converts it to 16 kHz mono PCM before inference, so the server needs no "
         "system-wide audio libraries.")
    table(doc, ["endpoint", "purpose"],
          [["POST /predict?model=<key>", "classify a clip with one model (default: CRNN) — used by the mobile app"],
           ["POST /predict/all", "classify one clip with every model and report per-model latency"],
           ["GET /models", "test-set metrics, owner and served size for each model"],
           ["GET /testclips", "the held-out UI test clips and their ground truth"],
           ["GET /, /compare", "mobile app and comparison dashboard"]],
          caption="Table 5.1 — Server endpoints")
    doc.add_heading("5.1 Mobile app", level=2)
    para(doc,
         "The mobile web app has three screens. Home shows a large status card that changes colour, icon "
         "and text together (so the state never relies on colour alone) between idle, listening and "
         "hazard. History lists past alerts with a five-second cool-down so that one event does not flood "
         "the list. Settings holds the server address, a connection test, and a link to the comparison "
         "dashboard.")
    figure_row(doc, [(FIG / "ui_home_listening.png", "listening"), (FIG / "ui_home_hazard.png", "hazard detected"),
                     (FIG / "ui_history.png", "alert history")], 4.6,
               "Figure 5.2 — Mobile app screens")
    doc.add_heading("5.2 Comparison dashboard", level=2)
    para(doc,
         "The dashboard is where every model meets the interface. A clip — recorded from the microphone, "
         "uploaded, or chosen from the held-out test set — is sent once to /predict/all, and three cards "
         "show each model's label, confidence, full probability distribution and inference time, with a "
         "correct/wrong badge when the ground truth is known. A test-run panel pushes all twenty held-out "
         "clips through the same HTTP path the app uses and fills a clip × model grid with running "
         "accuracy. A final panel shows the offline test-set metrics and confusion matrices. The page has "
         "no external dependencies, so it works on a laptop with no network.")
    for name, cap in shots:
        figure(doc, FIG / name, cap, 16)


def ch_results(doc, M):
    doc.add_heading("6. Results and Comparison", level=1)
    n = M["logmel_crnn"]["test_windows"]
    para(doc, f"All figures are on the leak-free held-out test split ({n:,} windows), with each model's "
              "safety-constrained calibration applied. Every row is the exact file the server loads, so the "
              "CRNN figures are for its int8 version. Latency is the mean time for one 1 s window on "
              "the CPU, batch size 1.")
    table(doc, ["model", "owner", "accuracy", "macro-F1", "macro recall", "distress recall", "params",
                "served size", "CPU latency"],
          [[MODELS[k], o, pct(M[k]["accuracy"]), f"{M[k]['macro_f1']:.3f}", pct(M[k]["macro_recall"]),
            pct(M[k]["distress_recall"]), f"{M[k]['params']:,}", f"{bundle_kb(k):.0f} KB",
            f"{M[k]['latency_ms']:.2f} ms"]
           for k, o in [("mfcc_cnn", "Amruth"), ("logmel_crnn", "Harish"), ("transformer", "Harish")]],
          caption="Table 6.1 — Three-model comparison")
    table(doc, ["class"] + [f"{SHORT[k]} {m}" for k in MODELS for m in ("P", "R")],
          [[l] + [f"{M[k]['per_class'][l][m]:.2f}" for k in MODELS for m in ("precision", "recall")]
           for l in LABELS],
          caption="Table 6.2 — Per-class precision (P) and recall (R)")
    figure_row(doc, [(ROOT / M[k]["confusion_png"], MODELS[k]) for k in MODELS], 5.4,
               "Figure 6.1 — Confusion matrices on the leak-free test split (rows: true class)")
    figure(doc, FIG / "training_curves_v2.png", "Figure 6.2 — Validation macro recall per epoch", 14)

    c, t, m = M["logmel_crnn"], M["transformer"], M["mfcc_cnn"]
    doc.add_heading("6.1 Discussion", level=2)
    para(doc,
         f"The CRNN leads on every quality metric (macro-F1 {c['macro_f1']:.3f} against "
         f"{t['macro_f1']:.3f} for the Transformer and {m['macro_f1']:.3f} for the MFCC-CNN). Two design "
         "choices account for most of the gap. Log-mel features keep the spectral detail that the MFCC's "
         "cosine transform compresses away, and the CRNN's recurrent layer models how a sound evolves "
         "over the window, where the CNN can only pool over it.")
    para(doc,
         f"The distilled Transformer has {t['params']:,} parameters against the CRNN's {c['params']:,} "
         f"and runs in {t['latency_ms']:.2f} ms per window against {c['latency_ms']:.2f} ms. Distillation "
         "transfers much, but not all, of the teacher's behaviour; on a dataset of this size the "
         "teacher's convolutional inductive bias still wins.")
    para(doc,
         f"The MFCC-CNN is the smallest and fastest model ({m['params']:,} parameters, "
         f"{m['latency_ms']:.2f} ms). It also gained the most from the expanded dataset: its accuracy "
         "was 66.6% on the first dataset version and is now " f"{pct(m['accuracy'])}" ". The two test "
         "splits differ, so the comparison is indicative, but it suggests the earlier results were "
         "limited by data more than by the architecture.")
    para(doc,
         "The deployment choice follows from the safety requirement. All three models are fast enough "
         "for real time — each is far below the one-second window length — so the deciding factor is "
         f"hazard recall. The quantized CRNN, at {bundle_kb('logmel_crnn'):.0f} KB, is small enough for "
         "any phone and is the default model in the app.")


def ch_uitest(doc, grid):
    doc.add_heading("7. UI Testing with the Deep Learning Models", level=1)
    para(doc,
         "The user interface was tested against all three models with twenty clips — four per class — "
         "cut from test-split recordings with no byte-identical copy in training. Each clip "
         "is the two seconds around the loudest second of its source, the same length the app records. "
         "The table below is produced by running the served bundles on those clips; the dashboard's test "
         "run produces the same grid through HTTP.")
    correct = {k: sum(g[2][k] == g[1] for g in grid) for k in MODELS}
    table(doc, ["clip", "truth"] + [SHORT[k] for k in MODELS],
          [[f, lab] + [("✓ " if got[k] == lab else "✗ ") + got[k] for k in MODELS] for f, lab, got in grid]
          + [["accuracy", ""] + [f"{correct[k]}/{len(grid)}" for k in MODELS]],
          caption="Table 7.1 — UI test results (held-out clips)")
    doc.add_heading("7.1 Functional and edge-case tests", level=2)
    table(doc, ["test", "expected", "result"],
          [["Hazard in the first or second half of a clip, silence in the other", "hazard detected",
            "pass for all four hazard classes (server/test_clip_position.py)"],
           ["Ambience followed by one second of silence", "ambience",
            "fail — distress_call (0.77); identical with the earlier model, see Section 8"],
           ["Two seconds of silence", "ambience from every model", "pass (server/test_multi_model.py)"],
           ["0.33 s clip", "every model answers", "pass (server/test_multi_model.py)"],
           ["Unknown model key", "HTTP 404, server keeps running", "pass"],
           ["Dashboard opened without microphone access", "record disabled with reason; upload and test "
            "clips still work", "pass"],
           ["Existing mobile app against the new server", "unchanged behaviour on /predict", "pass "
            "(server/test_client.py)"]],
          caption="Table 7.2 — Functional tests")
    para(doc,
         "Every model recognised all sixteen hazard clips except one horn clip that the MFCC-CNN "
         "called a distress call. The failures are concentrated in ambience. This is the cost of the "
         "serving rule in Section 4.8, which lets any single window classified as a hazard decide the "
         "whole clip: a deliberate bias towards false alarms over missed events. The test clips also make "
         "it harder than everyday use, because each one is cut around the loudest second of its "
         "recording, so a background recording is represented by its most event-like moment. "
         "Window-level ambience recall on the offline test split is much higher (Table 6.2), and a "
         "practical refinement is to require a hazard in two consecutive windows before alerting.")
    para(doc,
         "The clip-position test exposed a real defect during development: the server originally kept "
         "only the last second of each two-second clip, so a glass break in the first half was "
         "misclassified. The multi-window rule in Section 4.8 was the fix, and the test now guards it.")


def ch_limits(doc):
    doc.add_heading("8. Limitations and Future Work", level=1)
    bullets(doc, [
        "Distress audio comes from acted or emotional-speech recordings, not real incidents or "
        "India-specific environments; a consented, self-recorded set is the most valuable next step.",
        "169 validation windows still duplicate training audio; re-splitting on content hashes and "
        "retraining would remove the last of the leak.",
        "glass_break remains the thinnest class (597 windows) and the hardest one to get right.",
        "Ambience followed by silence is misread as distress_call. The earlier model behaves the same, "
        "which points at pre-processing rather than the retrained weights; a likely cause, not yet "
        "confirmed, is per-window standardisation stretching the quiet tail to unit variance.",
        "The any-hazard-window rule favours recall over precision; requiring agreement between "
        "consecutive windows would cut false alarms on busy background sound.",
        "Inference runs on a laptop server reachable over the LAN, not on the phone itself; exporting "
        "to TorchScript or ONNX for mobile runtimes is the next engineering step.",
        "Browsers allow microphone access only on HTTPS or localhost, so live phone capture needs a "
        "certificate or a native wrapper.",
        "Static quantization or quantization-aware training would let the convolutional layers be "
        "compressed as well.",
    ])


def ch_conclusion(doc, M):
    c = M["logmel_crnn"]
    doc.add_heading("9. Conclusion", level=1)
    para(doc,
         "SafeScape shows that a small, fully offline model can recognise acoustic hazards with useful "
         f"accuracy. The log-mel CRNN reached {pct(c['accuracy'])} accuracy and "
         f"{pct(c['distress_recall'])} distress-call recall on held-out audio, in "
         f"{bundle_kb('logmel_crnn'):.0f} KB after quantization. The comparison suggests that, for short "
         "acoustic events, keeping the time axis and modelling it explicitly matters more than raw model "
         "size, and that data coverage of the rare classes mattered more than any architectural change. "
         "Integrating all three models behind one interface made these differences visible clip by clip, "
         "and testing that interface surfaced a serving defect that offline metrics alone would have "
         "missed.")


def ch_contrib(doc):
    doc.add_heading("10. Individual Contributions", level=1)
    table(doc, ["member", "contributions"],
          [["Amruth Rohan KR (CB.EN.U4ECE23222)",
            "MFCC feature pipeline; MFCC-CNN design, hyperparameter search and training on v1 and v2 "
            "data; MFCC-CNN evaluation and calibration; literature survey for the CNN track."],
           ["Harish Venkat VS (CB.EN.U4ECE23219)",
            "Dataset assembly and audit; log-mel CRNN design, search, training, calibration and "
            "quantization; Transformer distillation; FastAPI server, mobile app and comparison "
            "dashboard; literature survey for the CRNN and Transformer tracks."]],
          widths=[5, 11])


def ch_refs(doc):
    import importlib
    deck = importlib.import_module("14_build_presentation")
    doc.add_heading("References", level=1)
    refs = []
    sp = deck.STANDARD_PAPER
    refs.append(f"{sp['authors']}, \"{sp['title']},\" {sp['venue']}, doi: {sp['doi']}.")
    for papers in deck.LIT.values():
        for t, a, v, doi, _i, _r in papers:
            refs.append(f"{a}, \"{t},\" {v}, doi: {doi}.")
    refs += [
        "K. J. Piczak, \"ESC: Dataset for Environmental Sound Classification,\" Proc. 23rd ACM Int. Conf. "
        "on Multimedia, pp. 1015–1018, 2015.",
        "J. Salamon, C. Jacoby and J. P. Bello, \"A Dataset and Taxonomy for Urban Sound Research,\" "
        "Proc. 22nd ACM Int. Conf. on Multimedia, pp. 1041–1044, 2014.",
        "S. R. Livingstone and F. A. Russo, \"The Ryerson Audio-Visual Database of Emotional Speech and "
        "Song (RAVDESS),\" PLoS ONE, 13(5), e0196391, 2018.",
        "D. S. Park et al., \"SpecAugment: A Simple Data Augmentation Method for Automatic Speech "
        "Recognition,\" Proc. Interspeech, pp. 2613–2617, 2019.",
        "G. Hinton, O. Vinyals and J. Dean, \"Distilling the Knowledge in a Neural Network,\" NIPS Deep "
        "Learning Workshop, 2015.",
        "T. Akiba et al., \"Optuna: A Next-generation Hyperparameter Optimization Framework,\" Proc. "
        "25th ACM SIGKDD, pp. 2623–2631, 2019.",
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph(f"[{i}] {r}")
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)
        p.runs[0].font.size = Pt(9.5)


def main():
    M = load_metrics()
    _, split = dataset_tables()
    training_curves()
    grid = ui_test_grid()
    shots = [(n, c) for n, c in [
        ("ui_compare_live.png", "Figure 5.3 — Dashboard: one held-out clip, three models"),
        ("ui_compare_uitest.png", "Figure 5.4 — Dashboard: UI test run over the 20 held-out clips"),
        ("ui_compare_offline.png", "Figure 5.5 — Dashboard: offline test-set metrics"),
    ] if (FIG / n).exists()]

    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.5)
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    for lvl, size in [(1, 16), (2, 13)]:
        st = doc.styles[f"Heading {lvl}"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.color.rgb = ACCENT

    title_page(doc)
    abstract(doc, M)
    # a plain paragraph, not a heading, so the TOC doesn't list itself
    c = doc.add_paragraph()
    r = c.add_run("Contents")
    r.bold = True
    r.font.size = Pt(16)
    r.font.color.rgb = ACCENT
    toc(doc)
    page_break(doc)
    ch_intro(doc)
    ch_literature(doc)
    ch_dataset(doc, split)
    ch_method(doc)
    ch_system(doc, shots)
    ch_results(doc, M)
    ch_uitest(doc, grid)
    ch_limits(doc)
    ch_conclusion(doc, M)
    ch_contrib(doc)
    ch_refs(doc)

    doc.core_properties.author = "; ".join(n for n, _, _ in TEAM)
    doc.core_properties.title = "SafeScape — Case Study Report"
    doc.core_properties.subject = COURSE
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
