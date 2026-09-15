"""Evaluation metrics: macro-recall (primary), full classification report, confusion matrix."""
import numpy as np
from sklearn.metrics import recall_score, classification_report, confusion_matrix

from .labels import LABELS


def macro_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro", zero_division=0)


def full_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=LABELS, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    return report, cm


def report_to_markdown(report, title=""):
    lines = [f"### {title}", "", "| class | precision | recall | f1 | support |", "|---|---|---|---|---|"]
    for label in LABELS:
        r = report[label]
        lines.append(f"| {label} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    lines.append(f"| **macro avg** | {report['macro avg']['precision']:.3f} | {report['macro avg']['recall']:.3f} | {report['macro avg']['f1-score']:.3f} | {int(report['macro avg']['support'])} |")
    lines.append(f"| **accuracy** | | | {report['accuracy']:.3f} | |")
    return "\n".join(lines)
