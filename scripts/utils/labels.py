"""Canonical class label order, shared by manifest building, training, and serving."""

LABELS = ["distress_call", "glass_break", "horn_skid", "alarm", "ambience"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
IDX_TO_LABEL = {i: l for i, l in enumerate(LABELS)}
HAZARD_LABELS = {"distress_call", "glass_break", "horn_skid", "alarm"}
