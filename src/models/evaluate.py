"""Classification metrics for the SLA-breach model.

SLA breach is a minority class (~20-25% of applications, see
data/README.md), so accuracy alone is a poor/misleading metric - a model
that always predicts "no breach" would already score ~75-80% accuracy while
being useless. ROC-AUC and PR-AUC (threshold-independent) are the primary
metrics; precision/recall/F1 use the default 0.5 threshold for reference.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """ROC-AUC, PR-AUC (average precision), precision, recall, F1, accuracy."""
    y_true = np.asarray(y_true)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
