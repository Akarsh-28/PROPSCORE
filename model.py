"""
Training, cross-validated evaluation, and feature importance.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
import pandas as pd
import config


def new_model():
    """Fresh, untrained model with the settings from config.py."""
    return HistGradientBoostingClassifier(**config.MODEL_PARAMS)


def evaluate_with_cv(X, y):
    """
    5-fold cross-validated evaluation. Returns a dict of metrics.
    We use CV instead of a single train/test split because with so few
    converters, one unlucky split can give a misleading result.
    """
    skf = StratifiedKFold(
        n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    oof_proba = cross_val_predict(new_model(), X, y, cv=skf, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, oof_proba)

    # Business-relevant metric: if we prioritize the top K% of users by score,
    # how much better is that than picking users at random?
    k = int(config.TOP_K_FRACTION * len(y))
    top_k_idx = np.argsort(-oof_proba)[:k]
    pred_topk = np.zeros(len(y), dtype=int)
    pred_topk[top_k_idx] = 1

    precision_at_k = precision_score(y, pred_topk, zero_division=0)
    recall_at_k = recall_score(y, pred_topk, zero_division=0)
    base_rate = y.mean()

    return {
        "auc": auc,
        "base_rate": base_rate,
        "k": k,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "lift_at_k": precision_at_k / base_rate if base_rate > 0 else float("nan"),
    }


def fit_final_model(X, y):
    """Train on all available data -- used for scoring/deployment, not evaluation."""
    model = new_model()
    model.fit(X, y)
    return model


def get_feature_importance(model, X, y):
    """Permutation importance: how much AUC drops when each feature is shuffled."""
    perm = permutation_importance(
        model, X, y, n_repeats=15, random_state=config.RANDOM_STATE, scoring="roc_auc"
    )
    return (
        pd.DataFrame({"feature": X.columns, "importance": perm.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )