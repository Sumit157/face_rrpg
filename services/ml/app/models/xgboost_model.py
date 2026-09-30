"""XGBoost baseline (only used if xgboost is installed and justified by
results -- see spec section 9 "if justified")."""
from __future__ import annotations
import numpy as np


def train_xgboost(X: np.ndarray, y: np.ndarray, **kwargs):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "xgboost is not installed. `pip install xgboost` to use this baseline."
        ) from exc

    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
    params.update(kwargs)
    clf = XGBClassifier(**params)
    clf.fit(X, y)
    return clf
