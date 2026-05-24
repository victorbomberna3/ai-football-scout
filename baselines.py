"""
Baselines for honest comparison against the two-tower model.

1. CosineBaseline   — emulates the v2 approach: encode player profile and
                      destination club style as flat vectors, score by cosine.
                      Captures correlations but NOT interactions.

2. LinearBaseline   — linear regression on concatenated (player ⊕ club ⊕ ctx)
                      features. Captures additive effects, not interactions.

3. GBMBaseline      — LightGBM on the same flat features. Should pick up
                      interactions; tests whether the two-tower's structural
                      prior adds anything over a strong off-the-shelf model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb


def _split(n: int, val_frac: float, test_frac: float, seed: int):
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    return perm[n_test + n_val:], perm[n_test:n_test + n_val], perm[:n_test]


def _metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    return {
        "test_mse": float(np.mean((y_pred - y_true) ** 2)),
        "test_mae": float(np.mean(np.abs(y_pred - y_true))),
        "test_r2": float(1 - np.var(y_pred - y_true) / np.var(y_true)),
        "test_spearman": float(pd.Series(y_pred).corr(pd.Series(y_true), method="spearman")),
    }


class CosineBaseline:
    """
    Mimics the v2 embedding approach.

    To get a fair test of what the v2 cosine-of-text idea is actually doing
    underneath, we drop the text layer (MiniLM) and use the raw player/club
    feature vectors. Player and club are projected to the same dimension via
    a fixed random projection so cosine is well-defined. The point isn't to
    beat the v2 system precisely — it's to demonstrate that cosine-of-flat-
    features cannot pick up the interaction signal in the data.
    """
    def __init__(self, dim: int = 32, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def fit(self, player_X, club_X, ctx_X, y):
        rng = np.random.default_rng(self.seed)
        self.W_p = rng.standard_normal((player_X.shape[1], self.dim)).astype(np.float32)
        self.W_c = rng.standard_normal((club_X.shape[1], self.dim)).astype(np.float32)
        # Calibrate raw cosine → success_score scale via simple linear fit
        sims = self._sim(player_X, club_X)
        # 1D linear: y = a*sim + b
        a, b = np.polyfit(sims, y, 1)
        self.a, self.b = a, b
        return self

    def _sim(self, p, c):
        pe = p @ self.W_p
        ce = c @ self.W_c
        pe /= np.linalg.norm(pe, axis=1, keepdims=True) + 1e-9
        ce /= np.linalg.norm(ce, axis=1, keepdims=True) + 1e-9
        return (pe * ce).sum(axis=1)

    def predict(self, player_X, club_X, ctx_X):
        return np.clip(self.a * self._sim(player_X, club_X) + self.b, 0, 1)


class LinearBaseline:
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, player_X, club_X, ctx_X, y):
        X = np.concatenate([player_X, club_X, ctx_X], axis=1)
        self.model.fit(X, y)
        return self

    def predict(self, player_X, club_X, ctx_X):
        X = np.concatenate([player_X, club_X, ctx_X], axis=1)
        return np.clip(self.model.predict(X), 0, 1)


class GBMBaseline:
    def __init__(self, n_estimators: int = 400, lr: float = 0.05, num_leaves: int = 31, seed: int = 0):
        self.params = dict(
            objective="regression", metric="mse", n_estimators=n_estimators,
            learning_rate=lr, num_leaves=num_leaves, random_state=seed,
            verbose=-1,
        )

    def fit(self, player_X, club_X, ctx_X, y, val_idx=None):
        X = np.concatenate([player_X, club_X, ctx_X], axis=1)
        self.model = lgb.LGBMRegressor(**self.params)
        if val_idx is not None:
            train_mask = np.ones(len(y), bool); train_mask[val_idx] = False
            self.model.fit(
                X[train_mask], y[train_mask],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
        else:
            self.model.fit(X, y)
        return self

    def predict(self, player_X, club_X, ctx_X):
        X = np.concatenate([player_X, club_X, ctx_X], axis=1)
        return np.clip(self.model.predict(X), 0, 1)


def evaluate_baselines(
    player_X: np.ndarray, club_X: np.ndarray, ctx_X: np.ndarray, y: np.ndarray,
    val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0,
) -> dict:
    """Train all baselines on the same train/val/test split and return metrics."""
    train_idx, val_idx, test_idx = _split(len(y), val_frac, test_frac, seed)

    def _slice(arr, idx): return arr[idx]

    cos = CosineBaseline().fit(
        _slice(player_X, train_idx), _slice(club_X, train_idx),
        _slice(ctx_X, train_idx), _slice(y, train_idx)
    )
    lin = LinearBaseline().fit(
        _slice(player_X, train_idx), _slice(club_X, train_idx),
        _slice(ctx_X, train_idx), _slice(y, train_idx)
    )
    gbm = GBMBaseline()
    # GBM gets val for early stopping; combine train+val and pass val_idx in local indexing
    X_full = (player_X, club_X, ctx_X)
    trainval_idx = np.concatenate([train_idx, val_idx])
    # rebuild local val indices inside trainval
    local_val = np.arange(len(train_idx), len(trainval_idx))
    gbm.fit(
        _slice(player_X, trainval_idx), _slice(club_X, trainval_idx),
        _slice(ctx_X, trainval_idx), _slice(y, trainval_idx),
        val_idx=local_val,
    )

    out = {}
    for name, model in [("cosine", cos), ("linear", lin), ("gbm", gbm)]:
        pred = model.predict(
            _slice(player_X, test_idx), _slice(club_X, test_idx), _slice(ctx_X, test_idx)
        )
        out[name] = _metrics(pred, _slice(y, test_idx))
        out[name]["test_pred"] = pred
        out[name]["model"] = model
    out["_test_idx"] = test_idx
    out["_test_y"] = _slice(y, test_idx)
    return out
