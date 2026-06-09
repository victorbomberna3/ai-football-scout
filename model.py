"""
Two-tower fit-prediction model.

Architecture rationale: a player has intrinsic traits (technical, physical,
pressing-inclination, etc.) that exist independently of where they play. A
club has a tactical identity that exists independently of who's in the squad.
"Fit" is the interaction of those two — which is exactly what a two-tower
model with a head MLP is built to capture.

Bonus side-effect: once trained, the player tower gives you a learned embedding
space for "find similar players" — without retraining, and grounded in actual
transfer outcomes rather than off-the-shelf sentence embeddings of stat lists.

The towers are deliberately small (sub-100k params total). With ~8k training
transfers, anything bigger overfits. Production scaling comes from more
transfers, more features, and longer history — not deeper nets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# --- feature columns ---------------------------------------------------------
PLAYER_STAT_COLS = [
    # Real from TM appearances (consistent train/inference)
    "goals_p90", "assists_p90",
    "yellow_cards_p90",      # aggression/pressing proxy (TM)
    "avg_min_per_game",      # starter vs sub (coach trust)
    "apps_pct_season",       # starter rate (n_apps/38, capped 0-1)
    # Proxied from real data (consistent train/inference)
    "npxg_p90", "key_passes_p90",
    # Career trajectory (from valuations)
    "mv_momentum_12m",
    # Injury risk — computed relative to transfer date so no temporal leakage
    "injury_days_last_2y",
    "has_serious_injury",
    # Excluded: contract_months_remaining — TM only exposes the *current* contract,
    # so training rows get a future contract signed after the transfer (temporal leakage).
    # Excluded: tkl_won_p90, interceptions_p90 — FBref misc match rate is ~47% in
    # training (non-Big-5 fallback rows have no FBref data), but 64% in inference pool.
    # Partial coverage with position-average fill creates train/inference mismatch.
    # These are kept in STAT_COLS for display in the UI (radar charts, Data Explorer).
    # Excluded: prog_passes_p90, pass_completion_pct, take_ons_p90,
    # prog_carries_p90, aerials_won_p90 — position×market-value heuristics;
    # FBref passing/possession pages use JS rendering, not available via static HTML.
]
CLUB_STAT_COLS = ["ppda", "possession_pct", "directness_idx", "line_height_m"]
POSITIONS = ["GK", "DEF", "MID", "ATT"]
LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]


@dataclass
class FeatureConfig:
    """Stored feature pipeline state — scalers, column orderings."""
    player_scaler: StandardScaler
    club_scaler: StandardScaler
    ctx_scaler: StandardScaler
    player_cols: list[str]
    club_cols: list[str]
    ctx_cols: list[str]


def build_player_features(df: pd.DataFrame, fit: FeatureConfig | None = None) -> tuple[np.ndarray, FeatureConfig]:
    """Player features: per-90 stats + log(market_value) + position one-hot.

    Two features intentionally excluded:
    - age: already in transfer context (transfer_age_norm); including it here
      creates a duplicate that dominates predictions producing age-sorted rankings.
    - orig_lg_* one-hots: bake in a ~10pt Premier League origin bonus regardless
      of player quality. Destination league is already in the club tower, so
      cross-league fit is captured without penalising Ligue 1/Bundesliga players.
    """
    feats = df[PLAYER_STAT_COLS].copy()
    feats["log_value"] = np.log1p(df["market_value_eur_m"])
    for p in POSITIONS:
        feats[f"pos_{p}"] = (df["position_group"] == p).astype(float)

    cols = feats.columns.tolist()
    X = feats.values.astype(np.float32)
    if fit is None:
        scaler = StandardScaler().fit(X)
        cfg = FeatureConfig(scaler, None, None, cols, [], [])
    else:
        scaler = fit.player_scaler
        cfg = fit
    Xs = scaler.transform(X).astype(np.float32)
    return Xs, cfg


def build_club_features(df: pd.DataFrame, fit: FeatureConfig | None = None) -> tuple[np.ndarray, FeatureConfig]:
    """Club features: tactical stats + league one-hot."""
    feats = df[CLUB_STAT_COLS].copy()
    for lg in LEAGUES:
        feats[f"lg_{lg}"] = (df["league"] == lg).astype(float)
    cols = feats.columns.tolist()
    X = feats.values.astype(np.float32)
    if fit is None or fit.club_scaler is None:
        scaler = StandardScaler().fit(X)
        if fit is None:
            cfg = FeatureConfig(None, scaler, None, [], cols, [])
        else:
            cfg = FeatureConfig(fit.player_scaler, scaler, fit.ctx_scaler, fit.player_cols, cols, fit.ctx_cols)
    else:
        scaler = fit.club_scaler
        cfg = fit
    Xs = scaler.transform(X).astype(np.float32)
    return Xs, cfg


def build_transfer_context(transfers_df: pd.DataFrame, fit: FeatureConfig | None = None) -> tuple[np.ndarray, FeatureConfig]:
    """Transfer-level context: fee only.

    transfer_age_norm intentionally excluded — it was the model's dominant
    predictor at inference (corr +0.72 with fit_score), sorting recommendations
    purely by age. Age filtering is handled upstream by Filters(min_age, max_age).
    """
    feats = pd.DataFrame({
        "log_fee": np.log1p(transfers_df["fee_eur_m"]),
    })
    cols = feats.columns.tolist()
    X = feats.values.astype(np.float32)
    if fit is None or fit.ctx_scaler is None:
        scaler = StandardScaler().fit(X)
        if fit is None:
            cfg = FeatureConfig(None, None, scaler, [], [], cols)
        else:
            cfg = FeatureConfig(fit.player_scaler, fit.club_scaler, scaler, fit.player_cols, fit.club_cols, cols)
    else:
        scaler = fit.ctx_scaler
        cfg = fit
    Xs = scaler.transform(X).astype(np.float32)
    return Xs, cfg


def assemble_training_arrays(
    players_df: pd.DataFrame,
    clubs_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    fit: FeatureConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FeatureConfig]:
    """
    Returns (player_X, club_X, ctx_X, y, config) aligned to transfers_df row order.

    If transfers_df already contains temporally-aligned player stat columns
    (written there by build_transfers_df), those are used directly — this is
    the temporally-correct path. Otherwise falls back to the old join on
    player_id (2024-25 snapshot for all transfers).
    """
    has_embedded = all(c in transfers_df.columns for c in PLAYER_STAT_COLS)

    if has_embedded:
        # Use season-aligned stats embedded in each transfer row
        transfer_player_df = transfers_df.copy()
        # build_player_features expects "age" not "transfer_age"
        if "age" not in transfer_player_df.columns and "transfer_age" in transfer_player_df.columns:
            transfer_player_df["age"] = transfer_player_df["transfer_age"]
        p_X_all, cfg = build_player_features(transfer_player_df, fit)
        player_X = p_X_all
    else:
        # Legacy path: look up 2024-25 stats by player_id
        p_X_all, cfg = build_player_features(players_df, fit)
        p_pos = {pid: i for i, pid in enumerate(players_df["player_id"].to_numpy())}
        player_X = np.stack([p_X_all[p_pos[pid]] for pid in transfers_df["player_id"]])

    c_X_all, cfg = build_club_features(clubs_df, cfg)
    ctx_X, cfg = build_transfer_context(transfers_df, cfg)

    c_pos = {cid: i for i, cid in enumerate(clubs_df["club_id"].to_numpy())}
    club_X = np.stack([c_X_all[c_pos[cid]] for cid in transfers_df["destination_club_id"]])
    y = transfers_df["success_score"].values.astype(np.float32)
    return player_X, club_X, ctx_X, y, cfg


# --- model -------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TwoTowerFitModel(nn.Module):
    """
    Player tower → 32-d player embedding  (15 feats: 10 stat cols + log_value + 4 pos)
    Club tower   → 32-d club embedding
    Head: concat(p_emb ⊙ c_emb, |p_emb − c_emb|, transfer_ctx) → MLP → sigmoid fit score
    """
    def __init__(self, n_player_feats: int, n_club_feats: int, n_ctx_feats: int,
                 emb_dim: int = 32, head_hidden: int = 64, dropout: float = 0.15):
        super().__init__()
        self.player_tower = _MLP([n_player_feats, 64, emb_dim], dropout)
        self.club_tower = _MLP([n_club_feats, 32, emb_dim], dropout)
        self.head = _MLP([emb_dim * 2 + n_ctx_feats, head_hidden, head_hidden, 1], dropout)

    def player_embedding(self, x):
        return self.player_tower(x)

    def club_embedding(self, x):
        return self.club_tower(x)

    def forward(self, player_x, club_x, ctx_x):
        p_emb = self.player_tower(player_x)
        c_emb = self.club_tower(club_x)
        # Bilinear: product captures alignment, abs-diff captures gap
        combined = torch.cat([p_emb * c_emb, (p_emb - c_emb).abs(), ctx_x], dim=-1)
        return torch.sigmoid(self.head(combined)).squeeze(-1)


# --- training loop -----------------------------------------------------------

def _player_split(
    n: int, player_ids: np.ndarray, val_frac: float, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split row indices so no player appears in more than one partition."""
    unique_pids = np.unique(player_ids)
    rng = np.random.default_rng(seed)
    shuffled = unique_pids[rng.permutation(len(unique_pids))]
    n_test_p = int(len(shuffled) * test_frac)
    n_val_p  = int(len(shuffled) * val_frac)
    test_pids  = set(shuffled[:n_test_p])
    val_pids   = set(shuffled[n_test_p:n_test_p + n_val_p])
    rows = np.arange(n)
    flags = np.array([
        0 if pid in test_pids else (1 if pid in val_pids else 2)
        for pid in player_ids
    ])
    return rows[flags == 2], rows[flags == 1], rows[flags == 0]  # train, val, test


def train_two_tower(
    player_X: np.ndarray, club_X: np.ndarray, ctx_X: np.ndarray, y: np.ndarray,
    player_ids: np.ndarray | None = None,
    val_frac: float = 0.15, test_frac: float = 0.15,
    epochs: int = 300, batch_size: int = 256, lr: float = 5e-4,
    weight_decay: float = 1e-4, patience: int = 25, seed: int = 0,
    verbose: bool = True,
) -> tuple[TwoTowerFitModel, dict]:
    """Train with early stopping on validation loss. Returns model + metrics dict."""
    torch.manual_seed(seed)
    n = len(y)
    if player_ids is not None:
        train_idx, val_idx, test_idx = _player_split(n, player_ids, val_frac, test_frac, seed)
    else:
        perm = np.random.default_rng(seed).permutation(n)
        n_test = int(n * test_frac)
        n_val  = int(n * val_frac)
        test_idx  = perm[:n_test]
        val_idx   = perm[n_test:n_test + n_val]
        train_idx = perm[n_test + n_val:]

    def _t(a, idx): return torch.from_numpy(a[idx])

    train_data = [_t(player_X, train_idx), _t(club_X, train_idx),
                  _t(ctx_X, train_idx), _t(y, train_idx)]
    val_data = [_t(player_X, val_idx), _t(club_X, val_idx),
                _t(ctx_X, val_idx), _t(y, val_idx)]
    test_data = [_t(player_X, test_idx), _t(club_X, test_idx),
                 _t(ctx_X, test_idx), _t(y, test_idx)]

    model = TwoTowerFitModel(player_X.shape[1], club_X.shape[1], ctx_X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )
    loss_fn = nn.MSELoss()

    n_train = len(train_idx)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            b = order[start:start + batch_size]
            pred = model(train_data[0][b], train_data[1][b], train_data[2][b])
            loss = loss_fn(pred, train_data[3][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(b)
        epoch_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_pred = model(val_data[0], val_data[1], val_data[2])
            val_loss = loss_fn(val_pred, val_data[3]).item()
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"  Early stop @ epoch {epoch}")
                break

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train {epoch_loss:.4f} | val {val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(test_data[0], test_data[1], test_data[2]).numpy()
    test_y = test_data[3].numpy()
    metrics = {
        "test_mse": float(np.mean((test_pred - test_y) ** 2)),
        "test_mae": float(np.mean(np.abs(test_pred - test_y))),
        "test_r2": float(1 - np.var(test_pred - test_y) / np.var(test_y)),
        "test_spearman": float(pd.Series(test_pred).corr(pd.Series(test_y), method="spearman")),
        "best_val_mse": best_val,
        "history": history,
        "test_pred": test_pred,
        "test_y": test_y,
        "test_idx": test_idx,
    }
    return model, metrics
