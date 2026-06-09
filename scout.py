"""
Scout v3 — trained-model scoring layer.

Drop-in replacement for v2's `scoring.py`. Same interface (Filters dataclass,
apply_filters, compute_fit_score), but the fit score is now a learned
prediction of *transfer success* given a destination club style, not a
weighted z-score sum.

Usage:
    from scout import Filters, apply_filters, compute_fit_score, DestinationClub
    
    filters = Filters(position="ATT", max_value_eur_m=40, min_age=20, max_age=27)
    candidates = apply_filters(player_pool, filters)
    
    destination = DestinationClub(
        ppda=8.5, possession_pct=58, directness_idx=0.4, line_height_m=52,
        league="Premier League",
    )
    ranked = compute_fit_score(candidates, destination, transfer_age=23, fee_eur_m=30)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import (
    TwoTowerFitModel, FeatureConfig,
    build_player_features, build_club_features, build_transfer_context,
    PLAYER_STAT_COLS, CLUB_STAT_COLS, LEAGUES,
)

MODEL_DIR = Path("models")


@dataclass
class Filters:
    """Same shape as v2's Filters — drop-in compatible."""
    position: str = "ATT"
    max_value_eur_m: float = 50.0
    min_age: int = 18
    max_age: int = 30
    min_minutes: int = 900
    leagues: list[str] | None = None
    contract_ends_before: pd.Timestamp | None = None
    max_contract_months: int | None = None


@dataclass
class DestinationClub:
    """
    Tactical signature of the destination club, plus league.

    These are the four observable team-level metrics the model was trained on.
    In production you compute them once per club-season from FBref team stats.

    Typical ranges (Big-5 leagues):
        ppda           : 6 (Liverpool/Bayern) → 18 (low-block midtable)
        possession_pct : 35 → 70
        directness_idx : 0.30 (Pep) → 0.70 (long-ball)
        line_height_m  : 30 (deep block) → 60 (high line)
    """
    ppda: float
    possession_pct: float
    directness_idx: float
    line_height_m: float
    league: str
    club_id: int = -1  # not used by the model but kept for compatibility


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """Hard-filter the player pool. Identical to v2."""
    out = df.copy()
    out = out[out["position_group"] == f.position]
    out = out[out["market_value_eur_m"].fillna(np.inf) <= f.max_value_eur_m]
    out = out[(out["age"] >= f.min_age) & (out["age"] <= f.max_age)]
    out = out[out["minutes"].fillna(0) >= f.min_minutes]
    if f.leagues:
        out = out[out["league"].isin(f.leagues)]
    if f.contract_ends_before is not None and "contract_expiration_date" in out.columns:
        out = out[out["contract_expiration_date"] <= f.contract_ends_before]
    if f.max_contract_months is not None and "contract_months_remaining" in out.columns:
        out = out[out["contract_months_remaining"].fillna(999) <= f.max_contract_months]
    return out.reset_index(drop=True)


def _load_model() -> tuple[TwoTowerFitModel, FeatureConfig]:
    cfg = pickle.loads((MODEL_DIR / "feature_config.pkl").read_bytes())
    assert cfg.player_scaler is not None, "feature_config.pkl is missing player_scaler"
    assert cfg.club_scaler is not None, "feature_config.pkl is missing club_scaler"
    assert cfg.ctx_scaler is not None, "feature_config.pkl is missing ctx_scaler"
    # Reconstruct model — dims inferred from saved scalers
    n_player = cfg.player_scaler.n_features_in_
    n_club = cfg.club_scaler.n_features_in_
    n_ctx = cfg.ctx_scaler.n_features_in_
    model = TwoTowerFitModel(n_player, n_club, n_ctx)
    model.load_state_dict(torch.load(MODEL_DIR / "two_tower.pt", weights_only=True))
    model.eval()
    return model, cfg


def _destination_to_df(dest: DestinationClub) -> pd.DataFrame:
    return pd.DataFrame([{
        "club_id": dest.club_id,
        "league": dest.league,
        "ppda": dest.ppda,
        "possession_pct": dest.possession_pct,
        "directness_idx": dest.directness_idx,
        "line_height_m": dest.line_height_m,
    }])


def compute_fit_score(
    candidates: pd.DataFrame,
    destination: DestinationClub,
    transfer_age: int | None = None,
    fee_eur_m: float | None = None,
) -> pd.DataFrame:
    """
    Predict transfer-success probability for each candidate at the destination.

    For each candidate, we evaluate the two-tower model with:
      - player tower input: their pre-transfer profile
      - club tower input:   the destination's tactical signature
      - ctx input:          transfer-level fee + age

    Returns candidates ranked by predicted success, with a 0-100 fit_score
    column and the raw model probability in fit_raw.
    """
    if len(candidates) == 0:
        return candidates.assign(fit_score=[], fit_raw=[])

    model, cfg = _load_model()

    p_X, _ = build_player_features(candidates, cfg)
    c_X_one, _ = build_club_features(_destination_to_df(destination), cfg)
    c_X = np.tile(c_X_one, (len(candidates), 1))

    # Context uses per-candidate values: the model learned that transfer age and
    # fee are predictive of transfer outcomes, and these signals also serve as
    # useful quality proxies. The resulting age/value gradient in scores is mild
    # and acceptable given the model's overall R²≈0.08.
    ages = candidates["age"].values if transfer_age is None else np.full(len(candidates), transfer_age)
    fees = candidates["market_value_eur_m"].values if fee_eur_m is None else np.full(len(candidates), fee_eur_m)
    ctx_df = pd.DataFrame({"transfer_age": ages, "fee_eur_m": fees})
    ctx_X, _ = build_transfer_context(ctx_df, cfg)

    with torch.no_grad():
        raw = model(
            torch.from_numpy(p_X), torch.from_numpy(c_X), torch.from_numpy(ctx_X)
        ).numpy()

    # Map sigmoid output (0–1) to 0–100 using the model's own probability scale
    normalised = (raw * 100).clip(0, 100)

    out = candidates.copy()
    out["fit_raw"] = raw.round(4)
    out["fit_score"] = normalised.round(1)

    # Percentile ranks for model features + extra display columns used in the UI
    _DISPLAY_EXTRA = [
        "prog_carries_p90", "pass_completion_pct", "prog_passes_p90",
        "take_ons_p90", "aerials_won_p90", "contract_months_remaining",
    ]
    for col in PLAYER_STAT_COLS + _DISPLAY_EXTRA:
        if col in out.columns:
            out[f"{col}_pct_rank"] = (
                out[col].rank(pct=True).fillna(0.5) * 100
            ).round(0)

    return out.sort_values("fit_score", ascending=False).reset_index(drop=True)


def get_player_embeddings(players_df: pd.DataFrame) -> tuple[np.ndarray, FeatureConfig]:
    """
    Side-benefit of two-tower training: a learned player embedding space.
    Useful for "find similar players" without retraining.
    """
    model, cfg = _load_model()
    p_X, _ = build_player_features(players_df, cfg)
    with torch.no_grad():
        emb = model.player_embedding(torch.from_numpy(p_X)).numpy()
    return emb, cfg


# --- demo --------------------------------------------------------------------
def demo():
    """Klopp-style high-press destination, find best ATT candidates."""
    from real_data import generate

    print("=" * 60)
    print("DEMO: shortlist for a Klopp-style high-press club")
    print("=" * 60)

    data = generate()
    pool = data["players"]

    filters = Filters(position="ATT", max_value_eur_m=50, min_age=20, max_age=27, min_minutes=1200)
    candidates = apply_filters(pool, filters)
    print(f"\nCandidates after hard filters: {len(candidates)}")

    # Klopp profile: aggressive press (low PPDA), moderate possession, direct,
    # high defensive line.
    destination = DestinationClub(
        ppda=7.5,
        possession_pct=55,
        directness_idx=0.55,
        line_height_m=58,
        league="Premier League",
    )
    ranked = compute_fit_score(candidates, destination, fee_eur_m=35)
    cols = ["player_name", "age", "league", "market_value_eur_m",
            "goals_p90", "npxg_p90", "avg_min_per_game", "prog_carries_p90",
            "fit_score", "fit_raw"]
    print("\nTop 10:")
    print(ranked[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    demo()
