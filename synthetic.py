"""
Synthetic transfer dataset with a structured underlying signal.

Design philosophy: the right way to validate a learned fit-prediction model is
to generate data where we *know* the true relationship, then check whether each
candidate model recovers it. This isolates whether the architecture is capable
in principle from whether real-world data is good enough — two failure modes
that get conflated otherwise.

Latent structure
----------------
- Each PLAYER has 4 latent traits ~ N(0,1):
    technical, physical, pressing_inclination, creativity
- Each CLUB has 4 latent style dimensions ~ N(0,1):
    press_intensity, possession_orientation, directness, defensive_line_height
- Observed features (FBref-style per-90s for players, team-stat aggregates
  for clubs) are noisy linear projections of these latents — i.e. exactly the
  situation real-world stats live in.

True transfer success depends on COMPLEMENTARITY:
    a pressing player at a pressing club fits;
    a technical player at a possession-based club fits;
    a physical player at a direct club fits;
plus age/fee plausibility terms.

Cosine-on-text baselines should perform near-chance on this because they have
no mechanism to capture interaction terms. Two-tower models should excel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# --- dimensions ---------------------------------------------------------------
N_PLAYERS = 3000
N_CLUBS = 50
N_TRANSFERS = 8000
POSITIONS = ["GK", "DEF", "MID", "ATT"]
LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
LEAGUE_STRENGTH = {  # rough UEFA coefficient ratios, hand-set for realism
    "Premier League": 1.00,
    "La Liga": 0.92,
    "Bundesliga": 0.88,
    "Serie A": 0.86,
    "Ligue 1": 0.78,
}


def _generate_players(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (observed dataframe, latent traits array of shape [n, 4])."""
    latents = rng.standard_normal((n, 4))  # tech, phys, press, creat
    tech, phys, press, creat = latents.T

    # Observed per-90 stats as noisy projections of latents.
    # Coefficients chosen so that no single observed stat fully reveals a latent.
    noise = lambda scale=0.3: rng.normal(0, scale, n)
    goals_p90 = np.clip(0.30 + 0.18 * tech + 0.10 * phys + noise(0.08), 0, None)
    assists_p90 = np.clip(0.20 + 0.15 * creat + 0.08 * tech + noise(0.06), 0, None)
    npxg_p90 = np.clip(0.28 + 0.16 * tech + 0.08 * phys + noise(0.07), 0, None)
    prog_passes_p90 = np.clip(4.0 + 1.5 * tech + 0.8 * creat + noise(0.6), 0, None)
    key_passes_p90 = np.clip(1.5 + 0.7 * creat + 0.3 * tech + noise(0.3), 0, None)
    tackles_p90 = np.clip(1.8 + 0.9 * press + 0.4 * phys + noise(0.4), 0, None)
    interceptions_p90 = np.clip(1.0 + 0.6 * press + 0.2 * phys + noise(0.3), 0, None)
    take_ons_p90 = np.clip(2.0 + 1.0 * tech + 0.7 * creat + noise(0.5), 0, None)
    prog_carries_p90 = np.clip(3.0 + 1.2 * tech + 0.6 * creat + noise(0.5), 0, None)
    aerials_won_p90 = np.clip(1.5 + 1.2 * phys + noise(0.4), 0, None)
    pass_completion_pct = np.clip(75 + 6 * tech + 2 * creat + noise(2.0), 40, 99)

    positions = rng.choice(POSITIONS, size=n, p=[0.08, 0.32, 0.32, 0.28])
    ages = rng.integers(17, 36, size=n)
    minutes = rng.integers(500, 3200, size=n)
    leagues = rng.choice(LEAGUES, size=n)
    # market value loosely correlated with technical + age curve + league
    age_curve = -((ages - 26) ** 2) / 80  # peaks around 26
    raw_val = 5 + 8 * tech + 4 * creat + 3 * phys + 6 * age_curve + \
              np.array([LEAGUE_STRENGTH[l] for l in leagues]) * 5 + noise(2.0)
    market_value_eur_m = np.clip(raw_val, 0.2, None)

    df = pd.DataFrame({
        "player_id": np.arange(n),
        "position_group": positions,
        "age": ages,
        "minutes": minutes,
        "league": leagues,
        "market_value_eur_m": market_value_eur_m.round(1),
        "goals_p90": goals_p90.round(2),
        "assists_p90": assists_p90.round(2),
        "npxg_p90": npxg_p90.round(2),
        "prog_passes_p90": prog_passes_p90.round(2),
        "key_passes_p90": key_passes_p90.round(2),
        "tackles_p90": tackles_p90.round(2),
        "interceptions_p90": interceptions_p90.round(2),
        "take_ons_p90": take_ons_p90.round(2),
        "prog_carries_p90": prog_carries_p90.round(2),
        "aerials_won_p90": aerials_won_p90.round(2),
        "pass_completion_pct": pass_completion_pct.round(1),
    })
    return df, latents


def _generate_clubs(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (observed dataframe, latent style array of shape [n, 4])."""
    latents = rng.standard_normal((n, 4))  # press, poss, direct, line
    press, poss, direct, line = latents.T

    noise = lambda scale=0.3: rng.normal(0, scale, n)
    # PPDA: lower = more pressing. Real-world range ~6-18.
    ppda = np.clip(12 - 3.0 * press + noise(0.8), 4, 22)
    possession_pct = np.clip(50 + 8 * poss + noise(2.0), 30, 72)
    directness_idx = np.clip(0.5 + 0.18 * direct + noise(0.05), 0, 1)
    line_height_m = np.clip(45 + 8 * line + noise(2.0), 25, 65)  # avg defensive line height

    leagues = rng.choice(LEAGUES, size=n)
    df = pd.DataFrame({
        "club_id": np.arange(n),
        "league": leagues,
        "ppda": ppda.round(2),
        "possession_pct": possession_pct.round(1),
        "directness_idx": directness_idx.round(3),
        "line_height_m": line_height_m.round(1),
    })
    return df, latents


def _compatibility(player_latent: np.ndarray, club_latent: np.ndarray) -> np.ndarray:
    """
    True underlying fit function. The whole point: it's INTERACTIVE, not additive.

    Rewards complementarity between player traits and club style.
    A linear model on flat features cannot recover this; an interaction-capable
    model (GBM, neural net with hidden layers) should.
    """
    tech, phys, press, creat = player_latent.T
    p_press, p_poss, p_direct, p_line = club_latent.T

    return (
        1.2 * press * p_press        # pressing player @ pressing club
      + 1.0 * tech * p_poss          # technical player @ possession club
      + 0.8 * phys * p_direct        # physical player @ direct club
      + 0.6 * creat * p_poss         # creative player @ possession club
      + 0.4 * phys * p_line          # physical @ high-line (handles space behind)
    )


def _generate_transfers(
    players_df: pd.DataFrame,
    player_latents: np.ndarray,
    clubs_df: pd.DataFrame,
    club_latents: np.ndarray,
    n_transfers: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate transfers + observed outcomes."""
    # Sample transfers, biased so younger players move more often
    age_weights = np.exp(-((players_df["age"].values - 23) ** 2) / 30)
    age_weights /= age_weights.sum()
    player_idx = rng.choice(len(players_df), size=n_transfers, p=age_weights)
    club_idx = rng.choice(len(clubs_df), size=n_transfers)

    p_lat = player_latents[player_idx]
    c_lat = club_latents[club_idx]
    compat = _compatibility(p_lat, c_lat)

    # Age and fee plausibility terms
    ages = players_df["age"].values[player_idx]
    age_pen = -np.abs(ages - 25) * 0.08

    values = players_df["market_value_eur_m"].values[player_idx]
    dest_leagues = clubs_df["league"].values[club_idx]
    dest_strength = np.array([LEAGUE_STRENGTH[l] for l in dest_leagues])
    # "Step-up" plausibility: a 2m player at a top-league club is risky;
    # a 30m player at a weaker league is overqualified (often unhappy).
    step_factor = 1 - np.abs(np.log(values + 1) - np.log(50 * dest_strength + 1)) * 0.2

    raw = compat + age_pen + step_factor + rng.normal(0, 0.5, n_transfers)
    success_logit = raw
    # Map to 0-1 latent success
    success = 1 / (1 + np.exp(-success_logit / 1.5))

    # Observed outcomes (noisy reflections of success):
    minutes_share_y1 = np.clip(success + rng.normal(0, 0.12, n_transfers), 0, 1)
    value_delta_18m = np.clip((success - 0.5) * 1.2 + rng.normal(0, 0.25, n_transfers), -0.9, 3.0)
    survival_2y = (success + rng.normal(0, 0.15, n_transfers) > 0.45).astype(int)

    # Composite label (what we'll predict)
    success_score = (
        0.5 * minutes_share_y1
      + 0.3 * (1 / (1 + np.exp(-value_delta_18m * 2)))  # sigmoid of value delta
      + 0.2 * survival_2y
    )

    df = pd.DataFrame({
        "transfer_id": np.arange(n_transfers),
        "player_id": players_df["player_id"].values[player_idx],
        "destination_club_id": clubs_df["club_id"].values[club_idx],
        "transfer_age": ages,
        "fee_eur_m": (values * rng.uniform(0.7, 1.4, n_transfers)).round(1),
        "destination_league": dest_leagues,
        # outcomes
        "minutes_share_y1": minutes_share_y1.round(3),
        "value_delta_18m": value_delta_18m.round(3),
        "survival_2y": survival_2y,
        "success_score": success_score.round(3),
    })
    return df


def generate(seed: int = 42, save_dir: str | None = None) -> dict:
    """Generate the full synthetic dataset. Returns dict with all artifacts."""
    rng = np.random.default_rng(seed)
    players_df, player_latents = _generate_players(N_PLAYERS, rng)
    clubs_df, club_latents = _generate_clubs(N_CLUBS, rng)
    transfers_df = _generate_transfers(
        players_df, player_latents, clubs_df, club_latents, N_TRANSFERS, rng
    )

    out = {
        "players": players_df,
        "clubs": clubs_df,
        "transfers": transfers_df,
        "player_latents": player_latents,
        "club_latents": club_latents,
    }
    if save_dir:
        from pathlib import Path
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        players_df.to_parquet(p / "players.parquet")
        clubs_df.to_parquet(p / "clubs.parquet")
        transfers_df.to_parquet(p / "transfers.parquet")
        np.save(p / "player_latents.npy", player_latents)
        np.save(p / "club_latents.npy", club_latents)
    return out


if __name__ == "__main__":
    data = generate(save_dir="data")
    print(f"Generated:")
    print(f"  players:   {len(data['players']):,} rows, cols={list(data['players'].columns)}")
    print(f"  clubs:     {len(data['clubs']):,} rows, cols={list(data['clubs'].columns)}")
    print(f"  transfers: {len(data['transfers']):,} rows")
    print(f"\nSuccess score distribution:")
    print(data['transfers']['success_score'].describe().round(3))
