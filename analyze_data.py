"""
Data analysis for AI Football Scout v3.

Produces a set of CSV files under analysis/ giving a full overview
of every dataset used by the model.

Run: python analyze_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("analysis")
OUT.mkdir(exist_ok=True)

DATA = Path("data")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading datasets...")
players   = pd.read_parquet(DATA / "real_players.parquet")
clubs     = pd.read_parquet(DATA / "real_clubs.parquet")
transfers = pd.read_parquet(DATA / "real_transfers.parquet")

REAL_COLS       = ["goals_p90", "assists_p90", "yellow_cards_p90",
                   "avg_min_per_game", "n_apps"]
PROXIED_COLS    = ["npxg_p90", "key_passes_p90"]
HEURISTIC_COLS  = ["prog_passes_p90", "pass_completion_pct",
                   "take_ons_p90", "prog_carries_p90", "aerials_won_p90"]
VALUATION_COLS  = ["mv_momentum_12m", "market_value_eur_m"]
CONTRACT_COLS   = ["contract_months_remaining", "injury_days_last_2y", "has_serious_injury"]

FEATURE_SOURCE = (
    [(c, "Real (TM appearances)")  for c in REAL_COLS]
    + [(c, "Proxied (×multiplier)") for c in PROXIED_COLS]
    + [(c, "Heuristic (pos×value)") for c in HEURISTIC_COLS]
    + [(c, "Real (TM valuations)")  for c in VALUATION_COLS]
    + [(c, "Real (contract/injury)") for c in CONTRACT_COLS]
)

# ---------------------------------------------------------------------------
# 1. High-level dataset overview
# ---------------------------------------------------------------------------
unique_transfer_players = transfers["player_id"].nunique()
overlap = len(set(transfers["player_id"]) & set(players["player_id"]))
transfers_with_apps = (transfers["n_apps"] > 0).sum()
survival_known = transfers["survival_2y"].notna().sum()

overview_rows = [
    # Players
    ("players",   "Total rows",                    len(players)),
    ("players",   "Unique players",                players["player_id"].nunique()),
    ("players",   "Season",                        "2024-25"),
    ("players",   "Missing values (any column)",   int(players.isnull().sum().sum())),
    ("players",   "Leagues covered",               players["league"].nunique()),
    ("players",   "Position breakdown — GK",       int((players["position_group"] == "GK").sum())),
    ("players",   "Position breakdown — DEF",      int((players["position_group"] == "DEF").sum())),
    ("players",   "Position breakdown — MID",      int((players["position_group"] == "MID").sum())),
    ("players",   "Position breakdown — ATT",      int((players["position_group"] == "ATT").sum())),
    ("players",   "Age min",                       float(players["age"].min())),
    ("players",   "Age mean",                      round(float(players["age"].mean()), 1)),
    ("players",   "Age max",                       float(players["age"].max())),
    ("players",   "Market value median (M€)",      round(float(players["market_value_eur_m"].median()), 2)),
    # Clubs
    ("clubs",     "Total rows",                    len(clubs)),
    ("clubs",     "Leagues covered",               clubs["league"].nunique()),
    ("clubs",     "Possession pct mean",           round(float(clubs["possession_pct"].mean()), 1)),
    ("clubs",     "PPDA mean",                     round(float(clubs["ppda"].mean()), 1)),
    ("clubs",     "Directness index mean",         round(float(clubs["directness_idx"].mean()), 3)),
    ("clubs",     "Line height mean (m)",          round(float(clubs["line_height_m"].mean()), 1)),
    # Transfers
    ("transfers", "Total rows",                    len(transfers)),
    ("transfers", "Unique players",                unique_transfer_players),
    ("transfers", "Overlap with scouting pool",    overlap),
    ("transfers", "Date range start",              str(transfers["transfer_date"].min().date())),
    ("transfers", "Date range end",                str(transfers["transfer_date"].max().date())),
    ("transfers", "Rows with n_apps > 0 (usable)", int(transfers_with_apps)),
    ("transfers", "Rows with n_apps = 0 (no stats)",int(len(transfers) - transfers_with_apps)),
    ("transfers", "survival_2y known",             int(survival_known)),
    ("transfers", "survival_2y missing (too recent)", int(transfers["survival_2y"].isna().sum())),
    ("transfers", "success_score mean",            round(float(transfers["success_score"].mean()), 3)),
    ("transfers", "success_score std",             round(float(transfers["success_score"].std()), 3)),
    ("transfers", "minutes_share_y1 mean",         round(float(transfers["minutes_share_y1"].mean()), 3)),
    ("transfers", "value_delta_18m mean",          round(float(transfers["value_delta_18m"].mean()), 3)),
    ("transfers", "Avg transfers per player",      round(float(len(transfers) / unique_transfer_players), 2)),
    ("transfers", "Players with 1 transfer",       int((transfers.groupby("player_id").size() == 1).sum())),
    ("transfers", "Players with 2 transfers",      int((transfers.groupby("player_id").size() == 2).sum())),
    ("transfers", "Players with 3+ transfers",     int((transfers.groupby("player_id").size() >= 3).sum())),
]

overview_df = pd.DataFrame(overview_rows, columns=["dataset", "metric", "value"])
overview_df.to_csv(OUT / "1_overview.csv", index=False)
print(f"  -> {OUT}/1_overview.csv  ({len(overview_df)} rows)")

# ---------------------------------------------------------------------------
# 2. Feature quality table
# ---------------------------------------------------------------------------
stat_rows = []
for col, source in FEATURE_SOURCE:
    for dataset_name, df in [("players", players), ("transfers", transfers)]:
        if col not in df.columns:
            continue
        s = df[col]
        stat_rows.append({
            "column":       col,
            "source":       source,
            "dataset":      dataset_name,
            "non_null":     int(s.notna().sum()),
            "zero_pct":     round(float((s == 0).sum() / len(s) * 100), 1),
            "min":          round(float(s.min()), 3),
            "p25":          round(float(s.quantile(0.25)), 3),
            "median":       round(float(s.median()), 3),
            "p75":          round(float(s.quantile(0.75)), 3),
            "max":          round(float(s.max()), 3),
            "mean":         round(float(s.mean()), 3),
            "std":          round(float(s.std()), 3),
        })

feature_df = pd.DataFrame(stat_rows)
feature_df.to_csv(OUT / "2_feature_quality.csv", index=False)
print(f"  -> {OUT}/2_feature_quality.csv  ({len(feature_df)} rows)")

# ---------------------------------------------------------------------------
# 3. Players — full table (sorted by market value)
# ---------------------------------------------------------------------------
p_export = players[[
    "player_id", "player_name", "position_group", "age", "league",
    "market_value_eur_m", "goals_p90", "assists_p90", "npxg_p90",
    "avg_min_per_game", "n_apps", "pass_completion_pct",
    "mv_momentum_12m", "contract_months_remaining",
    "injury_days_last_2y", "has_serious_injury",
]].sort_values("market_value_eur_m", ascending=False).reset_index(drop=True)
p_export.index += 1
p_export.to_csv(OUT / "3_players.csv", index_label="rank")
print(f"  -> {OUT}/3_players.csv  ({len(p_export)} rows)")

# ---------------------------------------------------------------------------
# 4. Players — per-league summary
# ---------------------------------------------------------------------------
league_summary = players.groupby("league").agg(
    n_players=("player_id", "count"),
    age_mean=("age", "mean"),
    mv_median_m=("market_value_eur_m", "median"),
    goals_p90_mean=("goals_p90", "mean"),
    assists_p90_mean=("assists_p90", "mean"),
    avg_min_per_game_mean=("avg_min_per_game", "mean"),
    n_apps_mean=("n_apps", "mean"),
    injury_days_mean=("injury_days_last_2y", "mean"),
    has_serious_injury_pct=("has_serious_injury", "mean"),
).round(2).reset_index()
league_summary.to_csv(OUT / "4_players_by_league.csv", index=False)
print(f"  -> {OUT}/4_players_by_league.csv")

# ---------------------------------------------------------------------------
# 5. Players — per-position summary
# ---------------------------------------------------------------------------
pos_summary = players.groupby("position_group").agg(
    n_players=("player_id", "count"),
    age_mean=("age", "mean"),
    mv_median_m=("market_value_eur_m", "median"),
    goals_p90_mean=("goals_p90", "mean"),
    assists_p90_mean=("assists_p90", "mean"),
    prog_passes_p90_mean=("prog_passes_p90", "mean"),
    take_ons_p90_mean=("take_ons_p90", "mean"),
    aerials_won_p90_mean=("aerials_won_p90", "mean"),
    contract_months_mean=("contract_months_remaining", "mean"),
).round(2).reset_index()
pos_summary.to_csv(OUT / "5_players_by_position.csv", index=False)
print(f"  -> {OUT}/5_players_by_position.csv")

# ---------------------------------------------------------------------------
# 6. Clubs — full table
# ---------------------------------------------------------------------------
clubs_export = clubs.sort_values("league").reset_index(drop=True)
clubs_export.to_csv(OUT / "6_clubs.csv", index=False)
print(f"  -> {OUT}/6_clubs.csv  ({len(clubs_export)} rows)")

# ---------------------------------------------------------------------------
# 7. Transfers — full table
# ---------------------------------------------------------------------------
t_export = transfers[[
    "transfer_id", "player_id", "destination_club_id", "transfer_age",
    "fee_eur_m", "transfer_date", "position_group", "league",
    "goals_p90", "assists_p90", "avg_min_per_game", "n_apps",
    "market_value_eur_m", "mv_momentum_12m",
    "contract_months_remaining", "injury_days_last_2y", "has_serious_injury",
    "minutes_share_y1", "value_delta_18m", "survival_2y", "success_score",
]].sort_values("transfer_date").reset_index(drop=True)
t_export.to_csv(OUT / "7_transfers.csv", index=False)
print(f"  -> {OUT}/7_transfers.csv  ({len(t_export)} rows)")

# ---------------------------------------------------------------------------
# 8. Transfers — per-year summary
# ---------------------------------------------------------------------------
transfers["year"] = transfers["transfer_date"].dt.year
year_summary = transfers.groupby("year").agg(
    n_transfers=("transfer_id", "count"),
    unique_players=("player_id", "nunique"),
    success_score_mean=("success_score", "mean"),
    minutes_share_mean=("minutes_share_y1", "mean"),
    value_delta_mean=("value_delta_18m", "mean"),
    survival_known_pct=("survival_2y", lambda x: x.notna().mean() * 100),
    fee_eur_m_mean=("fee_eur_m", "mean"),
    pct_with_stats=("n_apps", lambda x: (x > 0).mean() * 100),
).round(2).reset_index()
year_summary.to_csv(OUT / "8_transfers_by_year.csv", index=False)
print(f"  -> {OUT}/8_transfers_by_year.csv")

# ---------------------------------------------------------------------------
# 9. Transfers — per-player (how many transfers each player made)
# ---------------------------------------------------------------------------
per_player = transfers.groupby("player_id").agg(
    n_transfers=("transfer_id", "count"),
    first_transfer=("transfer_date", "min"),
    last_transfer=("transfer_date", "max"),
    avg_success_score=("success_score", "mean"),
    avg_fee_eur_m=("fee_eur_m", "mean"),
    position_group=("position_group", "first"),
    in_scouting_pool=("player_id", lambda x: x.iloc[0] in set(players["player_id"])),
).reset_index().sort_values("n_transfers", ascending=False)
per_player.to_csv(OUT / "9_transfers_per_player.csv", index=False)
print(f"  -> {OUT}/9_transfers_per_player.csv  ({len(per_player)} rows)")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print()
print("=" * 56)
print("SUMMARY")
print("=" * 56)
print(f"  Players (scouting pool 2024-25):  {len(players):>5,}")
print(f"  Clubs (Big-5 tactical features):  {len(clubs):>5,}")
print(f"  Transfers (training, 2018-2024):  {len(transfers):>5,}")
print(f"    Unique players in transfers:    {unique_transfer_players:>5,}")
print(f"    All in scouting pool:           {overlap == unique_transfer_players}")
print(f"    Usable rows (n_apps > 0):       {int(transfers_with_apps):>5,}  ({transfers_with_apps/len(transfers)*100:.0f}%)")
print(f"    survival_2y known:              {int(survival_known):>5,}  ({survival_known/len(transfers)*100:.0f}%)")
print()
print(f"  Feature columns: {len(FEATURE_SOURCE)} total")
print(f"    Real (TM appearances):   {len(REAL_COLS)}")
print(f"    Proxied (×multiplier):   {len(PROXIED_COLS)}")
print(f"    Heuristic (pos×value):   {len(HEURISTIC_COLS)}")
print(f"    Real (TM valuations):    {len(VALUATION_COLS)}")
print(f"    Real (contract/injury):  {len(CONTRACT_COLS)}")
print()
print(f"  Output written to: {OUT}/")
