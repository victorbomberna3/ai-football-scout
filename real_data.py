"""
Real data ETL for AI Football Scout v3.

Produces the same three dataframes as synthetic.generate():
  players   — FBref per-90 stats + TM metadata (from v2 scouting pool)
  clubs     — tactical features derived from match-level data (football-data.co.uk)
  transfers — TM transfers with outcome labels from appearances + valuations

Data sources
  v2 scouting pool  — ai-football-scout-v2/data/scouting_pool.parquet
                      already scraped: 2700 Big-5 players, 2024-25 season,
                      all 11 per-90 stat columns + TM player_id + market value
  football-data.co.uk — free per-match CSVs, no auth, Big-5 × 2018-2024
                         used for club tactical features (shots, fouls, corners)
  Transfermarkt         — dcaribou/transfermarkt-datasets public R2 bucket
                          transfers, appearances, player_valuations

Limitations vs future "ideal" pipeline
  1. Player stats are from a single snapshot (2024-25), not season-of-transfer.
     The model learns that a pressing player at a pressing club tends to succeed —
     that signal survives even without perfect temporal alignment.
  2. Club tactical features are match-level proxies, not event-data PPDA.
     They're directionally correct and sufficient for the model to learn on.

Run once; all results are cached under data/.
Usage:
    from real_data import generate
    data = generate()
    data = generate(save_dir="data")  # also writes parquet files
"""
from __future__ import annotations

import gzip
import io
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

V2_CACHE = Path("../ai-football-scout-v2/data/cache")

SEASONS = ["2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"]

BIG5_COMP = {"GB1": "Premier League", "ES1": "La Liga", "L1": "Bundesliga",
             "IT1": "Serie A", "FR1": "Ligue 1"}
MAX_SEASON_MIN = {"GB1": 3420, "ES1": 3420, "L1": 3060, "IT1": 3420, "FR1": 3420}

TM_BUCKET = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TM_NEEDED = ["players", "clubs", "competitions", "transfers", "appearances", "player_valuations"]

# football-data.co.uk: season code → (league_key → CSV filename)
FD_BASE = "https://www.football-data.co.uk/mmz4281"
FD_LEAGUES = {"E0": "Premier League", "SP1": "La Liga",
              "D1": "Bundesliga", "I1": "Serie A", "F1": "Ligue 1"}
FD_SEASON_CODES = {
    "2018-2019": "1819", "2019-2020": "1920", "2020-2021": "2021",
    "2021-2022": "2122", "2022-2023": "2223", "2023-2024": "2324",
}
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ai-football-scout/0.3; +research)"}
TM_INJURIES_CACHE = CACHE_DIR / "tm_injuries.parquet"


# ---------------------------------------------------------------------------
# 1. Transfermarkt download
# ---------------------------------------------------------------------------
def fetch_tm() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in TM_NEEDED:
        cache = CACHE_DIR / f"tm_{name}.parquet"
        if cache.exists():
            out[name] = pd.read_parquet(cache)
            print(f"  [cache] tm/{name}: {len(out[name]):,} rows")
            continue
        url = f"{TM_BUCKET}/{name}.csv.gz"
        print(f"  downloading tm/{name}...")
        r = requests.get(url, headers=HTTP_HEADERS, timeout=180)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(gzip.decompress(r.content)), low_memory=False)
        df.to_parquet(cache)
        out[name] = df
        print(f"  [fetched] tm/{name}: {len(df):,} rows")
    return out


# ---------------------------------------------------------------------------
# 2. football-data.co.uk — per-match CSVs → club tactical feature proxies
# ---------------------------------------------------------------------------
def fetch_fd_season(season_code: str, league_key: str) -> pd.DataFrame | None:
    """Download one season × league CSV; return raw DataFrame or None on error."""
    url = f"{FD_BASE}/{season_code}/{league_key}.csv"
    cache = CACHE_DIR / f"fd_{league_key}_{season_code}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    try:
        r = requests.get(url, headers=HTTP_HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.to_parquet(cache)
        return df
    except Exception as e:
        print(f"    warning: {league_key} {season_code} — {e}")
        return None


def build_clubs_df(tm: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Derive four tactical features per club from football-data.co.uk match stats.

    Features and their proxies:
      possession_pct  — shots_for / (shots_for + shots_against) × 100
                        Shot share tracks possession (r ≈ 0.85 with Opta possession).
      ppda            — 20 − fouls_committed_per_game × 0.45 (clipped 4–20)
                        High-press teams foul more in the opponent's half.
      directness_idx  — shots_on_target / shots (higher = more clinical/direct)
                        Possession sides take more speculative shots → lower ratio.
      line_height_m   — 65 − clearances_proxy × scale
                        Clearances not in football-data; use corners_against / game as
                        proxy (more corners against = opponents win the ball wide =
                        defence is deeper). Mapped to [25, 65] metre range.
    """
    tm_clubs = tm["clubs"].copy()
    big5 = tm_clubs[tm_clubs["domestic_competition_id"].isin(BIG5_COMP)].copy()

    # Aggregate match stats per team across all seasons
    records: list[dict] = []
    for season, code in FD_SEASON_CODES.items():
        for lg_key, lg_name in FD_LEAGUES.items():
            df = fetch_fd_season(code, lg_key)
            if df is None or df.empty:
                continue
            # Process both home and away perspective
            for side, opp, prefix in [("Home", "Away", "H"), ("Away", "Home", "A")]:
                team_col, opp_col = f"{side}Team", f"{opp}Team"
                if team_col not in df.columns:
                    continue
                agg = df.groupby(team_col).agg(
                    games=(team_col, "count"),
                    shots_for=(f"{prefix}S", "sum") if f"{prefix}S" in df.columns else (team_col, "count"),
                    shots_against=(f"{'A' if prefix == 'H' else 'H'}S", "sum") if "HS" in df.columns else (team_col, "count"),
                    sot_for=(f"{prefix}ST", "sum") if f"{prefix}ST" in df.columns else (team_col, "count"),
                    fouls=(f"{prefix}F", "sum") if f"{prefix}F" in df.columns else (team_col, "count"),
                    corners_against=(f"{'A' if prefix == 'H' else 'H'}C", "sum") if "HC" in df.columns else (team_col, "count"),
                    goals_for=(f"FT{'HG' if prefix == 'H' else 'AG'}", "sum") if "FTHG" in df.columns else (team_col, "count"),
                ).reset_index().rename(columns={team_col: "team_name"})
                agg["league"] = lg_name
                agg["season"] = season
                records.append(agg)

    if not records:
        print("  warning: no football-data.co.uk data fetched; using league-average defaults")
        return _default_clubs_df(big5)

    raw = pd.concat(records, ignore_index=True)

    # Aggregate per team (across home+away and all seasons → one row per club)
    team_stats = raw.groupby("team_name").agg(
        games=("games", "sum"),
        shots_for=("shots_for", "sum"),
        shots_against=("shots_against", "sum"),
        sot_for=("sot_for", "sum"),
        fouls=("fouls", "sum"),
        corners_against=("corners_against", "sum"),
        goals_for=("goals_for", "sum"),
        league=("league", "last"),
    ).reset_index()

    team_stats["possession_pct"] = (
        team_stats["shots_for"] / (team_stats["shots_for"] + team_stats["shots_against"] + 1e-6) * 100
    ).clip(30, 70)

    team_stats["ppda"] = (
        20 - (team_stats["fouls"] / team_stats["games"]) * 0.45
    ).clip(4, 20)

    team_stats["directness_idx"] = (
        team_stats["sot_for"] / (team_stats["shots_for"] + 1e-6)
    ).clip(0.20, 0.65)

    # corners_against per game: high → deep defence (lower line height)
    ca_per_game = team_stats["corners_against"] / team_stats["games"]
    team_stats["line_height_m"] = (
        65 - (ca_per_game - 3.5) / 2.5 * 15
    ).clip(25, 65)

    # Match team names to TM club IDs (fuzzy)
    from rapidfuzz import fuzz, process as rf_process

    big5["name_norm"] = big5["name"].apply(_norm)
    club_names = big5["name_norm"].tolist()
    club_ids = big5["club_id"].tolist()
    comp_ids = big5["domestic_competition_id"].tolist()

    team_stats["name_norm"] = team_stats["team_name"].apply(_norm)
    matched_ids, matched_comps = [], []
    for _, row in team_stats.iterrows():
        best = rf_process.extractOne(row["name_norm"], club_names, scorer=fuzz.WRatio)
        if best and best[1] >= 72:
            i = best[2]
            matched_ids.append(int(club_ids[i]))
            matched_comps.append(comp_ids[i])
        else:
            matched_ids.append(-1)
            matched_comps.append(None)

    team_stats["club_id"] = matched_ids
    team_stats["tm_comp_id"] = matched_comps
    team_stats = team_stats[team_stats["club_id"] >= 0].copy()
    team_stats["league"] = team_stats["tm_comp_id"].map(BIG5_COMP)
    team_stats = team_stats.drop_duplicates("club_id")

    out = team_stats[["club_id", "league", "possession_pct",
                       "ppda", "directness_idx", "line_height_m"]].copy()
    out = out.reset_index(drop=True)
    print(f"  {len(out):,} clubs with tactical features")
    return out


def _default_clubs_df(big5: pd.DataFrame) -> pd.DataFrame:
    """Fallback: populate every Big-5 club with league-average values."""
    LEAGUE_DEFAULTS = {
        "Premier League": (51.2, 11.2, 0.38, 45.0),
        "La Liga":        (53.0, 11.8, 0.37, 44.5),
        "Bundesliga":     (52.5, 10.8, 0.39, 46.0),
        "Serie A":        (50.0, 12.1, 0.36, 43.0),
        "Ligue 1":        (51.5, 11.5, 0.38, 44.0),
    }
    rows = []
    for _, row in big5.iterrows():
        lg = BIG5_COMP.get(row["domestic_competition_id"], "Premier League")
        poss, ppda, direct, line = LEAGUE_DEFAULTS.get(lg, (51, 11.5, 0.38, 44))
        rows.append({"club_id": int(row["club_id"]), "league": lg,
                     "possession_pct": poss, "ppda": ppda,
                     "directness_idx": direct, "line_height_m": line})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Build players_df from v2 scouting pool
# ---------------------------------------------------------------------------
STAT_COLS = [
    # Real from TM appearances (consistent train/inference)
    "goals_p90", "assists_p90",
    "yellow_cards_p90",       # aggression/pressing proxy
    "avg_min_per_game",       # starter vs sub (coach trust)
    "apps_pct_season",        # starter rate = n_apps/38, capped 0-1
    # Proxied from real data (consistent train/inference)
    "npxg_p90", "key_passes_p90",
    # Real FBref misc stats (joined by player_name × league × season)
    "tkl_won_p90",            # tackles won per 90 — pressing intensity
    "interceptions_p90",      # interceptions per 90 — pressing positioning
    # Career trajectory (from valuations)
    "mv_momentum_12m",
    # Position × market-value heuristics (consistently heuristic in both train/inference)
    "prog_passes_p90", "pass_completion_pct", "take_ons_p90",
    "prog_carries_p90", "aerials_won_p90",
    # Risk / availability features
    "contract_months_remaining",  # negotiating leverage + player motivation
    "injury_days_last_2y",        # availability risk
    "has_serious_injury",         # flag: any single injury >60 days in last 3 years
]


# ---------------------------------------------------------------------------
# 3b. Injury history — TM scraping with persistent cache
# ---------------------------------------------------------------------------

def fetch_player_injuries_bulk(player_ids: list) -> pd.DataFrame:
    """
    Scrape TM injury pages for a list of player_ids.
    Returns one row per injury: (player_id, injury_date, days_missed).
    Results are cached permanently in TM_INJURIES_CACHE.
    """
    import time, re

    already: set[int] = set()
    existing_rows: list[pd.DataFrame] = []

    if TM_INJURIES_CACHE.exists():
        cached = pd.read_parquet(TM_INJURIES_CACHE)
        already = set(cached["player_id"].unique())
        existing_rows.append(cached)

    to_fetch = [int(pid) for pid in player_ids if int(pid) not in already]
    if not to_fetch:
        return existing_rows[0] if existing_rows else pd.DataFrame(
            columns=["player_id", "injury_date", "days_missed"]
        )

    print(f"  [injuries] Fetching {len(to_fetch):,} new players "
          f"({len(already):,} already cached)…")

    new_rows: list[dict] = []
    for i, pid in enumerate(to_fetch):
        try:
            url = f"https://www.transfermarkt.com/player/verletzungen/spieler/{pid}"
            r = requests.get(url, headers=HTTP_HEADERS, timeout=20)
            if r.status_code == 200:
                matches = re.findall(
                    r'(\d{2}/\d{2}/\d{4})</td>.*?(\d+)\s*days', r.text, re.DOTALL
                )
                for date_str, days_str in matches:
                    try:
                        new_rows.append({
                            "player_id":    pid,
                            "injury_date":  pd.to_datetime(date_str, format="%m/%d/%Y"),
                            "days_missed":  int(days_str),
                        })
                    except ValueError:
                        pass
        except Exception:
            pass

        time.sleep(0.35)
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(to_fetch)} fetched…")

    all_frames = existing_rows
    if new_rows:
        all_frames.append(pd.DataFrame(new_rows))

    out = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(
        columns=["player_id", "injury_date", "days_missed"]
    )
    # Mark every fetched player as done (even those with no injuries) so we don't re-fetch
    fetched_df = pd.DataFrame({"player_id": to_fetch, "injury_date": pd.NaT, "days_missed": 0})
    out = pd.concat([out, fetched_df], ignore_index=True).drop_duplicates(
        subset=["player_id", "injury_date", "days_missed"]
    )
    out.to_parquet(TM_INJURIES_CACHE, index=False)
    return out


def compute_injury_features(
    injuries_df: pd.DataFrame,
    player_ids,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Compute per-player injury risk features relative to reference_date.
    For the scouting pool reference_date=today; for training use the transfer date.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()

    cutoff_2y = reference_date - pd.Timedelta(days=730)
    cutoff_3y = reference_date - pd.Timedelta(days=1095)

    rows = []
    for pid in player_ids:
        pid = int(pid)
        p = injuries_df[injuries_df["player_id"] == pid].dropna(subset=["injury_date"])
        recent  = p[(p["injury_date"] >= cutoff_2y) & (p["injury_date"] <= reference_date)]
        p3y     = p[(p["injury_date"] >= cutoff_3y) & (p["injury_date"] <= reference_date)]
        rows.append({
            "player_id":          pid,
            "injury_days_last_2y": float(recent["days_missed"].sum()),
            "has_serious_injury":  float((p3y["days_missed"] > 60).any()),
        })
    return pd.DataFrame(rows)


def build_players_df(tm: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the player-stats table directly from TM appearances (2024-25 season).

    All stats are derived from TM data — no FBref dependency.
      Real (TM appearances):   goals_p90, assists_p90, yellow_cards_p90,
                                avg_min_per_game, n_apps
      Proxied from real stats: npxg_p90, key_passes_p90
      Career trajectory:       mv_momentum_12m (TM valuations)
      Position × value heuristics: prog_passes_p90, pass_completion_pct,
                                    take_ons_p90, prog_carries_p90, aerials_won_p90
    """
    big5_comp_ids = list(BIG5_COMP.keys())

    tm_players = tm["players"].copy()
    tm_clubs   = tm["clubs"].copy()

    # Big-5 clubs and their league names
    big5_clubs = tm_clubs[tm_clubs["domestic_competition_id"].isin(big5_comp_ids)][
        ["club_id", "domestic_competition_id"]
    ].copy()
    big5_club_ids = set(big5_clubs["club_id"].unique())
    club_to_league = {
        row["club_id"]: BIG5_COMP[row["domestic_competition_id"]]
        for _, row in big5_clubs.iterrows()
    }

    # --- Aggregate 2024-25 Big-5 league appearances ---
    apps = tm["appearances"].copy()
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    for col in ["minutes_played", "goals", "assists", "yellow_cards"]:
        apps[col] = pd.to_numeric(apps[col], errors="coerce").fillna(0)

    season = apps[
        (apps["date"] >= "2024-07-01") &
        (apps["competition_id"].isin(big5_comp_ids))
    ].copy()

    agg = season.groupby("player_id").agg(
        minutes=("minutes_played", "sum"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        n_apps=("appearance_id", "count"),
    ).reset_index()

    # Minimum 90 minutes to be in the pool
    agg = agg[agg["minutes"] >= 90].copy()
    agg["_90s"] = agg["minutes"] / 90
    agg["goals_p90"]        = (agg["goals"]        / agg["_90s"]).round(3)
    agg["assists_p90"]      = (agg["assists"]       / agg["_90s"]).round(3)
    agg["yellow_cards_p90"] = (agg["yellow_cards"]  / agg["_90s"]).clip(0, 2).round(3)
    agg["avg_min_per_game"]  = (agg["minutes"] / agg["n_apps"]).clip(0, 90).round(1)
    agg["apps_pct_season"]   = (agg["n_apps"] / 38).clip(0, 1).round(3)

    # Proxies
    agg["npxg_p90"]       = (agg["goals_p90"]   * 0.88).round(3)
    agg["key_passes_p90"] = (agg["assists_p90"] * 2.0).round(3)

    # --- Join player metadata ---
    tm_players["market_value_in_eur"] = pd.to_numeric(
        tm_players["market_value_in_eur"], errors="coerce"
    )
    tm_players["dob"] = pd.to_datetime(tm_players["date_of_birth"], errors="coerce")
    tm_players["age"] = ((pd.Timestamp("2025-01-01") - tm_players["dob"]).dt.days / 365.25).round(1)

    p_meta = tm_players[
        ["player_id", "name", "sub_position", "age", "market_value_in_eur", "current_club_id"]
    ].copy()
    p_meta = p_meta[p_meta["market_value_in_eur"].notna()]

    out = agg.merge(p_meta, on="player_id", how="inner")

    # Keep only players currently at Big-5 clubs
    out = out[out["current_club_id"].isin(big5_club_ids)].copy()

    out["position_group"]    = out["sub_position"].apply(_pos_group)
    out["market_value_eur_m"] = out["market_value_in_eur"] / 1_000_000
    out["player_name"]       = out["name"]
    out["league"]            = out["current_club_id"].map(club_to_league)

    print(f"  Players after 2024-25 Big-5 aggregation: {len(out):,}")

    # --- Market value momentum (current vs 12 months ago) ---
    vals = tm["player_valuations"].copy()
    vals["date"] = pd.to_datetime(vals["date"], errors="coerce")
    vals["market_value_in_eur"] = pd.to_numeric(vals["market_value_in_eur"], errors="coerce")
    vals = vals.dropna(subset=["date", "market_value_in_eur"]).sort_values("date")
    now_ref  = pd.Timestamp("2025-01-01")
    prev_ref = pd.Timestamp("2024-01-01")
    _player_ids = pd.DataFrame({"player_id": out["player_id"].unique()})
    for ref_date, col_name in [(now_ref, "_mv_now"), (prev_ref, "_mv_prev")]:
        ref_df = _player_ids.copy()
        ref_df["_ref_date"] = ref_date
        joined = pd.merge_asof(
            ref_df.sort_values("_ref_date"),
            vals[["player_id", "date", "market_value_in_eur"]],
            left_on="_ref_date", right_on="date", by="player_id", direction="backward",
        ).rename(columns={"market_value_in_eur": col_name})
        out = out.merge(joined[["player_id", col_name]], on="player_id", how="left")
    out["mv_momentum_12m"] = (
        (out["_mv_now"] - out["_mv_prev"]) / out["_mv_prev"].clip(lower=1e5)
    ).clip(-0.9, 3.0).fillna(0.0)
    out = out.drop(columns=["_mv_now", "_mv_prev"], errors="ignore")

    # --- Position × market-value heuristics ---
    POS_DEFAULTS = {
        # prog_passes/90, pass_cmp%, take_ons/90, prog_carries/90, aerials/90
        "GK":      (2.0, 67.0, 0.10, 0.30, 2.0),
        "DEF":     (3.8, 79.0, 0.70, 1.20, 2.8),
        "MID":     (5.5, 82.0, 1.40, 2.20, 0.9),
        "ATT":     (2.8, 77.0, 2.80, 3.20, 0.7),
        "Unknown": (4.0, 79.0, 1.20, 1.80, 1.2),
    }
    mv_max   = out["market_value_eur_m"].clip(0.1).max()
    mv_scale = (np.log1p(out["market_value_eur_m"].clip(0.1)) /
                np.log1p(mv_max)).clip(0.5, 1.4)

    for col_idx, col in enumerate(
        ["prog_passes_p90", "pass_completion_pct", "take_ons_p90",
         "prog_carries_p90", "aerials_won_p90"]
    ):
        defaults = out["position_group"].map(
            {pos: v[col_idx] for pos, v in POS_DEFAULTS.items()}
        ).fillna(POS_DEFAULTS["Unknown"][col_idx])
        if col == "pass_completion_pct":
            out[col] = (defaults + mv_scale * 2).clip(55, 95).round(1)
        else:
            out[col] = (defaults * mv_scale).round(3)

    # --- FBref pressing stats (tkl_won_p90, interceptions_p90) ---
    # Position-average fallbacks for unmatched players
    PRESS_POS_DEFAULTS = {
        # tkl_won_p90, interceptions_p90
        "GK":      (0.05, 0.2),
        "DEF":     (1.80, 1.5),
        "MID":     (1.50, 1.2),
        "ATT":     (0.60, 0.5),
        "Unknown": (1.20, 1.0),
    }
    try:
        from fbref_advanced import fetch_fbref_pressing_stats, enrich_with_pressing
        press_df = fetch_fbref_pressing_stats(["2024-25"], cache_dir=CACHE_DIR, force_refresh=False)
        out = enrich_with_pressing(out, press_df, season_label="2024-25")
    except Exception as exc:
        print(f"  [fbref_press] skipping (not yet cached or failed): {exc}")
        out["tkl_won_p90"]       = np.nan
        out["interceptions_p90"] = np.nan

    for press_col, pos_idx in [("tkl_won_p90", 0), ("interceptions_p90", 1)]:
        mask = out[press_col].isna()
        if mask.any():
            fallback = out.loc[mask, "position_group"].map(
                {pos: v[pos_idx] for pos, v in PRESS_POS_DEFAULTS.items()}
            ).fillna(PRESS_POS_DEFAULTS["Unknown"][pos_idx])
            out.loc[mask, press_col] = fallback.values
        out[press_col] = out[press_col].clip(0).round(3)

    # --- Contract months remaining ---
    today = pd.Timestamp.now()
    contract_dates = tm_players.set_index("player_id")["contract_expiration_date"]
    out["_contract_date"] = pd.to_datetime(
        out["player_id"].map(contract_dates), errors="coerce"
    )
    out["contract_months_remaining"] = (
        (out["_contract_date"] - today).dt.days / 30.44
    ).clip(lower=0)
    med_contract = out["contract_months_remaining"].median()
    out["contract_months_remaining"] = out["contract_months_remaining"].fillna(med_contract).round(1)
    out = out.drop(columns=["_contract_date"], errors="ignore")

    # --- Injury features ---
    inj_df = fetch_player_injuries_bulk(out["player_id"].tolist())
    inj_feats = compute_injury_features(inj_df, out["player_id"].tolist())
    out = out.merge(inj_feats, on="player_id", how="left")
    out["injury_days_last_2y"] = out["injury_days_last_2y"].fillna(0.0)
    out["has_serious_injury"]  = out["has_serious_injury"].fillna(0.0)

    keep = ["player_id", "player_name", "position_group", "age",
            "minutes", "league", "market_value_eur_m", "current_club_id"] + STAT_COLS
    out = out[[c for c in keep if c in out.columns]].copy()
    out[STAT_COLS] = out[STAT_COLS].apply(pd.to_numeric, errors="coerce")
    out[STAT_COLS] = out[STAT_COLS].fillna(0.0)
    out = out.dropna(subset=["player_id", "age"]).reset_index(drop=True)
    out["player_id"] = out["player_id"].astype(int)
    out["season"] = "2024-2025"
    print(f"  {len(out):,} players in final pool")
    return out


def _pos_group(sub_pos: str | None) -> str:
    if not isinstance(sub_pos, str):
        return "Unknown"
    s = sub_pos.lower()
    if "goalkeeper" in s:
        return "GK"
    if "back" in s or "defender" in s:
        return "DEF"
    if "midfield" in s:
        return "MID"
    if "winger" in s or "striker" in s or "forward" in s:
        return "ATT"
    return "Unknown"


# ---------------------------------------------------------------------------
# 4. Build per-season player stats from TM appearances
# ---------------------------------------------------------------------------
def build_player_seasons_df(tm: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate TM appearances into per-season player stats.

    Primary source: Big-5 league appearances (real league column).
    Fallback:       all other competitions for (player_id, season_end_year)
                    pairs not covered by Big-5 (league = "Other").

    Returns one row per (player_id, season_end_year) with:
      - Real stats:    goals_p90, assists_p90 (from appearances)
      - Proxied:       npxg_p90, key_passes_p90
      - Heuristic:     tackles, interceptions, prog_passes, take_ons,
                       prog_carries, aerials, pass_completion (position × value)
      - Context:       age, market_value_eur_m (at season start), league

    season_end_year convention: 2021 = the 2020-21 season (ends Jun 2021).
    """
    apps = tm["appearances"].copy()
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    apps = apps.dropna(subset=["date"])

    # Season label: Aug-Dec Y → ends Y+1; Jan-Jul Y → ends Y
    apps["season_end_year"] = np.where(
        apps["date"].dt.month >= 8,
        apps["date"].dt.year + 1,
        apps["date"].dt.year,
    )
    # Extend back to 2016 to cover transfers from 2018 onward (pre_season_year ≥ 2017)
    apps = apps[(apps["season_end_year"] >= 2016) & (apps["season_end_year"] <= 2024)].copy()

    apps["goals"] = pd.to_numeric(apps.get("goals", pd.Series(0, index=apps.index)), errors="coerce").fillna(0)
    apps["assists"] = pd.to_numeric(apps.get("assists", pd.Series(0, index=apps.index)), errors="coerce").fillna(0)
    apps["minutes_played"] = pd.to_numeric(apps["minutes_played"], errors="coerce").fillna(0)

    def _agg_seasons(df: pd.DataFrame, default_league: str) -> pd.DataFrame:
        result = df.groupby(["player_id", "season_end_year"]).agg(
            minutes=("minutes_played", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            n_apps=("minutes_played", "count"),
            league=("league", lambda x: x.mode().iloc[0] if len(x) > 0 else default_league),
        ).reset_index()
        return result[result["minutes"] >= 90].copy()

    # --- Primary: Big-5 ---
    big5 = apps[apps["competition_id"].isin(BIG5_COMP)].copy()
    big5["league"] = big5["competition_id"].map(BIG5_COMP)
    big5_stats = _agg_seasons(big5, "Premier League")

    # --- Fallback: non-Big-5, only for (player, season) gaps ---
    non_big5 = apps[~apps["competition_id"].isin(BIG5_COMP)].copy()
    non_big5["league"] = "Other"
    non_big5_stats = _agg_seasons(non_big5, "Other")

    # Anti-join: drop non-Big-5 rows already covered by Big-5
    covered = big5_stats[["player_id", "season_end_year"]].copy()
    covered["_in_big5"] = True
    non_big5_stats = non_big5_stats.merge(covered, on=["player_id", "season_end_year"], how="left")
    non_big5_stats = non_big5_stats[non_big5_stats["_in_big5"].isna()].drop(columns="_in_big5")

    season_stats = pd.concat([big5_stats, non_big5_stats], ignore_index=True)

    mins90 = (season_stats["minutes"] / 90).clip(lower=0.1)
    season_stats["goals_p90"] = (season_stats["goals"] / mins90).round(3).clip(0, 3)
    season_stats["assists_p90"] = (season_stats["assists"] / mins90).round(3).clip(0, 2)
    season_stats["npxg_p90"] = (season_stats["goals_p90"] * 0.88).round(3)
    season_stats["key_passes_p90"] = (season_stats["assists_p90"] * 2.0).round(3)
    season_stats["yellow_cards_p90"] = (season_stats["yellow_cards"] / mins90).round(3).clip(0, 2)
    season_stats["avg_min_per_game"]  = (season_stats["minutes"] / season_stats["n_apps"]).clip(0, 90).round(1)
    season_stats["apps_pct_season"]   = (season_stats["n_apps"] / 38).clip(0, 1).round(3)

    # Market value at season start via merge_asof
    valuations = tm["player_valuations"].copy()
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    valuations["market_value_in_eur"] = pd.to_numeric(valuations["market_value_in_eur"], errors="coerce")
    valuations = valuations.dropna(subset=["date", "market_value_in_eur"]).sort_values("date")

    season_stats["season_start_date"] = pd.to_datetime(
        season_stats["season_end_year"].apply(lambda y: f"{y - 1}-08-01")
    )
    ss_sorted = season_stats.sort_values("season_start_date")

    mv_joined = pd.merge_asof(
        ss_sorted[["player_id", "season_end_year", "season_start_date"]],
        valuations[["player_id", "date", "market_value_in_eur"]],
        left_on="season_start_date",
        right_on="date",
        by="player_id",
        direction="backward",
    )
    season_stats = season_stats.merge(
        mv_joined[["player_id", "season_end_year", "market_value_in_eur"]],
        on=["player_id", "season_end_year"],
        how="left",
    )
    season_stats["market_value_eur_m"] = (
        season_stats["market_value_in_eur"].fillna(1_000_000) / 1_000_000
    ).clip(0.05)

    # Market value 12 months prior → momentum signal
    season_stats["prev_year_date"] = pd.to_datetime(
        season_stats["season_end_year"].apply(lambda y: f"{y - 2}-08-01")
    )
    prev_sorted = season_stats.sort_values("prev_year_date")
    mv_prev_joined = pd.merge_asof(
        prev_sorted[["player_id", "season_end_year", "prev_year_date"]],
        valuations[["player_id", "date", "market_value_in_eur"]],
        left_on="prev_year_date",
        right_on="date",
        by="player_id",
        direction="backward",
    ).rename(columns={"market_value_in_eur": "mv_prev_eur"})
    season_stats = season_stats.merge(
        mv_prev_joined[["player_id", "season_end_year", "mv_prev_eur"]],
        on=["player_id", "season_end_year"],
        how="left",
    )
    season_stats["mv_momentum_12m"] = (
        (season_stats["market_value_in_eur"].fillna(season_stats["mv_prev_eur"])
         - season_stats["mv_prev_eur"])
        / season_stats["mv_prev_eur"].clip(lower=1e5)
    ).clip(-0.9, 3.0).fillna(0.0)

    # Player metadata
    tm_players = tm["players"][["player_id", "name", "sub_position", "date_of_birth"]].copy()
    tm_players["position_group"] = tm_players["sub_position"].apply(_pos_group)
    tm_players["dob_year"] = pd.to_datetime(tm_players["date_of_birth"], errors="coerce").dt.year

    season_stats = season_stats.merge(
        tm_players[["player_id", "name", "position_group", "dob_year"]],
        on="player_id", how="left",
    )
    season_stats["player_name"] = season_stats["name"].fillna("")
    season_stats["position_group"] = season_stats["position_group"].fillna("Unknown")
    season_stats["age"] = (season_stats["season_end_year"] - 1 - season_stats["dob_year"]).clip(15, 45).fillna(25)

    # Position × market-value heuristics for stats not in TM appearances
    POS_DEFAULTS = {
        #          prog_pass  pass%   take_on  prog_car  aerial
        "GK":      (2.0,  67.0, 0.10, 0.30, 2.0),
        "DEF":     (3.8,  79.0, 0.70, 1.20, 2.8),
        "MID":     (5.5,  82.0, 1.40, 2.20, 0.9),
        "ATT":     (2.8,  77.0, 2.80, 3.20, 0.7),
        "Unknown": (4.0,  79.0, 1.20, 1.80, 1.2),
    }
    mv_max = season_stats["market_value_eur_m"].clip(0.1).max()
    mv_scale = (
        np.log1p(season_stats["market_value_eur_m"].clip(0.1)) / np.log1p(mv_max)
    ).clip(0.5, 1.4)

    heuristic_cols = [
        "prog_passes_p90", "pass_completion_pct", "take_ons_p90",
        "prog_carries_p90", "aerials_won_p90",
    ]
    for col_idx, col in enumerate(heuristic_cols):
        defaults = season_stats["position_group"].map(
            {pos: vals[col_idx] for pos, vals in POS_DEFAULTS.items()}
        ).fillna(POS_DEFAULTS["Unknown"][col_idx])
        if col == "pass_completion_pct":
            season_stats[col] = (defaults + mv_scale * 2).clip(55, 95).round(1)
        else:
            season_stats[col] = (defaults * mv_scale).round(3)

    # --- FBref pressing stats: add season_label for join ---
    PRESS_POS_DEFAULTS_S = {
        "GK":      (0.05, 0.2),
        "DEF":     (1.80, 1.5),
        "MID":     (1.50, 1.2),
        "ATT":     (0.60, 0.5),
        "Unknown": (1.20, 1.0),
    }
    # Map season_end_year (e.g. 2023) → season_label "2022-23"
    season_stats["season_label"] = season_stats["season_end_year"].apply(
        lambda y: f"20{str(y-1)[2:]}-{str(y)[2:]}"
    )
    try:
        from fbref_advanced import fetch_fbref_pressing_stats, enrich_with_pressing, SUPPORTED_SEASONS
        all_labels = [l for l in season_stats["season_label"].unique() if l in SUPPORTED_SEASONS]
        press_df = fetch_fbref_pressing_stats(all_labels, cache_dir=CACHE_DIR)
        season_stats = enrich_with_pressing(
            season_stats, press_df,
            season_label=None, season_label_col="season_label",
            league_col="league", name_col="player_name",
        )
    except Exception as exc:
        print(f"  [fbref_press] skipping for seasons: {exc}")
        season_stats["tkl_won_p90"]       = np.nan
        season_stats["interceptions_p90"] = np.nan

    for press_col, pos_idx in [("tkl_won_p90", 0), ("interceptions_p90", 1)]:
        mask = season_stats[press_col].isna()
        if mask.any():
            fallback = season_stats.loc[mask, "position_group"].map(
                {pos: v[pos_idx] for pos, v in PRESS_POS_DEFAULTS_S.items()}
            ).fillna(PRESS_POS_DEFAULTS_S["Unknown"][pos_idx])
            season_stats.loc[mask, press_col] = fallback.values
        season_stats[press_col] = season_stats[press_col].clip(0).round(3)

    present_stat_cols = [c for c in STAT_COLS if c in season_stats.columns]
    season_stats[present_stat_cols] = season_stats[present_stat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    season_stats = season_stats.drop(
        columns=["season_start_date", "prev_year_date", "dob_year",
                 "market_value_in_eur", "mv_prev_eur", "goals", "assists", "yellow_cards",
                 "name", "season_label"],
        errors="ignore",
    )
    n_big5 = len(big5_stats)
    n_fallback = len(non_big5_stats)
    print(f"  {len(season_stats):,} player-seasons (Big-5: {n_big5:,} | non-Big-5 fallback: {n_fallback:,}, 2016–2024)")
    return season_stats


# ---------------------------------------------------------------------------
# 5. Build transfers_df — TM transfers + outcome labels + temporal player stats
# ---------------------------------------------------------------------------
def build_transfers_df(
    tm: dict[str, pd.DataFrame],
    player_seasons_df: pd.DataFrame,
    clubs_df: pd.DataFrame,
    players_df: pd.DataFrame,
) -> pd.DataFrame:
    transfers = tm["transfers"].copy()
    appearances = tm["appearances"].copy()
    valuations = tm["player_valuations"].copy()

    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    transfers = transfers.dropna(subset=["transfer_date"])
    transfers = transfers[
        (transfers["transfer_date"] >= "2018-01-01") &
        (transfers["transfer_date"] < "2024-01-01")
    ].copy()

    # Keep only Big-5 destination clubs that we have features for
    valid_club_ids = set(clubs_df["club_id"].tolist())
    transfers = transfers[transfers["to_club_id"].isin(valid_club_ids)].copy()

    # Keep only players we have FBref stats for
    valid_player_ids = set(players_df["player_id"].tolist())
    transfers = transfers[transfers["player_id"].isin(valid_player_ids)].copy()

    print(f"  {len(transfers):,} transfers to process for outcome labels...")

    # --- Pre-index for speed ---
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    big5_app = appearances[appearances["competition_id"].isin(BIG5_COMP)].copy()
    app_grp = big5_app.groupby(["player_id", "player_club_id"])

    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    val_grp = valuations.sort_values("date").groupby("player_id")

    club_comp_map = (
        clubs_df.set_index("club_id")["league"]
        .map({v: k for k, v in BIG5_COMP.items()})
        .to_dict()
    )

    out_transfers = tm["transfers"].copy()
    out_transfers["transfer_date"] = pd.to_datetime(out_transfers["transfer_date"], errors="coerce")

    minutes_share, value_delta, survival, goal_contributions = [], [], [], []

    for _, row in tqdm(transfers.iterrows(), total=len(transfers), desc="  labels"):
        td = row["transfer_date"]

        # Season N+1 window: if before Oct 1 → starts same Aug, else next Aug
        n1_start = pd.Timestamp(td.year if td.month < 10 else td.year + 1, 8, 1)
        n1_end = n1_start + pd.DateOffset(months=11)

        # minutes_share_y1 + goal_contribution_y1 (same grp lookup, same window)
        try:
            grp = app_grp.get_group((row["player_id"], row["to_club_id"]))
            mask = (grp["date"] >= n1_start) & (grp["date"] <= n1_end)
            year1 = grp.loc[mask]
            mins = float(year1["minutes_played"].sum())
            comp_id = club_comp_map.get(row["to_club_id"], "GB1")
            max_min = MAX_SEASON_MIN.get(comp_id, 3420)
            minutes_share.append(min(mins / max_min, 1.0))
            # goals + assists per 90 in year 1, clipped to [0, 1]
            g_y1 = float(year1["goals"].sum()) if "goals" in year1.columns else 0.0
            a_y1 = float(year1["assists"].sum()) if "assists" in year1.columns else 0.0
            gc_p90 = (g_y1 + a_y1) / max(mins / 90, 1.0)
            goal_contributions.append(min(gc_p90, 1.0))
        except KeyError:
            minutes_share.append(np.nan)
            goal_contributions.append(np.nan)

        # value_delta_18m
        def _val_at(pid: int, target: pd.Timestamp) -> float:
            try:
                g = val_grp.get_group(pid)
                past = g[g["date"] <= target]
                return float(past.iloc[-1]["market_value_in_eur"]) if not past.empty else np.nan
            except KeyError:
                return np.nan

        vt = _val_at(row["player_id"], td)
        v18 = _val_at(row["player_id"], td + pd.DateOffset(months=18))
        if vt and vt > 0 and not np.isnan(v18):
            value_delta.append(float(np.clip(v18 / vt - 1, -0.9, 3.0)))
        else:
            value_delta.append(np.nan)

        # survival_2y
        deadline = td + pd.DateOffset(months=24)
        if deadline > pd.Timestamp("2024-06-01"):
            survival.append(np.nan)
        else:
            left = out_transfers[
                (out_transfers["player_id"] == row["player_id"]) &
                (out_transfers["from_club_id"] == row["to_club_id"]) &
                (out_transfers["transfer_date"] > td) &
                (out_transfers["transfer_date"] <= deadline)
            ]
            survival.append(0 if len(left) > 0 else 1)

    transfers["minutes_share_y1"] = minutes_share
    transfers["value_delta_18m"] = value_delta
    transfers["survival_2y"] = survival
    transfers["goal_contribution_y1"] = goal_contributions

    # Require minutes_share (primary label); drop rows where it is missing.
    transfers = transfers.dropna(subset=["minutes_share_y1"]).copy()
    transfers = transfers.reset_index(drop=True)

    # survival_2y is unknown for transfers after mid-2022 (window not elapsed).
    surv_proxy = (transfers["minutes_share_y1"] >= 0.30).astype(float)
    surv_filled = transfers["survival_2y"].fillna(surv_proxy)

    # goal_contribution_y1 may be NaN when minutes_share is very low; fill 0.
    gc_filled = transfers["goal_contribution_y1"].fillna(0.0)

    # pre_season_year needed here for the fit_surprise computation below.
    transfers["pre_season_year"] = np.where(
        transfers["transfer_date"].dt.month >= 7,
        transfers["transfer_date"].dt.year,
        transfers["transfer_date"].dt.year - 1,
    )

    # fit_surprise: did the player play MORE than their pre-transfer history predicted?
    # A player who normally plays 80% of games but plays only 30% at the new club
    # almost certainly doesn't fit the system — regardless of raw quality.
    # This gives the model a tactical-fit signal beyond pure player quality.
    ps_apps = (
        player_seasons_df[["player_id", "season_end_year", "apps_pct_season"]]
        .dropna(subset=["apps_pct_season"])
        .sort_values(["player_id", "season_end_year"])
        .copy()
    )
    # For each player×season, expected = rolling mean of the prior 2 seasons
    ps_apps["expected_apps_pct"] = (
        ps_apps.groupby("player_id")["apps_pct_season"]
        .transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
    )
    ps_apps = ps_apps.drop_duplicates(subset=["player_id", "season_end_year"])
    transfers = transfers.merge(
        ps_apps[["player_id", "season_end_year", "expected_apps_pct"]].rename(
            columns={"season_end_year": "pre_season_year"}
        ),
        on=["player_id", "pre_season_year"],
        how="left",
    ).reset_index(drop=True)

    # fit_surprise: actual minutes - expected minutes.
    # Raw mean is negative (~-0.22) because most transfers are upward moves —
    # players naturally play fewer minutes when joining a better club.
    # We centre on the population median so the signal measures "better or worse
    # than typical for this cohort", not "better or worse than own previous club".
    fit_surprise_raw = transfers["minutes_share_y1"] - transfers["expected_apps_pct"]
    pop_median = float(fit_surprise_raw.median())          # ≈ -0.15 to -0.25
    fit_surprise_centered = fit_surprise_raw - pop_median  # centred ≈ 0
    # Clip to [-0.5, 0.5] then shift to [0, 1]; NaN rows (no baseline) → 0.5 (neutral)
    fit_surprise = (fit_surprise_centered.clip(-0.5, 0.5) + 0.5).fillna(0.5).values
    # Use .values throughout to prevent index-alignment NaN after the merge
    surv_arr = surv_filled.values
    gc_arr   = gc_filled.values
    min_arr  = transfers["minutes_share_y1"].values
    transfers = transfers.drop(columns=["expected_apps_pct"])
    n_surprise = int(fit_surprise_raw.notna().sum())
    print(f"  fit_surprise: coverage {n_surprise:,}/{len(transfers):,}, "
          f"raw_mean={fit_surprise_raw.mean():.3f}, pop_median={pop_median:.3f}")

    # Target: did the transfer work out for the club?
    #   35% — playing time in season 1 (coach gave them the shirt)
    #   30% — goal contribution per 90 in year 1 (productive output)
    #   20% — stayed at the club for 2 years (relationship held)
    #   15% — fit_surprise: played more than pre-transfer history predicted
    #          (tactical-fit signal independent of raw quality)
    transfers["success_score"] = np.round(
        0.35 * min_arr + 0.30 * gc_arr + 0.20 * surv_arr + 0.15 * fit_surprise, 3
    )

    # Accurate transfer_age from birth dates
    tm_dob = tm["players"][["player_id", "date_of_birth"]].drop_duplicates("player_id").copy()
    tm_dob["dob"] = pd.to_datetime(tm_dob["date_of_birth"], errors="coerce")
    dob_map = tm_dob.set_index("player_id")["dob"].to_dict()
    transfers["transfer_age"] = transfers.apply(
        lambda r: int((r["transfer_date"] - dob_map[r["player_id"]]).days / 365.25)
        if r["player_id"] in dob_map and not pd.isna(dob_map.get(r["player_id"]))
        else np.nan,
        axis=1,
    ).fillna(25.0)

    # -----------------------------------------------------------------------
    # Temporal join: embed player stats from the season BEFORE the transfer.
    # pre_season_year already computed above for fit_surprise; reset index for
    # the merge_asof that follows.
    # -----------------------------------------------------------------------
    transfers = transfers.reset_index(drop=True)

    embed_cols = STAT_COLS + ["age", "market_value_eur_m", "position_group", "league"]
    embed_cols = [c for c in embed_cols if c in player_seasons_df.columns]

    # merge_asof: for each transfer, take the most recent player season ≤ pre_season_year
    transfers["_orig_idx"] = range(len(transfers))
    t_sorted = transfers[["_orig_idx", "player_id", "pre_season_year"]].sort_values(
        "pre_season_year"
    )
    ps_sorted = player_seasons_df[["player_id", "season_end_year"] + embed_cols].sort_values(
        "season_end_year"
    )

    joined = pd.merge_asof(
        t_sorted,
        ps_sorted,
        left_on="pre_season_year",
        right_on="season_end_year",
        by="player_id",
        direction="backward",
    ).sort_values("_orig_idx")

    for col in embed_cols:
        transfers[col] = joined[col].values

    # Position-median fallback for players with no TM season data
    POS_MEDIANS = {
        "goals_p90": {"GK": 0.0, "DEF": 0.05, "MID": 0.15, "ATT": 0.45, "Unknown": 0.1},
        "assists_p90": {"GK": 0.0, "DEF": 0.04, "MID": 0.18, "ATT": 0.20, "Unknown": 0.1},
    }
    nan_mask = transfers["goals_p90"].isna()
    if nan_mask.any():
        # Fill position_group from TM players
        pos_map = (
            tm["players"][["player_id", "sub_position"]]
            .drop_duplicates("player_id")
            .assign(position_group=lambda d: d["sub_position"].apply(_pos_group))
            .set_index("player_id")["position_group"]
            .to_dict()
        )
        transfers.loc[nan_mask, "position_group"] = transfers.loc[nan_mask, "player_id"].map(pos_map).fillna("Unknown")
        transfers.loc[nan_mask, "age"] = transfers.loc[nan_mask, "transfer_age"]
        transfers.loc[nan_mask, "market_value_eur_m"] = 5.0
        transfers.loc[nan_mask, "league"] = "Premier League"
        # Only fill columns that already exist in transfers (risk cols added later)
        for col in [c for c in STAT_COLS if c in transfers.columns]:
            if col in POS_MEDIANS:
                pg_series = transfers.loc[nan_mask, "position_group"]
                fills = pg_series.map(POS_MEDIANS[col]).fillna(0.1)
                transfers.loc[nan_mask, col] = fills
            else:
                transfers.loc[nan_mask, col] = 0.0

    transfers = transfers.drop(columns=["_orig_idx", "pre_season_year"], errors="ignore")

    # --- Contract months remaining at time of transfer ---
    tm_players_df = tm["players"].copy()
    contract_map  = pd.to_datetime(
        tm_players_df.set_index("player_id")["contract_expiration_date"], errors="coerce"
    )
    transfers["_contract_date"] = pd.to_datetime(
        transfers["player_id"].map(contract_map), errors="coerce"
    )
    transfers["contract_months_remaining"] = (
        (transfers["_contract_date"] - transfers["transfer_date"]).dt.days / 30.44
    ).clip(lower=0).fillna(12.0)   # 12-month fallback for missing contracts
    transfers = transfers.drop(columns=["_contract_date"], errors="ignore")

    # --- Injury features at time of transfer (vectorized) ---
    inj_df = fetch_player_injuries_bulk(transfers["player_id"].unique().tolist())
    if not inj_df.empty and "injury_date" in inj_df.columns:
        inj_clean = inj_df.dropna(subset=["injury_date"]).copy()
        inj_clean["injury_date"] = pd.to_datetime(inj_clean["injury_date"])
        # Cross-join transfers × injuries on player_id, then filter by date window
        t_tmp = transfers[["player_id", "transfer_date"]].copy().reset_index().rename(columns={"index": "_tid"})
        merged = t_tmp.merge(inj_clean, on="player_id", how="left")
        merged["days_before"] = (
            pd.to_datetime(merged["transfer_date"]) - merged["injury_date"]
        ).dt.days
        mask_2y = (merged["days_before"] >= 0) & (merged["days_before"] <= 730)
        mask_3y = (merged["days_before"] >= 0) & (merged["days_before"] <= 1095)
        days_agg = merged[mask_2y].groupby("_tid")["days_missed"].sum().rename("injury_days_last_2y")
        serious_agg = (
            merged[mask_3y].groupby("_tid")["days_missed"].max() > 60
        ).astype(float).rename("has_serious_injury")
        transfers["injury_days_last_2y"] = days_agg.reindex(transfers.index).fillna(0.0).values
        transfers["has_serious_injury"]  = serious_agg.reindex(transfers.index).fillna(0.0).values
    else:
        transfers["injury_days_last_2y"] = 0.0
        transfers["has_serious_injury"]  = 0.0

    # --- Loan detection (vectorised) ---
    # A transfer is flagged as a loan when the player departs the destination
    # club back to any club within 18 months. This catches standard loans and
    # season-long loans without requiring an explicit loan-type column.
    all_tm = tm["transfers"].copy()
    all_tm["transfer_date"] = pd.to_datetime(all_tm["transfer_date"], errors="coerce")
    all_tm = all_tm.dropna(subset=["transfer_date"])

    transfers["_idx"] = range(len(transfers))
    future_moves = all_tm[["player_id", "from_club_id", "transfer_date"]].rename(
        columns={"transfer_date": "future_date", "from_club_id": "future_from"}
    )
    loan_check = transfers[["_idx", "player_id", "to_club_id", "transfer_date"]].merge(
        future_moves, on="player_id", how="left"
    )
    loan_check["days_diff"] = (loan_check["future_date"] - loan_check["transfer_date"]).dt.days
    loan_mask_flag = (
        (loan_check["future_from"] == loan_check["to_club_id"]) &
        (loan_check["days_diff"] > 0) &
        (loan_check["days_diff"] <= 548)   # 18 months
    )
    loan_idxs = set(loan_check.loc[loan_mask_flag, "_idx"].unique())
    # Require fee == 0: paid transfers where the player moved on quickly are not loans,
    # just volatile permanent signings — keep them, they carry real club-fit signal.
    fee_zero = pd.to_numeric(transfers["transfer_fee"], errors="coerce").fillna(0) == 0
    transfers["is_loan"] = transfers["_idx"].isin(loan_idxs) & fee_zero
    transfers = transfers.drop(columns=["_idx"])

    n_loans = transfers["is_loan"].sum()
    print(f"  {n_loans:,} transfers flagged as loans ({n_loans/len(transfers)*100:.0f}%)")

    out_cols = [
        "player_id", "to_club_id", "transfer_age", "transfer_fee",
        "transfer_date", "is_loan",
        "minutes_share_y1", "value_delta_18m", "survival_2y",
        "goal_contribution_y1", "success_score",
        "contract_months_remaining", "injury_days_last_2y", "has_serious_injury",
    ] + embed_cols
    out = transfers[[c for c in out_cols if c in transfers.columns]].copy()
    out = out.rename(columns={"to_club_id": "destination_club_id", "transfer_fee": "fee_eur_m"})
    out["fee_eur_m"] = pd.to_numeric(out["fee_eur_m"], errors="coerce").fillna(0.0).clip(0)
    out = out.reset_index(drop=True)
    out.index.name = "transfer_id"
    out = out.reset_index()
    matched_pct = (1 - nan_mask.sum() / len(out)) * 100
    print(f"  {len(out):,} transfers with outcome labels  ({matched_pct:.0f}% with season-aligned stats)")
    return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z\s]", "", s).lower().strip()
    return re.sub(r"\s+", " ", s)


# ---------------------------------------------------------------------------
# 5. generate()
# ---------------------------------------------------------------------------
def generate(
    seasons: list[str] = SEASONS,
    save_dir: str | None = None,
    force_rebuild: bool = False,
) -> dict:
    """
    Fetch / load from cache and return {players, clubs, transfers}.
    Drop-in replacement for synthetic.generate().

    On subsequent calls the three parquet files under save_dir (or data/) are
    loaded directly, skipping the slow fuzzy-match and label-computation steps.
    Pass force_rebuild=True to redo the full ETL.
    """
    out_dir = Path(save_dir) if save_dir else DATA_DIR
    cached = (
        out_dir / "real_players.parquet",
        out_dir / "real_clubs.parquet",
        out_dir / "real_transfers.parquet",
    )
    if not force_rebuild and all(p.exists() for p in cached):
        import pyarrow.parquet as pq
        cached_cols = pq.read_schema(cached[2]).names
        required_cols = {"goals_p90", "contract_months_remaining", "injury_days_last_2y", "has_serious_injury"}
        if required_cols.issubset(set(cached_cols)):
            print("  [cache] Loading real data from parquet files...")
            players_df = pd.read_parquet(cached[0])
            clubs_df = pd.read_parquet(cached[1])
            transfers_df = pd.read_parquet(cached[2])
            print(f"  players={len(players_df):,}  clubs={len(clubs_df):,}  "
                  f"transfers={len(transfers_df):,}")
            return {"players": players_df, "clubs": clubs_df, "transfers": transfers_df}
        else:
            print("  [cache] Stale format (missing new columns) — rebuilding...")

    print("=" * 60)
    print("REAL DATA ETL — AI Football Scout v3")
    print("=" * 60)

    print("\n[1/4] Downloading Transfermarkt datasets...")
    tm = fetch_tm()

    print("\n[2/4] Building players_df from v2 scouting pool...")
    players_df = build_players_df(tm)

    print("\n[3/4] Building clubs_df from football-data.co.uk...")
    clubs_df = build_clubs_df(tm)

    print("\n[3b/4] Building per-season player stats from TM appearances...")
    player_seasons_df = build_player_seasons_df(tm)

    print("\n[4/4] Building transfers_df with outcome labels + temporal stats...")
    transfers_df = build_transfers_df(tm, player_seasons_df, clubs_df, players_df)

    print(f"\n{'=' * 60}")
    print("Dataset summary")
    print(f"  players:   {len(players_df):,} (2024-25 FBref stats, used for inference)")
    print(f"  clubs:     {len(clubs_df):,}")
    print(f"  transfers: {len(transfers_df):,} labelled (season-aligned player stats)")
    print(f"  success_score: mean={transfers_df['success_score'].mean():.3f}  "
          f"std={transfers_df['success_score'].std():.3f}")

    out = {"players": players_df, "clubs": clubs_df, "transfers": transfers_df}

    if save_dir:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        players_df.to_parquet(p / "real_players.parquet")
        clubs_df.to_parquet(p / "real_clubs.parquet")
        transfers_df.to_parquet(p / "real_transfers.parquet")
        print(f"\nSaved to {save_dir}/")

    return out


if __name__ == "__main__":
    data = generate(save_dir="data")
    print("\nPlayers sample:")
    print(data["players"][["player_name", "league", "position_group",
                            "age", "market_value_eur_m", "goals_p90",
                            "tackles_p90"]].head(5).to_string())
    print("\nClubs sample:")
    print(data["clubs"][["club_id", "league", "possession_pct",
                          "ppda", "directness_idx", "line_height_m"]].head(5).to_string())
    print("\nTransfers sample:")
    print(data["transfers"][["player_id", "destination_club_id", "transfer_age",
                              "success_score"]].head(5).to_string())
