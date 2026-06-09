"""
Fetches FBref pressing stats (tackles won, interceptions) from the misc table.

FBref's passing/possession tables use JavaScript to populate stats — the HTML
cells are empty and pd.read_html returns NaN. The misc table IS fully populated
in static HTML (soccerdata already fetches it correctly).

This module adds two genuine new features to the player model:
  tkl_won_p90       — tackles won per 90 (direct defensive pressing effort)
  interceptions_p90 — interceptions per 90 (off-ball positioning, press triggers)

These are not available from Transfermarkt and were completely absent from the
model before. They directly measure what high-press clubs care about.

Usage:
    from fbref_advanced import fetch_fbref_pressing_stats, enrich_with_pressing
    df = fetch_fbref_pressing_stats(["2022-23", "2023-24"], cache_dir=Path("data/cache"))
    players_df = enrich_with_pressing(players_df, df, season_label="2024-25")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED_SEASONS = [
    "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25",
]

FBREF_LEAGUE_MAP = {
    "ENG-Premier League": "Premier League",
    "FRA-Ligue 1":        "Ligue 1",
    "ESP-La Liga":        "La Liga",
    "GER-Bundesliga":     "Bundesliga",
    "ITA-Serie A":        "Serie A",
}

# All 5 Big-5 leagues as individual soccerdata league strings
BIG5_LEAGUES = list(FBREF_LEAGUE_MAP.keys())


def _norm_name(s: str) -> str:
    """Normalise player name for fuzzy matching: strip accents, lowercase."""
    import unicodedata
    if not isinstance(s, str):
        return ""
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )


def _skey_to_label(skey: str) -> str:
    """Convert soccerdata season key '2223' → '2022-23'."""
    return f"20{skey[:2]}-{skey[2:]}"


def fetch_fbref_pressing_stats(
    seasons: list[str],
    cache_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch tackles won and interceptions per 90 for Big-5 players.

    Fetches the 'misc' stat table from FBref via soccerdata for individual
    leagues (the Big-5 combined page's misc table is also fully populated,
    but per-league gives cleaner team attribution).

    Parameters
    ----------
    seasons : list[str]
        Season labels accepted by soccerdata, e.g. ["2022-23", "2023-24"].
    cache_dir : Path
        Where to store/read fbref_pressing.parquet.
    force_refresh : bool
        Re-fetch even if cached.

    Returns
    -------
    pd.DataFrame with columns:
        player_name, league, season_label, tkl_won_p90, interceptions_p90
    """
    import soccerdata as sd

    cache_path = cache_dir / "fbref_pressing.parquet"
    existing: pd.DataFrame | None = None
    already: set[str] = set()

    if not force_refresh and cache_path.exists():
        existing = pd.read_parquet(cache_path)
        already = set(existing["season_label"].unique())

    needed = [s for s in seasons if s not in already]
    if not needed:
        print(f"  [fbref_press] all {len(seasons)} seasons cached — skipping fetch")
        return existing if existing is not None else pd.DataFrame()

    print(f"  [fbref_press] fetching misc stats for {len(needed)} seasons: {needed}")

    new_frames: list[pd.DataFrame] = []
    for season_label in needed:
        frames: list[pd.DataFrame] = []
        for fbref_league in BIG5_LEAGUES:
            our_league = FBREF_LEAGUE_MAP[fbref_league]
            try:
                fb = sd.FBref([fbref_league], season_label)
                misc = fb.read_player_season_stats("misc")

                # Flatten MultiIndex columns
                flat_cols = [
                    "_".join(str(c).strip() for c in col if str(c).strip())
                    for col in misc.columns
                ]
                misc.columns = flat_cols

                # Reset index FIRST so all operations use integer index
                misc_reset = misc.reset_index()
                player_col = "player" if "player" in misc_reset.columns else "Player"
                skey_col   = "season"  if "season"  in misc_reset.columns else "Season"

                ninety_s = pd.to_numeric(misc_reset.get("90s"), errors="coerce").replace(0, np.nan)
                tkl_raw  = pd.to_numeric(misc_reset.get("Performance_TklW", misc_reset.get("TklW")), errors="coerce")
                int_raw  = pd.to_numeric(misc_reset.get("Performance_Int",  misc_reset.get("Int")),  errors="coerce")

                row_df = pd.DataFrame({
                    "player_name":       misc_reset[player_col],
                    "tkl_won_p90":       (tkl_raw / ninety_s).round(3),
                    "interceptions_p90": (int_raw / ninety_s).round(3),
                })

                # Season label
                skeys = misc_reset[skey_col].unique() if skey_col in misc_reset.columns else ["????"]
                sk = str(skeys[0]) if len(skeys) > 0 else "????"
                row_df["season_label"] = _skey_to_label(sk)
                row_df["league"] = our_league
                frames.append(row_df.dropna(subset=["player_name"]))

            except Exception as exc:
                print(f"    {our_league} {season_label}: {exc}")

        if frames:
            season_df = pd.concat(frames, ignore_index=True)
            season_df = season_df.dropna(subset=["tkl_won_p90", "interceptions_p90"], how="all")
            new_frames.append(season_df)
            print(f"    {season_label}: {len(season_df):,} player-league rows")

    if not new_frames:
        print("  [fbref_press] no new data fetched")
        return existing if existing is not None else pd.DataFrame()

    combined = pd.concat(new_frames, ignore_index=True)
    if existing is not None:
        combined = pd.concat([existing, combined], ignore_index=True)

    combined = combined.drop_duplicates(
        subset=["player_name", "league", "season_label"], keep="last"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    print(f"  [fbref_press] saved {len(combined):,} rows to {cache_path}")
    return combined


def enrich_with_pressing(
    df: pd.DataFrame,
    press_df: pd.DataFrame,
    season_label: str | None = None,
    season_label_col: str = "season_label",
    league_col: str = "league",
    name_col: str = "player_name",
) -> pd.DataFrame:
    """
    Merge pressing stats onto df by (player_name_normalised, league, season_label).

    Missing matches keep the original value (NaN → position average fallback
    applied downstream in build_players_df / build_player_seasons_df).

    Parameters
    ----------
    df : pd.DataFrame
        Player pool or player-seasons DataFrame to enrich.
    press_df : pd.DataFrame
        Output of fetch_fbref_pressing_stats.
    season_label : str | None
        If provided, filters press_df to this season before joining.
        If None, the df must have a `season_label` column.
    season_label_col : str
        Column in df that holds the season label (default "season_label").

    Returns
    -------
    df with tkl_won_p90 and interceptions_p90 columns.
    """
    if press_df.empty:
        df["tkl_won_p90"]       = np.nan
        df["interceptions_p90"] = np.nan
        return df

    if season_label is not None:
        press_sub = press_df[press_df["season_label"] == season_label].copy()
    else:
        press_sub = press_df.copy()

    press_sub["_key"] = (
        press_sub["player_name"].apply(_norm_name) + "|"
        + press_sub["league"].str.lower().fillna("") + "|"
        + press_sub["season_label"].fillna("").astype(str)
    )
    press_sub = press_sub.drop_duplicates("_key").set_index("_key")

    df = df.copy()

    if season_label is not None:
        df["_sl"] = season_label
    else:
        df["_sl"] = df[season_label_col].fillna("").astype(str)

    df["_key"] = (
        df[name_col].apply(_norm_name) + "|"
        + df[league_col].str.lower().fillna("") + "|"
        + df["_sl"]
    )

    for col in ["tkl_won_p90", "interceptions_p90"]:
        if col not in press_sub.columns:
            df[col] = np.nan
            continue
        vals = df["_key"].map(press_sub[col])
        df[col] = vals  # NaN where unmatched — caller fills with position average

    n_matched = df["tkl_won_p90"].notna().sum()
    print(f"  [fbref_press] matched {n_matched:,} / {len(df):,} players")
    df = df.drop(columns=["_key", "_sl"], errors="ignore")
    return df
