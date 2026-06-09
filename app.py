"""
AI Football Scout v3 — Streamlit UI
Run: streamlit run app.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from real_data import generate
from scout import Filters, DestinationClub, apply_filters, compute_fit_score, get_player_embeddings

st.set_page_config(
    page_title="AI Football Scout v3",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Plotly light theme ───────────────────────────────────────────
PLOTLY = dict(
    plot_bgcolor  = "#ffffff",
    paper_bgcolor = "#f8fafc",
    font          = dict(color="#0f172a", family="sans-serif"),
    xaxis         = dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1"),
    yaxis         = dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1"),
    margin        = dict(l=10, r=20, t=35, b=10),
)

POLAR_STYLE = dict(
    radialaxis=dict(
        visible=True, range=[0, 100],
        gridcolor="#e2e8f0", tickfont=dict(size=9, color="#64748b"),
    ),
    angularaxis=dict(
        tickfont=dict(size=11, color="#374151"),
        gridcolor="#e2e8f0",
    ),
    bgcolor="#ffffff",
)

def pl(**overrides):
    """Merge PLOTLY base with per-chart overrides (nested dicts merged, not replaced)."""
    base = {k: dict(v) if isinstance(v, dict) else v for k, v in PLOTLY.items()}
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base

LEAGUE_FLAGS = {
    "Premier League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "La Liga":        "🇪🇸",
    "Bundesliga":     "🇩🇪",
    "Serie A":        "🇮🇹",
    "Ligue 1":        "🇫🇷",
}
RADAR_BY_POS = {
    "ATT": (["goals_p90","npxg_p90","assists_p90","take_ons_p90","tkl_won_p90","pass_completion_pct"],
            ["Goals/90","npxG/90","Assists/90","Take-ons","Tkl won/90","Pass %"]),
    "MID": (["goals_p90","assists_p90","tkl_won_p90","interceptions_p90","take_ons_p90","npxg_p90"],
            ["Goals/90","Assists/90","Tkl won/90","Interceptions/90","Take-ons","npxG/90"]),
    "DEF": (["aerials_won_p90","interceptions_p90","tkl_won_p90","yellow_cards_p90","avg_min_per_game","assists_p90"],
            ["Aerials/90","Int/90","Tkl won/90","Y-cards/90","Min/game","Assists/90"]),
    "GK":  (["avg_min_per_game","pass_completion_pct","yellow_cards_p90","apps_pct_season","mv_momentum_12m","aerials_won_p90"],
            ["Min/game","Pass %","Y-cards/90","Starter rate","MV mom.","Aerials/90"]),
}

# ─── Club name cleaner ────────────────────────────────────────────
_CLUB_NAME_MAP = {
    # Premier League
    "Leeds United Association Football Club": "Leeds United",
    "Nottingham Forest Football Club": "Nottingham Forest",
    "Fulham Football Club": "Fulham",
    "Aston Villa Football Club": "Aston Villa",
    "Wolverhampton Wanderers Football Club": "Wolves",
    "Crystal Palace Football Club": "Crystal Palace",
    "Everton Football Club": "Everton",
    "Brighton and Hove Albion Football Club": "Brighton",
    "Tottenham Hotspur Football Club": "Tottenham",
    "Chelsea Football Club": "Chelsea",
    "Association Football Club Bournemouth": "Bournemouth",
    "Burnley Football Club": "Burnley",
    "Newcastle United Football Club": "Newcastle",
    "Arsenal Football Club": "Arsenal",
    "Liverpool Football Club": "Liverpool",
    "Brentford Football Club": "Brentford",
    "West Ham United Football Club": "West Ham",
    "Manchester City Football Club": "Manchester City",
    "Manchester United Football Club": "Manchester United",
    "Leicester City": "Leicester City",
    "Southampton FC": "Southampton",
    "Watford FC": "Watford",
    "West Bromwich Albion": "West Brom",
    "Sheffield United": "Sheffield United",
    "Luton Town": "Luton Town",
    "Ipswich Town": "Ipswich Town",
    "Sunderland Association Football Club": "Sunderland",
    "Swansea City": "Swansea City",
    "Stoke City": "Stoke City",
    "Queens Park Rangers": "QPR",
    "Cardiff City": "Cardiff City",
    "Huddersfield Town": "Huddersfield",
    "Norwich City": "Norwich City",
    # Bundesliga
    "1. Fußball- und Sportverein Mainz 05": "Mainz 05",
    "1. Fußballclub Heidenheim 1846": "Heidenheim",
    "1. Fußballclub Union Berlin": "Union Berlin",
    "Fußball-Club St. Pauli von 1910": "St. Pauli",
    "Fußball-Club Augsburg 1907": "FC Augsburg",
    "Turn- und Sportgemeinschaft 1899 Hoffenheim Fußball-Spielbetriebs": "Hoffenheim",
    "Sportverein Werder Bremen von 1899": "Werder Bremen",
    "Eintracht Frankfurt Fußball AG": "Eintracht Frankfurt",
    "Verein für Leibesübungen Wolfsburg": "Wolfsburg",
    "RasenBallsport Leipzig": "RB Leipzig",
    "Sport-Club Freiburg": "SC Freiburg",
    "Bayer 04 Leverkusen Fußball": "Bayer Leverkusen",
    "Verein für Bewegungsspiele Stuttgart 1893": "VfB Stuttgart",
    "Borussia Verein für Leibesübungen 1900 Mönchengladbach": "Borussia M'gladbach",
    "SpVgg Greuther Fürth": "Greuther Fürth",
    "FC Bayern München": "Bayern Munich",
    "Borussia Dortmund": "Borussia Dortmund",
    "VfL Bochum": "VfL Bochum",
    "SV Darmstadt 98": "Darmstadt 98",
    "Fortuna Düsseldorf": "Fortuna Düsseldorf",
    "Hannover 96": "Hannover 96",
    "Hertha BSC": "Hertha BSC",
    "SC Paderborn 07": "Paderborn",
    "FC Schalke 04": "Schalke 04",
    "Arminia Bielefeld": "Arminia Bielefeld",
    "Hamburger Sport Verein": "Hamburger SV",
    "1. Fußball-Club Köln": "FC Köln",
    "Holstein Kiel": "Holstein Kiel",
    # La Liga
    "Getafe Club de Fútbol S. A. D. Team Dubai": "Getafe",
    "Real Club Deportivo Mallorca S.A.D.": "Mallorca",
    "Rayo Vallecano de Madrid S. A. D.": "Rayo Vallecano",
    "Real Sociedad de Fútbol S.A.D.": "Real Sociedad",
    "Deportivo Alavés S. A. D.": "Alavés",
    "Elche Club de Fútbol S.A.D.": "Elche",
    "Club Atlético Osasuna": "Osasuna",
    "Levante Unión Deportiva S.A.D.": "Levante",
    "Reial Club Deportiu Espanyol de Barcelona S.A.D.": "Espanyol",
    "Real Club Celta de Vigo S. A. D.": "Celta Vigo",
    "Real Valladolid CF": "Valladolid",
    "Sevilla Fútbol Club S.A.D.": "Sevilla",
    "Valencia Club de Fútbol S. A. D.": "Valencia",
    "Villarreal Club de Fútbol S.A.D.": "Villarreal",
    "Club Atlético de Madrid S.A.D.": "Atlético Madrid",
    "Girona Fútbol Club S. A. D.": "Girona",
    "Real Betis Balompié S.A.D.": "Real Betis",
    "Futbol Club Barcelona": "FC Barcelona",
    "Real Madrid Club de Fútbol": "Real Madrid",
    "Athletic Club Bilbao": "Athletic Bilbao",
    "UD Almería": "Almería",
    "Cádiz CF": "Cádiz",
    "SD Eibar": "Eibar",
    "Granada CF": "Granada",
    "SD Huesca": "Huesca",
    "UD Las Palmas": "Las Palmas",
    "CD Leganés": "Leganés",
    # Serie A
    "Torino Calcio": "Torino",
    "Verona Hellas Football Club": "Hellas Verona",
    "Spezia Calcio": "Spezia",
    "Genoa Cricket and Football Club": "Genoa",
    "Cagliari Calcio": "Cagliari",
    "Parma Calcio 1913": "Parma",
    "Udinese Calcio": "Udinese",
    "Atalanta Bergamasca Calcio S.p.a.": "Atalanta",
    "Associazione Calcio Fiorentina": "Fiorentina",
    "Unione Sportiva Lecce": "Lecce",
    "Bologna Football Club 1909": "Bologna",
    "Associazione Sportiva Roma": "AS Roma",
    "Juventus Football Club": "Juventus",
    "Football Club Internazionale Milano S.p.A.": "Inter Milan",
    "Società Sportiva Lazio S.p.A.": "Lazio",
    "Unione Sportiva Sassuolo Calcio": "Sassuolo",
    "Unione Sportiva Cremonese S.p.A.": "Cremonese",
    "Società Sportiva Calcio Napoli": "Napoli",
    "Benevento Calcio": "Benevento",
    "Brescia Calcio": "Brescia",
    "Frosinone Calcio": "Frosinone",
    "US Salernitana 1919": "Salernitana",
    "Associazione Calcio Milan": "AC Milan",
    "AC Monza": "Monza",
    "UC Sampdoria": "Sampdoria",
    "SPAL": "SPAL",
    "Venezia FC": "Venezia",
    "FC Empoli": "Empoli",
    "Chievo Verona": "Chievo Verona",
    "FC Crotone": "Crotone",
    # Ligue 1
    "Toulouse Football Club": "Toulouse",
    "Association sportive de Monaco Football Club": "Monaco",
    "Football Club de Nantes": "Nantes",
    "FC Girondins Bordeaux": "Bordeaux",
    "Football Club de Metz": "Metz",
    "Racing Club de Strasbourg Alsace": "Strasbourg",
    "Racing Club de Lens": "Lens",
    "Le Havre Athletic Club": "Le Havre",
    "Thonon Évian Grand Genève FC": "Évian",
    "Lille Olympique Sporting Club": "Lille",
    "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille",
    "Stade brestois 29": "Brest",
    "Association de la Jeunesse auxerroise": "Auxerre",
    "Football Club Lorient-Bretagne Sud": "Lorient",
    "ESTAC Troyes": "Troyes",
    "Clermont Foot 63": "Clermont",
    "Paris Football Club": "Paris FC",
    "Paris Saint-Germain Football Club": "PSG",
    "Olympique Gymnaste Club Nice Côte d'Azur": "OGC Nice",
    "Stade Rennais Football Club": "Rennes",
    "AS Saint-Étienne": "Saint-Étienne",
    "Angers Sporting Club de l'Ouest": "Angers SCO",
    "Montpellier HSC": "Montpellier",
    "Stade Reims": "Stade Reims",
    "Nîmes Olympique": "Nîmes",
    "Valenciennes FC": "Valenciennes",
    "Amiens SC": "Amiens",
    "SM Caen": "Caen",
    "Dijon FCO": "Dijon",
    "EA Guingamp": "Guingamp",
    "AC Ajaccio": "Ajaccio",
}

# ─── Data ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data():
    return generate()

@st.cache_data(show_spinner="Building player embeddings…")
def load_embeddings(_pool):
    emb, _ = get_player_embeddings(_pool)
    return emb

def load_club_presets():
    real_clubs = pd.read_parquet("data/real_clubs.parquet")
    tm_clubs   = pd.read_parquet("data/cache/tm_clubs.parquet")
    merged = real_clubs.merge(tm_clubs[["club_id", "name"]], on="club_id", how="left")
    merged["display_name"] = merged["name"].map(_CLUB_NAME_MAP).fillna(merged["name"])
    merged = merged.sort_values(["league", "ppda"])
    return merged

data       = load_data()
pool       = data["players"]
clubs      = data["clubs"]
transfers  = data["transfers"]
embeddings = load_embeddings(pool)
club_presets_df = load_club_presets()

# Attach clean current-club names to player pool
_tm_clubs = pd.read_parquet("data/cache/tm_clubs.parquet")[["club_id", "name"]]
_club_id_to_name = (
    _tm_clubs.set_index("club_id")["name"]
    .map(lambda n: _CLUB_NAME_MAP.get(n, n))
)
pool = pool.copy()
pool["current_club"] = pool["current_club_id"].map(_club_id_to_name).fillna("—")

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Football Scout")

    with st.form("scout_form"):
        st.caption("PLAYER FILTERS")
        position    = st.selectbox("Position", ["ATT", "MID", "DEF", "GK"])
        c1, c2      = st.columns(2)
        min_age     = c1.number_input("Min age", 14, 40, 16)
        max_age     = c2.number_input("Max age", 14, 40, 30)
        budget      = st.slider("Max value (€M)", 1, 200, 100)
        min_minutes = st.slider("Min minutes played", 0, 3420, 500, step=90)
        _league_pills = st.pills(
            "Filter by league",
            ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"],
            selection_mode="multi",
            default=None,
            help="Leave empty to search all leagues. Select one or more to narrow the pool.",
        )
        source_leagues = _league_pills if _league_pills else None
        contract_filter = st.selectbox(
            "Contract status",
            ["Any", "Expiring ≤ 24 months", "Expiring ≤ 12 months", "Free / out of contract (≤ 6m)"],
        )
        _contract_map = {
            "Any": None,
            "Expiring ≤ 24 months": 24,
            "Expiring ≤ 12 months": 12,
            "Free / out of contract (≤ 6m)": 6,
        }
        show_injury_watch = st.checkbox(
            "Injury watch list",
            help="Re-score players with serious injury history as if injury-free and show them separately — good fits that carry medical risk.",
        )

        st.caption("DESTINATION CLUB")

        # ── Real club picker ──────────────────────────────────────
        LEAGUE_ORDER = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
        _club_options  = ["— Manual / style preset —"]
        _club_row_map  = {}
        for _lg in LEAGUE_ORDER:
            _sub = club_presets_df[club_presets_df["league"] == _lg]
            for _, _row in _sub.iterrows():
                _label = f"{_row['display_name']}  ({LEAGUE_FLAGS.get(_lg, _lg[:3])})"
                _club_options.append(_label)
                _club_row_map[_label] = _row

        selected_club_label = st.selectbox("Select a club", _club_options)
        _real_club = _club_row_map.get(selected_club_label)

        if _real_club is not None:
            ppda           = float(_real_club["ppda"])
            possession_pct = float(_real_club["possession_pct"])
            directness_idx = float(_real_club["directness_idx"])
            line_height_m  = float(_real_club["line_height_m"])
            st.caption(
                f"Tactical profile auto-filled from **{_real_club['display_name']}** "
                f"(avg 2018–2024) — PPDA {ppda:.1f} · Poss {possession_pct:.0f}% · "
                f"Directness {directness_idx:.2f} · Line {line_height_m:.0f}m"
            )
        else:
            PRESETS = {
                "Custom":                     None,
                "High possession / technical": dict(ppda=15.8, possession_pct=66, directness_idx=0.34, line_height_m=64),
                "Balanced pressing":           dict(ppda=15.2, possession_pct=54, directness_idx=0.38, line_height_m=58),
                "Physical / direct":           dict(ppda=13.5, possession_pct=46, directness_idx=0.48, line_height_m=50),
                "Low block / counter":         dict(ppda=13.0, possession_pct=41, directness_idx=0.55, line_height_m=43),
            }
            preset_name = st.selectbox("Style preset", list(PRESETS.keys()))
            preset      = PRESETS[preset_name]
            if preset:
                ppda, possession_pct          = preset["ppda"], preset["possession_pct"]
                directness_idx, line_height_m = preset["directness_idx"], preset["line_height_m"]
            else:
                ppda           = st.slider("Play style (↑ = more technical / possession)", 12.0, 16.5, 14.5, 0.1)
                possession_pct = st.slider("Possession %", 30, 70, 52)
                directness_idx = st.slider("Directness", 0.20, 0.65, 0.40, 0.01)
                line_height_m  = st.slider("Line height (m)", 25, 65, 48)

        press_val = round((ppda - 12.0) / 4.5 * 100)   # 12 (physical) → 16.5 (technical)
        poss_val  = round((possession_pct - 30) / 40 * 100)
        dir_val   = round((directness_idx - 0.20) / 0.45 * 100)
        line_val  = round((line_height_m - 25) / 40 * 100)

        dna_labels        = ["Technical", "Possession", "Directness", "Line height"]
        dna_vals          = [press_val, poss_val, dir_val, line_val]
        dna_vals_closed   = dna_vals + [dna_vals[0]]
        dna_labels_closed = dna_labels + [dna_labels[0]]

        fig_dna = go.Figure(go.Scatterpolar(
            r=dna_vals_closed,
            theta=dna_labels_closed,
            fill="toself",
            fillcolor="rgba(37,99,235,0.15)",
            line_color="#2563eb",
            line_width=2,
        ))
        fig_dna.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="#e2e8f0"),
                angularaxis=dict(tickfont=dict(size=10, color="#374151"), gridcolor="#e2e8f0"),
                bgcolor="#ffffff",
            ),
            margin=dict(l=10, r=10, t=0, b=0),
            height=160,
            paper_bgcolor="#f8fafc",
        )
        st.plotly_chart(fig_dna, use_container_width=True)

        league_options = sorted(pool["league"].dropna().unique().tolist())
        if _real_club is not None:
            _auto_lg   = str(_real_club["league"])
            _lg_idx    = league_options.index(_auto_lg) if _auto_lg in league_options else 0
            dest_league = st.selectbox("Destination league", league_options, index=_lg_idx)
        else:
            dest_league = st.selectbox("Destination league", league_options)
        top_n          = st.slider("Show top N", 5, 30, 10)

        st.form_submit_button("Scout", use_container_width=True, type="primary")

# ─── Shared scoring ───────────────────────────────────────────────
filters     = Filters(position=position, max_value_eur_m=float(budget),
                      min_age=int(min_age), max_age=int(max_age), min_minutes=int(min_minutes),
                      max_contract_months=_contract_map[contract_filter],
                      leagues=source_leagues)
destination = DestinationClub(ppda=ppda, possession_pct=possession_pct,
                              directness_idx=directness_idx, line_height_m=line_height_m,
                              league=dest_league)
candidates  = apply_filters(pool, filters)

if len(candidates) > 0:
    if show_injury_watch:
        # Zero out injury penalty for all candidates so injured players
        # compete on talent alone; restore original flags for display.
        _for_scoring = candidates.copy()
        _inj_mask = _for_scoring["has_serious_injury"] == 1.0
        _for_scoring.loc[_inj_mask, "has_serious_injury"]  = 0.0
        _for_scoring.loc[_inj_mask, "injury_days_last_2y"] = 0.0
        ranked = compute_fit_score(_for_scoring, destination)
        # Re-attach original injury flag so cards / expander can show the warning
        _orig_flags = candidates.set_index("player_name")["has_serious_injury"]
        ranked["has_serious_injury"] = ranked["player_name"].map(_orig_flags).fillna(0.0)
        _orig_days = candidates.set_index("player_name")["injury_days_last_2y"]
        ranked["injury_days_last_2y"] = ranked["player_name"].map(_orig_days).fillna(0.0)
    else:
        ranked = compute_fit_score(candidates, destination)
else:
    ranked = candidates

top = ranked.head(top_n).copy()
RADAR_COLS, RADAR_LABELS = RADAR_BY_POS.get(position, RADAR_BY_POS["ATT"])

# ─── Tabs ─────────────────────────────────────────────────────────
tab_scout, tab_similar, tab_data, tab_model = st.tabs([
    "Scout", "Similar Players", "Data Explorer", "Model",
])


# ═══════════════════════════════════════════════════
# TAB 1 — SCOUT
# ═══════════════════════════════════════════════════
with tab_scout:
    if len(candidates) == 0:
        st.warning("No players match these filters — try relaxing age, budget or minutes.")
        st.stop()

    if _real_club is not None:
        style_label = str(_real_club["display_name"])
    else:
        style_label = preset_name if preset_name != "Custom" else "Custom style"

    # ── Report header ─────────────────────────────────────────
    st.markdown(f"### Scouting Report — {position} · {style_label}")
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Candidates screened", f"{len(candidates):,}")
    hc2.metric("Position", position)
    hc3.metric("Budget ceiling", f"€{budget}M")
    hc4.metric("Age window", f"{min_age} – {max_age}")
    st.divider()

    # ── Display columns ────────────────────────────────────────
    display_cols = {
        "player_name":"Player","age":"Age","league":"League",
        "market_value_eur_m":"Value (€M)","contract_months_remaining":"Contract (m)",
        "goals_p90":"Goals/90",
        "assists_p90":"Assists/90","yellow_cards_p90":"Y-cards/90",
        "avg_min_per_game":"Min/game","pass_completion_pct":"Pass %",
        "mv_momentum_12m":"MV mom.","fit_score":"Fit",
    }
    tbl = top[[c for c in display_cols if c in top.columns]].rename(columns=display_cols)
    tbl.index = range(1, len(tbl) + 1)

    # ── Top 3 recommended targets ─────────────────────────────
    n_cards = min(3, len(top))
    if n_cards:
        st.markdown("##### Recommended targets")
        card_cols = st.columns(n_cards)
        for i in range(n_cards):
            row        = top.iloc[i]
            flag       = LEAGUE_FLAGS.get(str(row.get("league", "")), "")
            fit        = row["fit_score"]
            age        = int(row["age"]) if pd.notna(row.get("age")) else "—"
            val        = row["market_value_eur_m"]
            club       = row.get("current_club", "—")
            g90        = row.get("goals_p90", 0)
            a90        = row.get("assists_p90", 0)
            npxg       = row.get("npxg_p90", 0)
            min_pg     = row.get("avg_min_per_game", 0)
            contract_m = row.get("contract_months_remaining", None)
            mv_mom     = row.get("mv_momentum_12m", 0)
            is_injury  = row.get("has_serious_injury", 0) == 1.0

            if pd.notna(contract_m) and contract_m is not None:
                contract_str = f"{contract_m:.0f} mo"
            else:
                contract_str = "—"

            with card_cols[i]:
                with st.container(border=True):
                    # Rank + fit
                    rc, fc = st.columns([1, 1])
                    rc.caption(f"#{i + 1}{'  · Top pick' if i == 0 else ''}")
                    fc.markdown(
                        f"<p style='text-align:right;margin:0;font-size:1.2rem;"
                        f"font-weight:700;color:#2563eb;line-height:1.3'>{fit:.1f}"
                        f"<span style='font-size:0.65rem;color:#94a3b8;font-weight:400'> / 100</span></p>",
                        unsafe_allow_html=True,
                    )
                    # Name + club
                    st.markdown(f"#### {row['player_name']}")
                    st.caption(f"{flag} {club}  ·  Age {age}  ·  €{val:.0f}M")

                    # Single key stat
                    st.markdown(f"**{g90:.2f}** Goals / 90")

                    # Details popover
                    with st.popover("Details", use_container_width=True):
                        st.markdown(f"**{row['player_name']}** — {flag} {club}")
                        st.caption(f"{row.get('league','—')}  ·  Age {age}  ·  €{val:.0f}M  ·  Contract {contract_str}")
                        st.divider()

                        if pd.notna(contract_m) and contract_m is not None:
                            if contract_m <= 12:
                                st.error(f"Contract expiring — **{contract_m:.0f} months** remaining")
                            elif contract_m <= 24:
                                st.warning(f"Contract running down — **{contract_m:.0f} months** remaining")
                            else:
                                st.success(f"Contract secure — **{contract_m:.0f} months** remaining")

                        if is_injury:
                            inj_days = int(row.get("injury_days_last_2y", 0))
                            st.warning(
                                f"Serious injury history — {inj_days} days missed in last 2 seasons.  \n"
                                + ("Fit score shown without penalty. Medical assessment required."
                                   if show_injury_watch else
                                   "Enable Injury watch list to see unpenalised ranking.")
                            )

                        st.divider()
                        pa, pb = st.columns(2)
                        pa.metric("Goals / 90",   f"{g90:.2f}")
                        pb.metric("Assists / 90", f"{a90:.2f}")
                        pa.metric("npxG / 90",    f"{npxg:.2f}")
                        pb.metric("Min / game",   f"{min_pg:.0f}")
                        pa.metric("Pass %",       f"{row.get('pass_completion_pct', 0):.0f}%")
                        pb.metric("MV trend",     f"{mv_mom:+.2f}")
        st.divider()

    # ── Full-width ranking table ──────────────────────────────
    st.markdown("##### Shortlist ranking")
    rank_cols = {
        "player_name": "Player", "current_club": "Current Club",
        "fit_score": "Fit ↓", "league": "League", "age": "Age",
        "market_value_eur_m": "Value (€M)", "goals_p90": "G/90",
        "assists_p90": "A/90", "pass_completion_pct": "Pass %",
        "contract_months_remaining": "Contract (m)", "mv_momentum_12m": "MV trend",
    }
    rank_tbl = top[[c for c in rank_cols if c in top.columns]].rename(columns=rank_cols)
    _inj_flags = top["has_serious_injury"].values if "has_serious_injury" in top.columns else np.zeros(len(top))
    rank_tbl.index = range(1, len(rank_tbl) + 1)

    def _highlight_injury_rows(row):
        idx = row.name - 1
        if show_injury_watch and idx < len(_inj_flags) and _inj_flags[idx] == 1.0:
            return ["background-color: #fff7ed; color: #9a3412"] * len(row)
        return [""] * len(row)

    _fmt = {
        "Value (€M)":    "€{:.0f}M",
        "G/90":          "{:.2f}",
        "A/90":          "{:.2f}",
        "Pass %":        "{:.0f}%",
        "MV trend":      "{:+.2f}",
        "Fit ↓":         "{:.1f}",
        "Contract (m)":  "{:.0f}",
    }
    st.dataframe(
        rank_tbl.style
                .apply(_highlight_injury_rows, axis=1)
                .background_gradient(subset=["Fit ↓"], cmap="Blues")
                .format({k: v for k, v in _fmt.items() if k in rank_tbl.columns}),
        use_container_width=True,
        height=min(len(rank_tbl) * 38 + 44, 560),
    )
    _dl_row, _leg_row = st.columns([3, 7])
    with _dl_row:
        csv = tbl.to_csv(index=True).encode("utf-8")
        st.download_button(
            "Export shortlist CSV", csv,
            file_name=f"shortlist_{position}_{style_label.replace(' ','_')}.csv",
            mime="text/csv",
        )
    with _leg_row:
        if show_injury_watch and any(_inj_flags == 1.0):
            st.caption("🟠 Orange rows = serious injury history — fit shown without penalty")

    st.divider()

    # ── Player profile radar ───────────────────────────────────
    st.markdown("##### Player profiles — top 5")
    st.caption("Percentile vs Big-5 peers at the same position. Hover a player name in the legend to isolate their trace.")

    _, radar_col, _ = st.columns([1, 8, 1])
    with radar_col:
        RADAR_COLORS = ["#2563eb","#059669","#d97706","#dc2626","#7c3aed"]
        fig_radar = go.Figure()
        for i, (_, row) in enumerate(top.head(5).iterrows()):
            vals = [
                min(float(row.get(c, 0)) / max(float(pool[c].quantile(0.95)), 1e-6) * 100, 100)
                for c in RADAR_COLS
            ]
            vals.append(vals[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=RADAR_LABELS + [RADAR_LABELS[0]],
                fill="toself",
                fillcolor=f"rgba({','.join(str(int(c,16)) for c in [RADAR_COLORS[i][1:3], RADAR_COLORS[i][3:5], RADAR_COLORS[i][5:7]])},{ 0.12 if i == 0 else 0.05 })",
                name=row["player_name"],
                line_color=RADAR_COLORS[i],
                line_width=2.5 if i == 0 else 1.5,
            ))
        fig_radar.update_layout(
            **pl(margin=dict(l=20, r=20, t=20, b=90)),
            polar=POLAR_STYLE,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="top", y=-0.12,
                xanchor="center", x=0.5,
                font=dict(size=11, color="#374151"),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=440,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Why these players fit — collapsible ───────────────────
    with st.expander("Why these players fit — top 5 breakdown"):

        def _pctile(col, val_):
            s = candidates[col].dropna()
            return float((s < val_).mean() * 100) if len(s) > 0 else 50.0

        for i, (_, row) in enumerate(top.head(5).iterrows()):
            name       = row["player_name"]
            fit        = row["fit_score"]
            club       = row.get("current_club", "—")
            age        = int(row["age"]) if pd.notna(row.get("age")) else None
            val        = float(row.get("market_value_eur_m", 0))
            contract_m = row.get("contract_months_remaining", None)
            mv_mom     = float(row.get("mv_momentum_12m", 0))
            is_injury  = row.get("has_serious_injury", 0) == 1.0

            g90      = float(row.get("goals_p90", 0))
            a90      = float(row.get("assists_p90", 0))
            npxg     = float(row.get("npxg_p90", 0))
            mins_pg  = float(row.get("avg_min_per_game", 0))
            pass_cmp = float(row.get("pass_completion_pct", 0))
            carries  = float(row.get("prog_carries_p90", 0))
            tkl      = float(row.get("tkl_won_p90", 0))

            g90_pct   = _pctile("goals_p90", g90)
            a90_pct   = _pctile("assists_p90", a90)
            npxg_pct  = _pctile("npxg_p90", npxg)
            mins_pct  = _pctile("avg_min_per_game", mins_pg)
            pass_pct  = _pctile("pass_completion_pct", pass_cmp)
            carry_pct = _pctile("prog_carries_p90", carries)
            tkl_pct   = _pctile("tkl_won_p90", tkl)

            positives = []
            concerns  = []

            # Quality
            if g90_pct >= 70:
                positives.append(f"Goals/90 in **{g90_pct:.0f}th percentile** among {position} candidates — {g90:.2f}/90")
            if a90_pct >= 70:
                positives.append(f"Assists/90 in **{a90_pct:.0f}th percentile** — creative output to complement goal threat")
            if npxg_pct >= 65:
                positives.append(f"npxG/90 in **{npxg_pct:.0f}th percentile** — underlying chance quality validates the output")
            if carry_pct >= 65:
                positives.append(f"Progressive carries in **{carry_pct:.0f}th percentile** — advances play through the lines")
            if mins_pct >= 70:
                positives.append(f"Averages **{mins_pg:.0f} min/game** — reliable starter, durable over a full season")
            if mv_mom >= 0.10:
                positives.append(f"Market value up **{mv_mom*100:.0f}%** in 12 months — in-form and attracting interest")

            # Tactical fit
            if possession_pct >= 55 and pass_pct >= 60:
                positives.append(f"Pass completion in **{pass_pct:.0f}th percentile** — fits {possession_pct:.0f}% possession system")
            if ppda <= 10 and tkl_pct >= 60:
                positives.append(f"Tackles won in **{tkl_pct:.0f}th percentile** — pressing work rate matches high-press demand (PPDA {ppda:.1f})")
            if directness_idx >= 0.46 and carry_pct >= 55:
                positives.append(f"Carry numbers suit a direct, transition-focused system")

            # Age / value
            if age is not None and age <= 23:
                positives.append(f"Age **{age}** — significant development upside, value likely to appreciate")
            elif age is not None and age <= 26:
                positives.append(f"Age **{age}** — peak-prime window; can deliver returns immediately")

            if pd.notna(contract_m) and contract_m is not None:
                if contract_m <= 12:
                    positives.append(f"Contract expires in **{contract_m:.0f} months** — potential cut-price deal, strong negotiating position")
                elif contract_m <= 24:
                    positives.append(f"Contract running down (**{contract_m:.0f} months**) — seller may accept below market fee")

            if val > 0 and val <= budget * 0.55:
                positives.append(f"€{val:.0f}M — **€{budget - val:.0f}M under budget ceiling**, preserving funds for depth")

            # Concerns
            if is_injury:
                inj_days = int(row.get("injury_days_last_2y", 0))
                concerns.append(
                    f"Serious injury history — **{inj_days} days** missed last 2 seasons"
                    + (" · shown without penalty (injury watch mode)" if show_injury_watch else "")
                )
            if age is not None and age >= 29:
                concerns.append(f"Age **{age}** — limited resale upside, narrow window to maximise the investment")
            if mv_mom <= -0.15:
                concerns.append(f"Market value down **{abs(mv_mom)*100:.0f}%** in 12 months — investigate form or fitness")
            if mins_pct <= 30:
                concerns.append(f"Low minutes per game (**{mins_pg:.0f}**) — rotation player or availability concerns")
            if pd.notna(contract_m) and contract_m is not None and contract_m >= 42:
                concerns.append(f"Contract runs **{contract_m:.0f} more months** — seller holds full leverage on fee")

            if not positives:
                positives.append("Fit driven by age-value profile and tactical overlap with the destination style")

            lc, rc = st.columns([4, 1])
            lc.markdown(f"**#{i+1} {name}** — {club}")
            rc.markdown(
                f"<p style='text-align:right;font-weight:700;color:#2563eb;margin:0'>Fit {fit:.1f}</p>",
                unsafe_allow_html=True,
            )
            for s in positives:
                st.caption(f"  ✓  {s}")
            for s in concerns:
                st.caption(f"  ⚠  {s}")
            if i < 4:
                st.divider()

    st.divider()

    # ── Market intelligence — value for money ─────────────────────
    st.markdown("##### Market intelligence — value for money")
    st.caption("Fit score per €M spent. Green = affordable, orange = expensive. Best deals at the top.")

    _eff = top.copy()
    _eff["_score_per_m"] = (_eff["fit_score"] / _eff["market_value_eur_m"].clip(lower=1)).round(3)
    _eff = _eff.sort_values("_score_per_m", ascending=True)
    _eff["_surname"] = _eff["player_name"].str.split().str[-1]

    fig_eff = go.Figure(go.Bar(
        y=_eff["_surname"],
        x=_eff["_score_per_m"],
        orientation="h",
        marker=dict(
            color=_eff["market_value_eur_m"],
            colorscale=[[0, "#10b981"], [0.5, "#3b82f6"], [1, "#f59e0b"]],
            showscale=True,
            colorbar=dict(
                title=dict(text="€M", font=dict(size=11, color="#374151")),
                thickness=12, len=0.6,
                tickfont=dict(size=10, color="#374151"),
            ),
            line=dict(width=0),
        ),
        text=_eff.apply(
            lambda r: f"Fit {r['fit_score']:.1f} · €{r['market_value_eur_m']:.0f}M", axis=1
        ),
        textposition="outside",
        textfont=dict(size=10, color="#374151"),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Fit score: %{customdata[1]:.1f}<br>"
            "Market value: €%{customdata[2]:.0f}M<br>"
            "Efficiency: %{x:.2f} fit/M€"
            "<extra></extra>"
        ),
        customdata=_eff[["player_name", "fit_score", "market_value_eur_m"]].values,
    ))
    fig_eff.update_layout(
        **pl(
            xaxis=dict(title="Fit score / €M spent (higher = better deal)", gridcolor="#e2e8f0"),
            yaxis=dict(title=""),
            margin=dict(l=10, r=160, t=10, b=30),
        ),
        height=max(300, len(top) * 36 + 60),
        showlegend=False,
    )
    st.plotly_chart(fig_eff, use_container_width=True)

    if show_injury_watch:
        n_inj = int((top["has_serious_injury"] == 1.0).sum())
        if n_inj:
            st.info(
                f"**Injury watch mode active** — {n_inj} player(s) in this shortlist have a serious injury history. "
                "Fit scores are shown without the injury penalty. Open Player details on their card for the full medical flag."
            )

    with st.expander("Club style guide"):
        st.markdown("""
| Parameter | Low | High |
|---|---|---|
| **PPDA** | 4–8 → intense high press | 14–20 → passive / low-block |
| **Possession %** | 35–42 → counter-attack | 60–70 → tiki-taka |
| **Directness** | 0.30–0.38 → patient build-up | 0.55–0.65 → direct / long-ball |
| **Line height** | 25–35 m → deep defensive block | 55–65 m → high press line |
        """)


# ═══════════════════════════════════════════════════
# TAB 2 — SIMILAR PLAYERS
# ═══════════════════════════════════════════════════
with tab_similar:
    st.subheader("Find Similar Players")
    st.caption("Similarity is computed in the 32-dimensional player embedding space learned by the model — it reflects playing style, not just raw stats.")

    c_in, c_n = st.columns([4, 1])
    search_name = c_in.text_input("Search player", placeholder="e.g. Pedri, Saka, Kane, Yamal…")
    n_similar   = c_n.number_input("Results", 5, 20, 10)

    if not search_name:
        st.info("Try: Pedri · Saka · Musiala · Lamine Yamal · Haaland · Bellingham · Mbappé")
    else:
        matches = pool[pool["player_name"].str.contains(search_name, case=False, na=False)]
        if matches.empty:
            st.warning(f"No player found for '{search_name}'.")
        else:
            if len(matches) > 1:
                chosen  = st.selectbox("Multiple matches:", matches["player_name"].tolist())
                matches = matches[matches["player_name"] == chosen]

            target     = matches.iloc[0]
            target_idx = matches.index[0]

            # Target info
            flag = LEAGUE_FLAGS.get(str(target.get("league", "")), "🌍")
            st.markdown(f"""
**{flag} {target['player_name']}** &nbsp;·&nbsp;
{target.get('position_group','—')} &nbsp;·&nbsp;
Age {int(target['age']) if pd.notna(target.get('age')) else '—'} &nbsp;·&nbsp;
€{target['market_value_eur_m']:.0f}M &nbsp;·&nbsp;
G/90: {target.get('goals_p90',0):.2f} &nbsp;·&nbsp;
A/90: {target.get('assists_p90',0):.2f}
            """)
            st.divider()

            # Cosine similarity
            t_emb = embeddings[target_idx]
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-8)
            sims  = (embeddings / norms) @ (t_emb / np.linalg.norm(t_emb).clip(1e-8))
            sims[target_idx] = -1
            top_idx = np.argsort(sims)[::-1][:n_similar]
            similar = pool.iloc[top_idx].copy()
            similar["Similarity %"] = (sims[top_idx] * 100).round(1)

            # ── Full-width similarity table ───────────────────────
            st.markdown("##### Most similar players")
            sim_cols = {
                "player_name": "Player", "current_club": "Current Club",
                "league": "League", "age": "Age",
                "market_value_eur_m": "Value (€M)", "goals_p90": "Goals/90",
                "assists_p90": "Assists/90", "Similarity %": "Sim %",
            }
            sim_tbl = similar[[c for c in sim_cols if c in similar.columns]].rename(columns=sim_cols)
            sim_tbl.index = range(1, len(sim_tbl) + 1)
            st.dataframe(
                sim_tbl.style
                       .background_gradient(subset=["Sim %"], cmap="Blues")
                       .format({"Value (€M)": "€{:.0f}M", "Goals/90": "{:.2f}",
                                "Assists/90": "{:.2f}", "Sim %": "{:.1f}%"}),
                use_container_width=True,
                height=min((n_similar + 1) * 38, 520),
            )

            # Budget alternatives
            cheaper = similar[similar["market_value_eur_m"] < target["market_value_eur_m"] * 0.6].head(3)
            if not cheaper.empty:
                st.markdown("##### Budget alternatives")
                for _, row in cheaper.iterrows():
                    saving = target["market_value_eur_m"] - row["market_value_eur_m"]
                    f2 = LEAGUE_FLAGS.get(str(row.get("league", "")), "")
                    st.success(
                        f"**{row['player_name']}** · {f2} {row.get('league','—')} · "
                        f"€{row['market_value_eur_m']:.0f}M · "
                        f"{float(row['Similarity %']):.0f}% similar · "
                        f"saves ~€{saving:.0f}M"
                    )

            st.divider()

            # ── Centred radar ─────────────────────────────────────
            st.markdown(f"##### Player profiles — {target['player_name']} vs top 3 similar")
            st.caption("Percentile vs Big-5 peers at the same position. ★ = target player.")

            _, cmp_col, _ = st.columns([1, 8, 1])
            with cmp_col:
                fig_cmp  = go.Figure()
                all_p    = pd.concat([target.to_frame().T, similar.head(3)], ignore_index=True)
                cmp_cols = ["#f59e0b","#10b981","#38bdf8","#f87171"]
                for i, (_, row) in enumerate(all_p.iterrows()):
                    vals = []
                    for c in RADAR_COLS:
                        v   = float(row.get(c, 0))
                        q95 = float(pool[c].quantile(0.95))
                        vals.append(min(v / max(q95, 1e-6) * 100, 100))
                    vals.append(vals[0])
                    lbl = str(row["player_name"]) + (" ★" if i == 0 else "")
                    col = cmp_cols[i]
                    r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
                    fig_cmp.add_trace(go.Scatterpolar(
                        r=vals, theta=RADAR_LABELS + [RADAR_LABELS[0]],
                        fill="toself",
                        fillcolor=f"rgba({r},{g},{b},{0.12 if i == 0 else 0.05})",
                        name=lbl,
                        line_color=col, line_width=3 if i == 0 else 1.5,
                    ))
                fig_cmp.update_layout(
                    **pl(margin=dict(l=20, r=20, t=20, b=90)),
                    polar=POLAR_STYLE,
                    showlegend=True,
                    legend=dict(
                        orientation="h", yanchor="top", y=-0.12,
                        xanchor="center", x=0.5,
                        font=dict(size=11, color="#374151"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    height=440,
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

            # Comparable targets — similarity vs value
            st.divider()
            st.markdown("##### Comparable targets — similarity vs value")
            st.caption("Bubble size = goals / 90. Gold star = target player. Best alternatives: top-left (high similarity, lower cost).")

            _sim_plot = similar.copy()
            _sim_plot["_size"] = (_sim_plot["goals_p90"].clip(0, 1.5) + 0.2) * 22
            _sim_plot["_surname"] = _sim_plot["player_name"].str.split().str[-1]

            LEAGUE_COLORS = {
                "Premier League": "#3b82f6",
                "La Liga":        "#ef4444",
                "Bundesliga":     "#f59e0b",
                "Serie A":        "#10b981",
                "Ligue 1":        "#8b5cf6",
            }

            fig_vs = go.Figure()
            for lg, grp in _sim_plot.groupby("league", sort=False):
                fig_vs.add_trace(go.Scatter(
                    x=grp["market_value_eur_m"],
                    y=grp["Similarity %"],
                    mode="markers+text",
                    name=str(lg),
                    marker=dict(
                        size=grp["_size"],
                        color=LEAGUE_COLORS.get(str(lg), "#94a3b8"),
                        opacity=0.75,
                        line=dict(width=1, color="#ffffff"),
                    ),
                    text=grp["_surname"],
                    textposition="top center",
                    textfont=dict(size=9, color="#374151"),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Similarity: %{y:.1f}%<br>"
                        "Value: €%{x:.0f}M<br>"
                        "Age: %{customdata[1]}  ·  Goals/90: %{customdata[2]:.2f}"
                        "<extra></extra>"
                    ),
                    customdata=grp[["player_name", "age", "goals_p90"]].values,
                ))

            # Target player marker
            t_val = float(target.get("market_value_eur_m", 0))
            if pd.notna(t_val) and t_val > 0:
                fig_vs.add_trace(go.Scatter(
                    x=[t_val], y=[100],
                    mode="markers+text",
                    name=str(target["player_name"]),
                    marker=dict(symbol="star", size=18, color="#f59e0b",
                                line=dict(width=1.5, color="#92400e")),
                    text=[target["player_name"].split()[-1] + " ★"],
                    textposition="top center",
                    textfont=dict(size=10, color="#92400e"),
                    hovertemplate=f"<b>{target['player_name']}</b> (target)<br>Value: €{t_val:.0f}M<extra></extra>",
                    showlegend=True,
                ))
                fig_vs.add_vline(
                    x=t_val, line_dash="dot", line_color="#f59e0b", line_width=1.5,
                    annotation_text="Target price", annotation_font_color="#92400e",
                    annotation_font_size=10,
                )

            fig_vs.update_layout(
                **pl(
                    xaxis=dict(title="Market value (€M)"),
                    yaxis=dict(title="Similarity %", range=[
                        max(0, float(_sim_plot["Similarity %"].min()) - 3),
                        102,
                    ]),
                    margin=dict(l=10, r=20, t=20, b=10),
                ),
                height=360,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0,
                    font=dict(size=10, color="#374151"),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            st.plotly_chart(fig_vs, use_container_width=True)


# ═══════════════════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ═══════════════════════════════════════════════════
with tab_data:
    st.subheader("Data Explorer")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players",            f"{len(pool):,}",      "Big-5 · 2024-25")
    c2.metric("Clubs",              f"{len(clubs):,}",     "with tactical data")
    c3.metric("Labelled transfers", f"{len(transfers):,}", "2018–2024")
    c4.metric("Avg success score",  f"{transfers['success_score'].mean():.2f}", "0 – 1 scale")

    st.divider()

    st.markdown("**Value vs Performance**")
    perf = st.selectbox("Y-axis", ["goals_p90","assists_p90","npxg_p90","pass_completion_pct",
                                    "prog_carries_p90","mv_momentum_12m","avg_min_per_game"])
    fig_sc = px.scatter(pool, x="market_value_eur_m", y=perf, color="position_group",
                        hover_name="player_name",
                        hover_data={"age":True,"league":True,"market_value_eur_m":":.0f"},
                        opacity=0.65, color_discrete_sequence=px.colors.qualitative.Set2,
                        labels={"market_value_eur_m":"Market value (€M)"})
    fig_sc.update_layout(**PLOTLY, height=360,
                         legend=dict(font=dict(color="#374151"), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()
    st.markdown("**Stat distributions by position**")
    cl, cr = st.columns(2)
    with cl:
        stat = st.selectbox("Stat", ["goals_p90","assists_p90","yellow_cards_p90","avg_min_per_game",
                                      "apps_pct_season","tkl_won_p90","interceptions_p90",
                                      "mv_momentum_12m","pass_completion_pct","market_value_eur_m"])
        fig_h = px.histogram(pool, x=stat, color="position_group", barmode="overlay",
                             nbins=40, opacity=0.7,
                             color_discrete_sequence=px.colors.qualitative.Set2)
        fig_h.update_layout(**PLOTLY, height=280, showlegend=False)
        st.plotly_chart(fig_h, use_container_width=True)
    with cr:
        fig_b = px.box(pool, x="position_group", y=stat, color="position_group",
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig_b.update_layout(**PLOTLY, height=280, showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)

    st.divider()
    st.markdown("**Club tactical space**")
    c1, c2 = st.columns(2)
    for col, x, y, xl, yl in [
        (c1,"ppda","possession_pct","PPDA (↓ = more pressing)","Possession %"),
        (c2,"directness_idx","line_height_m","Directness","Line height (m)"),
    ]:
        fig = px.scatter(clubs, x=x, y=y, color="league", hover_data=["club_id"],
                         labels={x:xl,y:yl},
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(**PLOTLY, height=320,
                          legend=dict(font=dict(color="#374151"), bgcolor="rgba(0,0,0,0)"))
        with col:
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Transfer outcome labels**")
    c1, c2 = st.columns(2)
    with c1:
        fig5 = px.histogram(transfers, x="success_score", nbins=40,
                            labels={"success_score":"Success score"},
                            color_discrete_sequence=["#10b981"])
        fig5.add_vline(x=transfers["success_score"].mean(), line_dash="dash", line_color="#f59e0b",
                       annotation_text=f"mean={transfers['success_score'].mean():.2f}",
                       annotation_font_color="#f59e0b")
        fig5.update_layout(**PLOTLY, height=260)
        st.plotly_chart(fig5, use_container_width=True)
    with c2:
        fig6 = px.histogram(transfers, x="minutes_share_y1", nbins=40,
                            labels={"minutes_share_y1":"Minutes share yr 1"},
                            color_discrete_sequence=["#38bdf8"])
        fig6.update_layout(**PLOTLY, height=260)
        st.plotly_chart(fig6, use_container_width=True)


# ═══════════════════════════════════════════════════
# TAB 4 — MODEL
# ═══════════════════════════════════════════════════
with tab_model:
    st.subheader("Model Performance")

    try:
        with open("models/metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        st.error("models/metrics.json not found — run `python train.py` first.")
        st.stop()

    with st.expander("Architecture", expanded=True):
        st.code("""
Player stats (15 feats)            Club tactics (9 feats)
        │                                   │
  Player Tower                        Club Tower
  15 → 64 → 32-d embedding           9 → 32 → 32-d embedding
        │                                   │
        └──────────── Head MLP ─────────────┘
           input: concat( p⊙c,  |p−c|,  ctx )   = 65 feats
           layers: 65 → 64 → 64 → 1 → sigmoid
                             │
                   Transfer success score  (0 – 1)
        """, language=None)
        st.caption("The player tower's 32-d output is also the embedding used in the Similar Players tab.")

    rows = []
    for name in ["cosine","linear","gbm"]:
        m = metrics["baselines"][name]
        rows.append({"Model":name.capitalize(),"R²":m["test_r2"],"MAE":m["test_mae"],"Spearman ρ":m["test_spearman"]})
    tt = metrics["two_tower"]
    rows.append({"Model":"Two-tower","R²":tt["test_r2"],"MAE":tt["test_mae"],"Spearman ρ":tt["test_spearman"]})
    df_m = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    for col, y_col, title in [(c1,"R²","R² — higher is better"),(c2,"Spearman ρ","Spearman ρ — ranking ability")]:
        fig = px.bar(df_m, x="Model", y=y_col, color="Model", text=y_col, title=title,
                     color_discrete_sequence=["#475569","#3b82f6","#10b981","#f59e0b"])
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
        fig.update_layout(**PLOTLY, showlegend=False, height=320,
                          yaxis_range=[0, df_m[y_col].max() * 1.2], title_font_size=14)
        with col:
            st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df_m.set_index("Model").style.format("{:.3f}").background_gradient(cmap="Greens"),
        use_container_width=True,
    )

    st.divider()
    tt_r2  = metrics["two_tower"]["test_r2"]
    tt_mae = metrics["two_tower"]["test_mae"]
    tt_sp  = metrics["two_tower"]["test_spearman"]
    gbm_r2 = metrics["baselines"]["gbm"]["test_r2"]
    st.markdown(f"""
**How to read the numbers**

| Metric | Value | What it means |
|---|---|---|
| **R²** | {tt_r2:.2f} | {tt_r2*100:.0f}% of variance in real transfer outcomes explained |
| **MAE** | {tt_mae:.2f} | Off by ~{tt_mae:.2f} points on a 0–1 scale on average |
| **Spearman ρ** | {tt_sp:.2f} | Ranking ability — how well the model orders players by fit |

{"**Two-tower vs GBM:** The separate encoder architecture captures player × club interaction that a flat feature vector misses." if tt_r2 > gbm_r2 else f"**Note:** GBM currently outperforms the two-tower (R² {gbm_r2:.2f} vs {tt_r2:.2f}). The two-tower needs more training data to earn its structural advantage — but its 32-d player embeddings still power the Similar Players feature."}
    """)
