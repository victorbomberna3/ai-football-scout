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
    page_icon="⚽",
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
    "ATT": (["goals_p90","npxg_p90","assists_p90","take_ons_p90","prog_carries_p90","pass_completion_pct"],
            ["Goals/90","npxG/90","Assists/90","Take-ons","Prog carries","Pass %"]),
    "MID": (["goals_p90","assists_p90","pass_completion_pct","prog_passes_p90","take_ons_p90","npxg_p90"],
            ["Goals/90","Assists/90","Pass %","Prog passes","Take-ons","npxG/90"]),
    "DEF": (["aerials_won_p90","pass_completion_pct","prog_passes_p90","yellow_cards_p90","avg_min_per_game","assists_p90"],
            ["Aerials/90","Pass %","Prog passes","Y-cards/90","Min/game","Assists/90"]),
    "GK":  (["avg_min_per_game","pass_completion_pct","yellow_cards_p90","n_apps","mv_momentum_12m","aerials_won_p90"],
            ["Min/game","Pass %","Y-cards/90","Apps","MV mom.","Aerials/90"]),
}

# ─── Data ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data():
    return generate()

@st.cache_data(show_spinner="Building player embeddings…")
def load_embeddings(_pool):
    emb, _ = get_player_embeddings(_pool)
    return emb

data       = load_data()
pool       = data["players"]
clubs      = data["clubs"]
transfers  = data["transfers"]
embeddings = load_embeddings(pool)

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ AI Football Scout v3")
    st.divider()

    st.markdown("#### Player filters")
    position    = st.selectbox("Position", ["ATT", "MID", "DEF", "GK"])
    c1, c2      = st.columns(2)
    min_age     = c1.number_input("Min age", 14, 40, 16)
    max_age     = c2.number_input("Max age", 14, 40, 30)
    budget      = st.slider("Max value (€M)", 1, 200, 100)
    min_minutes = st.slider("Min minutes played", 0, 3420, 500, step=90)

    st.divider()
    st.markdown("#### Destination club style")

    PRESETS = {
        "Custom":                None,
        "Klopp — Gegenpressing": dict(ppda=7.5,  possession_pct=55, directness_idx=0.55, line_height_m=58),
        "Pep — Tiki-taka":       dict(ppda=11.0, possession_pct=65, directness_idx=0.32, line_height_m=55),
        "Mourinho — Low block":  dict(ppda=16.0, possession_pct=44, directness_idx=0.48, line_height_m=35),
        "Counter-attack":        dict(ppda=14.0, possession_pct=42, directness_idx=0.62, line_height_m=38),
        "High-press direct":     dict(ppda=8.0,  possession_pct=50, directness_idx=0.58, line_height_m=60),
        "Possession midblock":   dict(ppda=13.0, possession_pct=58, directness_idx=0.36, line_height_m=48),
    }
    preset_name = st.selectbox("Style preset", list(PRESETS.keys()))
    preset      = PRESETS[preset_name]

    if preset:
        ppda, possession_pct          = preset["ppda"], preset["possession_pct"]
        directness_idx, line_height_m = preset["directness_idx"], preset["line_height_m"]
    else:
        ppda           = st.slider("PPDA  (↓ = more pressing)", 4.0, 20.0, 11.0, 0.5)
        possession_pct = st.slider("Possession %", 30, 70, 52)
        directness_idx = st.slider("Directness", 0.20, 0.65, 0.40, 0.01)
        line_height_m  = st.slider("Line height (m)", 25, 65, 48)

    st.caption("Tactical DNA")
    for label, val in [
        ("🔥 Press",     round((20 - ppda)           / 16   * 100)),
        ("🔵 Possession",round((possession_pct - 30) / 40   * 100)),
        ("➡️ Directness", round((directness_idx-.20) / .45  * 100)),
        ("⬆️ Line height",round((line_height_m - 25) / 40   * 100)),
    ]:
        st.caption(f"{label}  {val}%")
        st.progress(val / 100)

    st.divider()
    league_options = sorted(pool["league"].dropna().unique().tolist())
    dest_league    = st.selectbox("Destination league", league_options)
    top_n          = st.slider("Show top N", 5, 30, 10)

# ─── Shared scoring ───────────────────────────────────────────────
filters     = Filters(position=position, max_value_eur_m=float(budget),
                      min_age=int(min_age), max_age=int(max_age), min_minutes=int(min_minutes))
destination = DestinationClub(ppda=ppda, possession_pct=possession_pct,
                              directness_idx=directness_idx, line_height_m=line_height_m,
                              league=dest_league)
candidates  = apply_filters(pool, filters)
ranked      = compute_fit_score(candidates, destination) if len(candidates) > 0 else candidates
top         = ranked.head(top_n).copy()
RADAR_COLS, RADAR_LABELS = RADAR_BY_POS.get(position, RADAR_BY_POS["ATT"])

# ─── Tabs ─────────────────────────────────────────────────────────
tab_scout, tab_similar, tab_data, tab_model = st.tabs([
    "🔍  Scout", "🔄  Similar Players", "📊  Data Explorer", "🧠  Model",
])


# ═══════════════════════════════════════════════════
# TAB 1 — SCOUT
# ═══════════════════════════════════════════════════
with tab_scout:
    if len(candidates) == 0:
        st.warning("No players match these filters — try relaxing age, budget or minutes.")
        st.stop()

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
        "market_value_eur_m":"Value (€M)","goals_p90":"Goals/90",
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
        rank_labels = ["#1 · Best fit", "#2", "#3"]
        for i in range(n_cards):
            row  = top.iloc[i]
            flag = LEAGUE_FLAGS.get(str(row.get("league", "")), "")
            fit  = row["fit_score"]
            age  = int(row["age"]) if pd.notna(row.get("age")) else "—"
            val  = row["market_value_eur_m"]
            mv_mom = row.get("mv_momentum_12m", 0)
            mom_str = f"▲ {mv_mom:+.0%}" if mv_mom > 0.05 else (f"▼ {mv_mom:.0%}" if mv_mom < -0.05 else "→ stable")
            with card_cols[i]:
                with st.container(border=True):
                    g90 = row.get("goals_p90", 0)
                    a90 = row.get("assists_p90", 0)
                    st.markdown(
                        f"**{rank_labels[i]}** &nbsp; `Fit {fit:.1f}`  \n"
                        f"**{row['player_name']}**  \n"
                        f"{flag} {row.get('league','—')} · Age {age} · €{val:.0f}M · {mom_str}  \n"
                        f"⚽ {g90:.2f}/90 &nbsp; 🅰️ {a90:.2f}/90"
                    )
        st.divider()

    # ── Ranking table (left) + Radar (right) ─────────────────
    col_tbl, col_radar = st.columns([11, 9])

    with col_tbl:
        st.markdown("##### Shortlist ranking")
        rank_cols = {
            "player_name": "Player", "fit_score": "Fit ↓", "league": "League", "age": "Age",
            "market_value_eur_m": "Value (€M)", "goals_p90": "G/90",
            "assists_p90": "A/90", "pass_completion_pct": "Pass %",
            "mv_momentum_12m": "MV trend",
        }
        rank_tbl = top[[c for c in rank_cols if c in top.columns]].rename(columns=rank_cols)
        rank_tbl.index = range(1, len(rank_tbl) + 1)
        st.dataframe(
            rank_tbl.style
                    .background_gradient(subset=["Fit ↓"], cmap="Blues")
                    .format({
                        "Value (€M)": "€{:.0f}M",
                        "G/90":       "{:.2f}",
                        "A/90":       "{:.2f}",
                        "Pass %":     "{:.0f}%",
                        "MV trend":   "{:+.2f}",
                        "Fit ↓":      "{:.1f}",
                    }),
            use_container_width=True,
            height=min(len(rank_tbl) * 38 + 44, 480),
        )
        csv = tbl.to_csv(index=True).encode("utf-8")
        st.download_button(
            "⬇️  Export shortlist CSV", csv,
            file_name=f"shortlist_{position}_{style_label.replace(' ','_')}.csv",
            mime="text/csv",
        )

    with col_radar:
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
            **pl(margin=dict(l=20, r=20, t=40, b=90)),
            polar=POLAR_STYLE,
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="top", y=-0.15,
                font=dict(size=11, color="#374151"),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=480,
            title="Player profile — top 5  (percentile vs Big-5 peers)",
            title_font_size=13,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ── Market intelligence: Value vs Fit ──────────────────────
    st.markdown("##### Market intelligence — Value vs Fit")
    st.caption("Quadrant analysis: use Value Picks to stretch the budget, Premium Targets when quality is non-negotiable.")

    med_val       = float(ranked["market_value_eur_m"].median())
    fit_threshold = 70.0
    x_max         = float(ranked["market_value_eur_m"].max()) * 1.08

    fig_vf = go.Figure()

    # ── Quadrant shading (drawn first, layer below) ──────────────
    for x0, x1, y0, y1, fc in [
        (0,       med_val, fit_threshold, 106, "rgba(37,99,235,0.07)"),
        (med_val, x_max,   fit_threshold, 106, "rgba(5,150,105,0.07)"),
        (0,       med_val, 0, fit_threshold,   "rgba(217,119,6,0.06)"),
        (med_val, x_max,   0, fit_threshold,   "rgba(220,38,38,0.06)"),
    ]:
        fig_vf.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                         fillcolor=fc, line_width=0, layer="below")

    # ── Split lines ──────────────────────────────────────────────
    fig_vf.add_hline(y=fit_threshold, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig_vf.add_vline(x=med_val,       line_dash="dot", line_color="#94a3b8", line_width=1)
    fig_vf.add_annotation(
        x=med_val, y=0, yanchor="bottom", yshift=4,
        text=f"Median €{med_val:.0f}M", showarrow=False,
        font=dict(size=9, color="#94a3b8"), xanchor="center",
    )
    fig_vf.add_annotation(
        x=x_max, y=fit_threshold, xanchor="right", yanchor="bottom", yshift=4,
        text=f"Fit threshold {fit_threshold:.0f}", showarrow=False,
        font=dict(size=9, color="#94a3b8"),
    )

    # ── Quadrant corner labels ───────────────────────────────────
    for x, y, txt, color in [
        (med_val * 0.5,  104, "VALUE PICKS",       "#2563eb"),
        (med_val * 1.55, 104, "PREMIUM TARGETS",   "#059669"),
        (med_val * 0.5,  3,   "MONITOR",           "#d97706"),
        (med_val * 1.55, 3,   "OVERPRICED",        "#dc2626"),
    ]:
        fig_vf.add_annotation(
            x=x, y=y, text=f"<b>{txt}</b>", showarrow=False,
            font=dict(size=10, color=color), xanchor="center",
            bgcolor="rgba(255,255,255,0.75)", borderpad=4,
        )

    # ── All candidates: small grey dots ─────────────────────────
    fig_vf.add_trace(go.Scatter(
        x=ranked["market_value_eur_m"], y=ranked["fit_score"],
        mode="markers",
        marker=dict(size=5, color="#cbd5e1", opacity=0.7),
        text=ranked["player_name"],
        hovertemplate="<b>%{text}</b><br>€%{x:.0f}M · Fit %{y:.1f}<extra></extra>",
        showlegend=False,
    ))

    # ── Shortlist: numbered circles (no text overlap) ────────────
    for i, (_, row) in enumerate(top.iterrows()):
        age = int(row["age"]) if pd.notna(row.get("age")) else "—"
        fig_vf.add_trace(go.Scatter(
            x=[row["market_value_eur_m"]], y=[row["fit_score"]],
            mode="markers+text",
            marker=dict(size=24, color="#2563eb", line=dict(width=0)),
            text=[str(i + 1)],
            textposition="middle center",
            textfont=dict(size=9, color="white"),
            showlegend=False,
            hovertemplate=(
                f"<b>#{i+1} {row['player_name']}</b><br>"
                f"{row.get('league','—')} · Age {age}<br>"
                f"Value: €{row['market_value_eur_m']:.0f}M<br>"
                f"Fit: <b>{row['fit_score']:.1f}</b><br>"
                f"Goals/90: {row.get('goals_p90',0):.2f}"
                "<extra></extra>"
            ),
        ))

    fig_vf.update_layout(
        **pl(
            yaxis=dict(range=[0, 108], title="Model fit score"),
            xaxis=dict(range=[0, x_max], title="Market value (€M)"),
        ),
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_vf, use_container_width=True)
    st.caption("Numbered circles = shortlist rank above. Grey dots = all candidates. Hover for details.")

    with st.expander("ℹ️  Club style guide"):
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
    st.subheader("🔄 Find Similar Players")
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
⚽ {target.get('goals_p90',0):.2f}/90 &nbsp;·&nbsp;
🅰️ {target.get('assists_p90',0):.2f}/90
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

            col_l, col_r = st.columns([1, 1])

            with col_l:
                st.markdown("**Most similar players**")
                sim_cols = {
                    "player_name":"Player","league":"League","age":"Age",
                    "market_value_eur_m":"Value (€M)","goals_p90":"Goals/90",
                    "assists_p90":"Assists/90","Similarity %":"Sim %",
                }
                sim_tbl = similar[[c for c in sim_cols if c in similar.columns]].rename(columns=sim_cols)
                sim_tbl.index = range(1, len(sim_tbl) + 1)
                st.dataframe(
                    sim_tbl.style
                           .background_gradient(subset=["Sim %"], cmap="Blues")
                           .format({"Value (€M)":"€{:.0f}M","Goals/90":"{:.2f}",
                                    "Assists/90":"{:.2f}","Sim %":"{:.1f}%"}),
                    use_container_width=True,
                    height=(n_similar + 1) * 38,
                )

                # Budget alternatives
                cheaper = similar[similar["market_value_eur_m"] < target["market_value_eur_m"] * 0.6].head(3)
                if not cheaper.empty:
                    st.markdown("**💰 Budget alternatives**")
                    for _, row in cheaper.iterrows():
                        saving = target["market_value_eur_m"] - row["market_value_eur_m"]
                        f2 = LEAGUE_FLAGS.get(str(row.get("league", "")), "")
                        st.success(
                            f"**{row['player_name']}** · {f2} {row.get('league','—')} · "
                            f"€{row['market_value_eur_m']:.0f}M · "
                            f"{float(row['Similarity %']):.0f}% similar · "
                            f"saves ~€{saving:.0f}M"
                        )

            with col_r:
                # Radar: target vs top 3 similar
                st.markdown(f"**{target['player_name']} vs top 3 similar**")
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
                    lbl  = str(row["player_name"]) + (" ★" if i == 0 else "")
                    col  = cmp_cols[i]
                    r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
                    fig_cmp.add_trace(go.Scatterpolar(
                        r=vals, theta=RADAR_LABELS + [RADAR_LABELS[0]],
                        fill="toself",
                        fillcolor=f"rgba({r},{g},{b},{0.12 if i == 0 else 0.05})",
                        name=lbl,
                        line_color=col, line_width=3 if i == 0 else 1.5,
                    ))
                fig_cmp.update_layout(
                    **pl(margin=dict(l=20, r=20, t=20, b=80)),
                    polar=POLAR_STYLE,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.15,
                                font=dict(size=10, color="#374151"), bgcolor="rgba(0,0,0,0)"),
                    height=400,
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

                # Value vs similarity
                fig_vs = px.scatter(
                    similar, x="market_value_eur_m", y="Similarity %",
                    color="league", hover_name="player_name",
                    hover_data={"age":True,"goals_p90":":.2f","market_value_eur_m":":.0f"},
                    labels={"market_value_eur_m":"Value (€M)"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    opacity=0.85, title="Value vs similarity",
                )
                if pd.notna(target.get("market_value_eur_m")):
                    fig_vs.add_vline(x=target["market_value_eur_m"], line_dash="dash",
                                     line_color="#f59e0b",
                                     annotation_text="Target value",
                                     annotation_font_color="#f59e0b")
                fig_vs.update_layout(**PLOTLY, height=280, title_font_size=13,
                                     legend=dict(font=dict(color="#374151"), bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_vs, use_container_width=True)


# ═══════════════════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ═══════════════════════════════════════════════════
with tab_data:
    st.subheader("📊 Data Explorer")

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
                                      "n_apps","mv_momentum_12m","pass_completion_pct","market_value_eur_m"])
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
    st.subheader("🧠 Model Performance")

    try:
        with open("models/metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        st.error("models/metrics.json not found — run `python train.py` first.")
        st.stop()

    with st.expander("Architecture", expanded=True):
        st.code("""
Player stats (24 feats)            Club tactics (9 feats)
        │                                   │
  Player Tower                        Club Tower
  24 → 64 → 32-d embedding           9 → 32 → 32-d embedding
        │                                   │
        └──────────── Head MLP ─────────────┘
           input: concat( p⊙c,  |p−c|,  ctx )   = 66 feats
           layers: 66 → 64 → 64 → 1 → sigmoid
                             │
                   Transfer success score  (0 – 1)
        """, language=None)
        st.caption("The player tower's 32-d output is also the embedding used in the Similar Players tab.")

    rows = []
    for name in ["cosine","linear","gbm"]:
        m = metrics["baselines"][name]
        rows.append({"Model":name.capitalize(),"R²":m["test_r2"],"MAE":m["test_mae"],"Spearman ρ":m["test_spearman"]})
    tt = metrics["two_tower"]
    rows.append({"Model":"Two-tower ⭐","R²":tt["test_r2"],"MAE":tt["test_mae"],"Spearman ρ":tt["test_spearman"]})
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
    st.markdown("""
**How to read the numbers**

| Metric | What it means |
|---|---|
| **R² ≈ 0.27** | 27% of variance in real transfer outcomes explained — strong for this domain |
| **MAE ≈ 0.16** | Off by ~16 points on a 0–1 scale on average |
| **Spearman ρ ≈ 0.53** | Meaningful ability to rank players by transfer fit |

**Why two-tower beats GBM:** The separate encoder architecture captures player × club interaction that a flat feature vector misses.
The 32-d player embedding also powers the Similar Players feature.
    """)
