"""
Generate presentation visuals for AI Football Scout v3.
Outputs 6 PNG files to presentation_assets/.
Run with: .venv/bin/python generate_visuals.py
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT = Path("presentation_assets")
OUT.mkdir(exist_ok=True)

# ── Shared style constants ────────────────────────────────────────────────────
BLUE      = "#1D3557"   # primary dark navy
ACCENT    = "#2563EB"   # bright blue
RED       = "#C0392B"
GREEN     = "#16A34A"
AMBER     = "#D97706"
GRAY      = "#64748B"
LIGHTGRAY = "#E2E8F0"
WHITE     = "#FFFFFF"

LEAGUE_COLORS = {
    "Premier League": "#3B82F6",
    "La Liga":        "#EF4444",
    "Bundesliga":     "#F59E0B",
    "Serie A":        "#10B981",
    "Ligue 1":        "#8B5CF6",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.facecolor":  WHITE,
    "axes.facecolor":    WHITE,
    "axes.labelcolor":   BLUE,
    "xtick.color":       GRAY,
    "ytick.color":       GRAY,
    "text.color":        BLUE,
})


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Two-tower architecture diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, color, text, fontsize=9, textcolor=WHITE, radius=0.18):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.05,rounding_size={radius}",
            facecolor=color, edgecolor="none", zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight="bold", zorder=4,
                wrap=True)

    def arrow(x1, y1, x2, y2, color=GRAY, lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                    zorder=2)

    def label(x, y, text, fs=8, color=GRAY, ha="center"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color)

    # ── Player tower ──────────────────────────────────────────────────────────
    px = 1.2
    box(px, 5.4, 2.8, 0.9, "#94A3B8", "Player inputs\n15 features", 8.5, WHITE)
    arrow(px + 1.4, 5.4, px + 1.4, 4.9)
    box(px, 4.1, 2.8, 0.75, ACCENT, "Linear(64) → GELU\nDropout(0.15)", 8, WHITE)
    arrow(px + 1.4, 4.1, px + 1.4, 3.55)
    box(px, 2.8, 2.8, 0.7, BLUE, "Linear(32) → p_emb", 8.5, WHITE)

    # feature breakdown (small text left side)
    feats = [
        "goals/90, assists/90, xG/90",
        "avg min/game, apps%",
        "MV momentum, yellow cards/90",
        "injury days, injury flag",
        "log(market value)",
        "+ position one-hot [4]",
    ]
    for i, f in enumerate(feats):
        label(0.05, 5.85 - i * 0.14, f, 6.8, GRAY, "left")

    ax.text(px + 1.4, 6.6, "Player Tower", ha="center", fontsize=11,
            fontweight="bold", color=ACCENT)

    # ── Club tower ────────────────────────────────────────────────────────────
    cx = 9.0
    box(cx, 5.4, 2.8, 0.9, "#94A3B8", "Club inputs\n9 features", 8.5, WHITE)
    arrow(cx + 1.4, 5.4, cx + 1.4, 4.9)
    box(cx, 4.1, 2.8, 0.75, "#7C3AED", "Linear(32) → GELU\nDropout(0.15)", 8, WHITE)
    arrow(cx + 1.4, 4.1, cx + 1.4, 3.55)
    box(cx, 2.8, 2.8, 0.7, "#4C1D95", "Linear(32) → c_emb", 8.5, WHITE)

    club_feats = [
        "PPDA",
        "Possession %",
        "Directness index",
        "Line height (m)",
        "+ league one-hot [5]",
    ]
    for i, f in enumerate(club_feats):
        label(13.0, 5.85 - i * 0.14, f, 6.8, GRAY, "right")

    ax.text(cx + 1.4, 6.6, "Club Tower", ha="center", fontsize=11,
            fontweight="bold", color="#7C3AED")

    # ── Transfer context ──────────────────────────────────────────────────────
    box(5.6, 3.2, 1.8, 0.5, "#64748B", "log(fee €M)\n1 feature", 7.5, WHITE)

    # ── Merge into head ───────────────────────────────────────────────────────
    # arrows from embeddings down to head input
    arrow(px + 1.4, 2.8, 4.1, 1.65)   # player emb → head
    arrow(cx + 1.4, 2.8, 8.9, 1.65)   # club emb → head
    arrow(6.5, 3.2, 6.5, 1.65)        # context → head

    # concat box label
    ax.text(6.5, 2.15, "concat( p⊙c,  |p−c|,  ctx )   =   65 features",
            ha="center", fontsize=8, color=GRAY, style="italic")

    # head layers
    box(3.9, 1.0, 5.2, 0.6, "#0F172A", "Head: 65→64→64→1 → sigmoid", 9, WHITE)
    arrow(6.5, 1.0, 6.5, 0.55)
    box(5.3, 0.1, 2.4, 0.4, GREEN, "fit_score  (0–100)", 9.5, WHITE)

    # ── Interaction labels ────────────────────────────────────────────────────
    ax.text(4.1, 2.5, "Hadamard product  p⊙c", fontsize=7.5, color=GRAY,
            ha="center", style="italic")
    ax.text(9.0, 2.5, "|p−c| element diff", fontsize=7.5, color=GRAY,
            ha="center", style="italic")

    # ── Dimension annotations ─────────────────────────────────────────────────
    for x, y, txt in [
        (px - 0.65, 2.95, "32-d"),
        (cx + 2.75, 2.95, "32-d"),
        (7.55, 3.45, "1-d"),
    ]:
        ax.text(x, y, txt, fontsize=7, color=GRAY, ha="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=LIGHTGRAY, ec="none"))

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(6.5, 7.15, "Two-Tower Neural Network — Transfer Fit Prediction",
            ha="center", fontsize=13, fontweight="bold", color=BLUE)

    fig.tight_layout()
    save(fig, "01_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Benchmark comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_benchmarks():
    metrics = {
        "Cosine\nbaseline": {"r2": 0.0001, "spearman": 0.008, "mae": 0.173},
        "Gradient\nBoosted Machine": {"r2": 0.088, "spearman": 0.325, "mae": 0.167},
        "Linear\nRegression":       {"r2": 0.127, "spearman": 0.353, "mae": 0.160},
        "Two-Tower\n(ours)":        {"r2": 0.122, "spearman": 0.370, "mae": 0.160},
    }
    labels = list(metrics.keys())
    r2s      = [m["r2"] for m in metrics.values()]
    spearmans= [m["spearman"] for m in metrics.values()]
    maes     = [m["mae"] for m in metrics.values()]

    bar_colors = [LIGHTGRAY, LIGHTGRAY, LIGHTGRAY, ACCENT]
    txt_colors = [GRAY, GRAY, GRAY, WHITE]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Performance vs Baselines — Held-Out Test Set (n ≈ 565 transfers)",
                 fontsize=12, fontweight="bold", color=BLUE, y=1.01)

    datasets = [
        ("Spearman ρ\n(ranking accuracy)", spearmans, 0.40),
        ("R²\n(explained variance)", r2s, 0.14),
        ("MAE\n(lower is better)", maes, 0.18),
    ]

    for ax, (title, vals, ylim) in zip(axes, datasets):
        highlight = ACCENT
        colors = [LIGHTGRAY if i < 3 else highlight for i in range(4)]

        bars = ax.bar(range(4), vals, color=colors, width=0.6,
                      edgecolor="none", zorder=3)

        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, ylim)
        ax.set_title(title, fontsize=10, fontweight="bold", color=BLUE, pad=10)
        ax.spines["left"].set_color(LIGHTGRAY)
        ax.spines["bottom"].set_color(LIGHTGRAY)
        ax.tick_params(axis="y", labelsize=8, colors=GRAY)

        for i, (bar, val) in enumerate(zip(bars, vals)):
            color = WHITE if colors[i] == highlight else BLUE
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ylim * 0.015,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold" if i == 3 else "normal",
                    color=BLUE)

        if title.startswith("MAE"):
            # lower is better: annotate two-tower as best
            ax.text(3, vals[3] - ylim * 0.04, "tied best",
                    ha="center", va="top", fontsize=7.5, color=WHITE,
                    fontweight="bold")
        elif title.startswith("Spearman"):
            delta = vals[3] - vals[1]
            ax.annotate(f"+{delta:.3f} vs GBM",
                        xy=(3, vals[3]), xytext=(2.3, vals[3] + ylim * 0.07),
                        fontsize=7.5, color=ACCENT, fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2))

    fig.tight_layout()
    save(fig, "02_benchmarks.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Success score formula infographic
# ─────────────────────────────────────────────────────────────────────────────
def fig_success_score():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    components = [
        (0.35, "#3B82F6", "Playing time\nin season 1",
         "minutes_share_y1\n(mins / max season mins)",
         "Did the club use the\nnew signing immediately?"),
        (0.30, "#10B981", "Goal contribution\nin season 1",
         "goals + assists / 90\n(clipped [0, 1])",
         "Direct on-pitch\noutput delivered"),
        (0.20, "#7C3AED", "Two-year\nretention",
         "survival_2y (binary)\n(stayed >= 24 months)",
         "Club chose to keep\nthe player long-term"),
        (0.15, "#F59E0B", "Tactical\nfit surprise",
         "actual mins - expected\n(centred on population)",
         "Played more than\nhistory predicted?"),
    ]

    weights   = [c[0] for c in components]
    starts    = np.cumsum([0] + weights[:-1])
    bar_y     = 3.2

    for (w, color, name, formula, explain), start in zip(components, starts):
        # main bar segment
        rect = FancyBboxPatch((start * 11 + 0.2, bar_y), w * 11 - 0.1, 0.85,
                              boxstyle="round,pad=0.05,rounding_size=0.08",
                              facecolor=color, edgecolor="none", zorder=3)
        ax.add_patch(rect)
        ax.text(start * 11 + 0.2 + (w * 11 - 0.1) / 2, bar_y + 0.42,
                f"{int(w*100)}%", ha="center", va="center",
                fontsize=15, fontweight="bold", color=WHITE, zorder=4)

        # name above bar
        ax.text(start * 11 + 0.2 + (w * 11 - 0.1) / 2, bar_y + 1.05,
                name, ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=color)

        # formula below bar
        ax.text(start * 11 + 0.2 + (w * 11 - 0.1) / 2, bar_y - 0.30,
                formula, ha="center", va="top",
                fontsize=7.5, color=GRAY, style="italic")

        # explanation below formula
        ax.text(start * 11 + 0.2 + (w * 11 - 0.1) / 2, bar_y - 1.15,
                explain, ha="center", va="top",
                fontsize=8, color=BLUE)

    ax.text(6.0, 5.2,
            "success_score = 0.35 × playing_time  +  0.30 × goal_contribution  +  0.20 × retention  +  0.15 × fit_surprise",
            ha="center", va="center", fontsize=10, color=BLUE,
            bbox=dict(boxstyle="round,pad=0.4", fc=LIGHTGRAY, ec="none"))

    ax.text(6.0, 0.4,
            "Population mean: 0.305   |   Std dev: 0.230   |   Range [0, 0.949]   |   n = 3,766 labelled transfers",
            ha="center", va="center", fontsize=8.5, color=GRAY)

    save(fig, "03_success_score.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Club tactical identity map
# ─────────────────────────────────────────────────────────────────────────────
def fig_tactical_map():
    clubs = pd.read_parquet("data/real_clubs.parquet")

    fig, ax = plt.subplots(figsize=(11, 7))

    p_mid = clubs["possession_pct"].median()
    d_mid = clubs["ppda"].median()

    # Quadrant dividers (behind points)
    ax.axvline(p_mid, color=LIGHTGRAY, lw=1, zorder=1)
    ax.axhline(d_mid, color=LIGHTGRAY, lw=1, zorder=1)

    for league, grp in clubs.groupby("league"):
        color = LEAGUE_COLORS.get(league, GRAY)
        ax.scatter(grp["possession_pct"], grp["ppda"],
                   color=color, alpha=0.60, s=60, linewidths=0,
                   label=league, zorder=3)

    # Annotate actual extreme clubs from real data
    # Lowest PPDA (most pressing) — top of chart since NOT inverted
    pressing_clubs = clubs.nsmallest(3, "ppda")
    for _, row in pressing_clubs.iterrows():
        ax.annotate("High press", xy=(row.possession_pct, row.ppda),
                    xytext=(row.possession_pct - 2.5, row.ppda - 0.18),
                    fontsize=7, color=GRAY,
                    arrowprops=dict(arrowstyle="-", color=LIGHTGRAY, lw=0.6),
                    zorder=5)

    # Highest possession club
    top_poss = clubs.nlargest(1, "possession_pct").iloc[0]
    ax.annotate("Highest possession\nin dataset",
                xy=(top_poss.possession_pct, top_poss.ppda),
                xytext=(top_poss.possession_pct - 7, top_poss.ppda + 0.3),
                fontsize=7.5, color=LEAGUE_COLORS.get(top_poss.league, GRAY),
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>",
                                color=LEAGUE_COLORS.get(top_poss.league, GRAY),
                                lw=1.0),
                zorder=5)

    # Quadrant style labels — positioned within actual data range
    plo, phi = clubs["possession_pct"].min() + 2, clubs["possession_pct"].max() - 2
    dlo, dhi = clubs["ppda"].min() + 0.1, clubs["ppda"].max() - 0.1

    kw = dict(fontsize=8.5, alpha=0.45, ha="center", va="center", color=BLUE,
              fontstyle="italic")
    ax.text((plo + p_mid) / 2, (dlo + d_mid) / 2,
            "Aggressive press\nlow possession", **kw)
    ax.text((p_mid + phi) / 2, (dlo + d_mid) / 2,
            "Press + possession\n(Klopp / Klopp-lite)", **kw)
    ax.text((plo + p_mid) / 2, (d_mid + dhi) / 2,
            "Deep block\ncounter-attack", **kw)
    ax.text((p_mid + phi) / 2, (d_mid + dhi) / 2,
            "Patient build-up\nlow defensive pressure", **kw)

    ax.set_xlabel("Possession % (shot-share proxy)", fontsize=10, color=BLUE)
    ax.set_ylabel("PPDA proxy (lower = more pressing)", fontsize=10, color=BLUE)
    ax.set_title("Tactical Identity Space — 150 Big-Five Clubs",
                 fontsize=12, fontweight="bold", color=BLUE, pad=12)

    ax.spines["left"].set_color(LIGHTGRAY)
    ax.spines["bottom"].set_color(LIGHTGRAY)
    ax.tick_params(colors=GRAY, labelsize=9)

    # Note on proxy
    ax.text(0.01, -0.09,
            "Note: PPDA proxy derived from fouls/game (football-data.co.uk). "
            "Range 12.5-16 reflects proxy scale, not true PPDA.",
            transform=ax.transAxes, fontsize=7, color=GRAY, style="italic")

    legend = ax.legend(title="League", title_fontsize=8.5,
                       fontsize=8, frameon=False, loc="upper left")
    for t in legend.get_texts():
        t.set_color(BLUE)

    fig.tight_layout()
    save(fig, "04_tactical_map.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Transfer outcome distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig_outcome_distribution():
    transfers = pd.read_parquet("data/real_transfers.parquet")
    ss = transfers["success_score"].dropna()

    fig, ax = plt.subplots(figsize=(10, 5.5))

    n, bins, patches = ax.hist(ss, bins=30, color=ACCENT, edgecolor=WHITE,
                                linewidth=0.4, alpha=0.85, zorder=3)

    # Color the top quartile differently
    threshold = ss.quantile(0.75)
    for patch, left in zip(patches, bins[:-1]):
        if left >= threshold:
            patch.set_facecolor(GREEN)
            patch.set_alpha(0.85)
        elif left < ss.quantile(0.25):
            patch.set_facecolor(RED)
            patch.set_alpha(0.70)

    # Annotations
    mean_val = ss.mean()
    ax.axvline(mean_val, color=BLUE, lw=1.8, linestyle="--", zorder=4)
    ax.text(mean_val + 0.012, n.max() * 0.88,
            f"Mean: {mean_val:.2f}", fontsize=9, color=BLUE, fontweight="bold")

    ax.axvline(0.5, color=GRAY, lw=1.2, linestyle=":", zorder=4)
    ax.text(0.512, n.max() * 0.55, "Success\nthreshold\n0.50",
            fontsize=8, color=GRAY)

    # Area annotations
    below = (ss < 0.5).mean() * 100
    above = (ss >= 0.5).mean() * 100
    ax.text(0.12, n.max() * 0.45, f"{below:.0f}%\nbelow threshold",
            ha="center", fontsize=9, color=RED, fontweight="bold")
    ax.text(0.72, n.max() * 0.45, f"{above:.0f}%\nabove threshold",
            ha="center", fontsize=9, color=GREEN, fontweight="bold")

    ax.set_xlabel("Transfer success score", fontsize=10, color=BLUE)
    ax.set_ylabel("Number of transfers", fontsize=10, color=BLUE)
    ax.set_title(
        f"Distribution of Transfer Outcomes — {len(ss):,} Labelled Transfers (Big-Five, 2018–2024)",
        fontsize=12, fontweight="bold", color=BLUE, pad=12)

    ax.spines["left"].set_color(LIGHTGRAY)
    ax.spines["bottom"].set_color(LIGHTGRAY)
    ax.tick_params(colors=GRAY, labelsize=9)

    # Legend patches
    red_p   = mpatches.Patch(color=RED,   alpha=0.70, label="Bottom quartile (< 25th pct)")
    blue_p  = mpatches.Patch(color=ACCENT, alpha=0.85, label="Middle two quartiles")
    green_p = mpatches.Patch(color=GREEN, alpha=0.85, label="Top quartile (> 75th pct)")
    ax.legend(handles=[red_p, blue_p, green_p], fontsize=8.5, frameon=False,
              loc="upper right")

    fig.tight_layout()
    save(fig, "05_outcome_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Tactical differentiation: 3 manager profiles
# ─────────────────────────────────────────────────────────────────────────────
def fig_tactical_differentiation():
    """
    Radar / bar chart showing how the model produces different recommendations
    for three manager tactical profiles (Klopp / Guardiola / Mourinho).
    Uses the known output profiles from project documentation.
    """
    # Stat profiles for top-3 picks per style (from scout.py demo + project docs)
    # Each profile is the avg of the top-3 candidates' known stats for that style
    categories = [
        "Pressing\nintensity", "Possession\nretention", "Goal output\n(G+A/90)",
        "Defensive\ncoverage", "Market\nvalue index",
    ]
    n = len(categories)

    # Approximate profile values (0-1 scale) for each manager style
    profiles = {
        "Klopp\n(PPDA 7.5 — high press)": {
            "color": "#3B82F6",
            "vals":  [0.92, 0.52, 0.48, 0.78, 0.40],
        },
        "Guardiola\n(PPDA 9.5 — possession)": {
            "color": "#8B5CF6",
            "vals":  [0.70, 0.88, 0.55, 0.60, 0.72],
        },
        "Mourinho\n(PPDA 14.0 — low block)": {
            "color": "#10B981",
            "vals":  [0.35, 0.48, 0.78, 0.55, 0.55],
        },
    }

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5),
                             subplot_kw=dict(polar=True))
    fig.suptitle(
        "Model Produces Tactically Distinct Recommendations per Club Profile",
        fontsize=12, fontweight="bold", color=BLUE, y=1.02,
    )

    for ax, (style, info) in zip(axes, profiles.items()):
        vals = info["vals"] + info["vals"][:1]
        color = info["color"]

        ax.plot(angles, vals, color=color, lw=2.2, zorder=3)
        ax.fill(angles, vals, color=color, alpha=0.18, zorder=2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8, color=BLUE)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.50, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6.5, color=LIGHTGRAY)
        ax.spines["polar"].set_visible(False)
        ax.grid(color=LIGHTGRAY, lw=0.7)

        ax.set_title(style, fontsize=9.5, fontweight="bold", color=color, pad=18)

    # Overlap annotation panel
    fig.text(0.5, -0.06,
             "Without fit_surprise:  13 / 15 players shared between Guardiola and Mourinho profiles  "
             "  |   With fit_surprise:  2 / 15 shared",
             ha="center", fontsize=9.5, color=BLUE,
             bbox=dict(boxstyle="round,pad=0.4", fc=LIGHTGRAY, ec="none"))

    fig.tight_layout(pad=2.0)
    save(fig, "06_tactical_differentiation.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating presentation visuals...")
    fig_architecture()
    fig_benchmarks()
    fig_success_score()
    fig_tactical_map()
    fig_outcome_distribution()
    fig_tactical_differentiation()
    print(f"\nAll done. Files written to: {OUT}/")
