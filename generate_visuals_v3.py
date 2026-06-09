"""
Generate three supplementary figures for the thesis.
  11_pred_vs_actual.png   — predicted vs actual scatter + residuals
  12_tactical_output.png  — actual model output for 3 tactical profiles
  13_label_correlations.png — label component correlation matrix
Run with: .venv/bin/python generate_visuals_v3.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import pickle
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")

OUT  = Path("presentation_assets")
OUT.mkdir(exist_ok=True)

BLUE      = "#1D3557"
ACCENT    = "#2563EB"
RED       = "#DC2626"
GREEN     = "#16A34A"
AMBER     = "#D97706"
GRAY      = "#64748B"
LIGHTGRAY = "#E2E8F0"
WHITE     = "#FFFFFF"
PURPLE    = "#7C3AED"

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


def load_model():
    cfg = pickle.loads(Path("models/feature_config.pkl").read_bytes())
    from model import TwoTowerFitModel
    net = TwoTowerFitModel(cfg.player_scaler.n_features_in_,
                           cfg.club_scaler.n_features_in_,
                           cfg.ctx_scaler.n_features_in_)
    net.load_state_dict(torch.load("models/two_tower.pt", weights_only=True))
    net.eval()
    return net, cfg


# ─────────────────────────────────────────────────────────────────────────────
# 1. Predicted vs actual scatter + residual distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig_pred_vs_actual():
    preds   = np.load(OUT / "model_preds.npy")
    actuals = np.load(OUT / "model_actuals.npy")
    resids  = preds - actuals

    rho_full, _ = spearmanr(preds, actuals)
    # Test-set metrics from metrics.json (authoritative)
    rho_test  = 0.370
    r2_test   = 0.122
    mae_test  = 0.160

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Evaluation: Predicted vs Actual Transfer Success Score",
                 fontsize=13, fontweight="bold", color=BLUE, y=1.01)

    # ── Left: hexbin scatter ─────────────────────────────────────────────────
    ax = axes[0]
    hb = ax.hexbin(preds, actuals, gridsize=30, cmap="Blues",
                   mincnt=1, alpha=0.9, zorder=3)
    cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Transfer count", fontsize=8.5, color=GRAY)
    cb.ax.tick_params(labelsize=8, colors=GRAY)

    # Diagonal reference
    lo, hi = 0, 1
    ax.plot([lo, hi], [lo, hi], color=LIGHTGRAY, lw=1.5,
            linestyle="--", zorder=2, label="Perfect prediction")

    # Binned mean trend line
    bins = np.linspace(preds.min(), preds.max(), 12)
    bin_ids = np.digitize(preds, bins)
    bin_means_x, bin_means_y = [], []
    for b in range(1, len(bins)):
        mask = bin_ids == b
        if mask.sum() >= 10:
            bin_means_x.append(preds[mask].mean())
            bin_means_y.append(actuals[mask].mean())
    if bin_means_x:
        smoothed = gaussian_filter1d(bin_means_y, sigma=0.8)
        ax.plot(bin_means_x, smoothed, color=ACCENT, lw=2.5,
                zorder=5, label="Binned mean")
        ax.scatter(bin_means_x, bin_means_y, color=ACCENT,
                   s=28, zorder=6)

    # Metric annotations
    metrics_text = (
        f"Test set metrics\n"
        f"Spearman ρ = {rho_test:.3f}\n"
        f"R² = {r2_test:.3f}\n"
        f"MAE = {mae_test:.3f}"
    )
    ax.text(0.97, 0.05, metrics_text,
            transform=ax.transAxes, fontsize=9.5, color=BLUE,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc=LIGHTGRAY, ec="none"))

    ax.set_xlabel("Predicted success score", fontsize=10.5, color=BLUE)
    ax.set_ylabel("Actual success score", fontsize=10.5, color=BLUE)
    ax.set_title("Predicted vs Actual", fontsize=11, fontweight="bold",
                 color=BLUE, pad=10)
    ax.set_xlim(0, 0.75)
    ax.set_ylim(-0.05, 1.05)
    ax.spines["left"].set_color(LIGHTGRAY)
    ax.spines["bottom"].set_color(LIGHTGRAY)
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    # ── Right: residual distribution ─────────────────────────────────────────
    ax2 = axes[1]
    n, bins2, patches = ax2.hist(resids, bins=40, edgecolor=WHITE,
                                  linewidth=0.4, zorder=3)
    for patch, left in zip(patches, bins2[:-1]):
        if left < -0.15:
            patch.set_facecolor(RED)
            patch.set_alpha(0.75)
        elif left > 0.15:
            patch.set_facecolor(AMBER)
            patch.set_alpha(0.75)
        else:
            patch.set_facecolor(ACCENT)
            patch.set_alpha(0.80)

    ax2.axvline(0, color=BLUE, lw=1.5, linestyle="--", zorder=4)
    ax2.axvline(resids.mean(), color=GREEN, lw=1.5, linestyle=":", zorder=4)
    ax2.text(resids.mean() + 0.01, n.max() * 0.92,
             f"Mean residual\n{resids.mean():.3f}",
             fontsize=8.5, color=GREEN, fontweight="bold")

    # Percentiles
    p10, p90 = np.percentile(resids, 10), np.percentile(resids, 90)
    ax2.text(0.03, 0.97,
             f"80% of predictions within\n[{p10:.2f}, {p90:.2f}] of actual",
             transform=ax2.transAxes, fontsize=9, color=BLUE,
             va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc=LIGHTGRAY, ec="none"))

    ax2.set_xlabel("Residual (predicted − actual)", fontsize=10.5, color=BLUE)
    ax2.set_ylabel("Number of transfers", fontsize=10.5, color=BLUE)
    ax2.set_title("Residual Distribution", fontsize=11,
                  fontweight="bold", color=BLUE, pad=10)
    ax2.spines["left"].set_color(LIGHTGRAY)
    ax2.spines["bottom"].set_color(LIGHTGRAY)
    ax2.tick_params(colors=GRAY, labelsize=9)

    import matplotlib.patches as mpatches
    leg = [mpatches.Patch(color=RED,   alpha=0.75, label="Model over-estimates actual"),
           mpatches.Patch(color=ACCENT, alpha=0.80, label="Near-zero error (±0.15)"),
           mpatches.Patch(color=AMBER, alpha=0.75, label="Model under-estimates actual")]
    ax2.legend(handles=leg, fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout(pad=2.5)
    save(fig, "11_pred_vs_actual.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tactical differentiation — actual model output
# ─────────────────────────────────────────────────────────────────────────────
def fig_tactical_output():
    from model import build_player_features, build_club_features, build_transfer_context
    from scout import apply_filters, Filters, DestinationClub

    players = pd.read_parquet("data/real_players.parquet")
    net, cfg = load_model()

    # Hard filters: MID, age 21-28, budget ≤ €80M, min 900 min
    f = Filters(position="MID", max_value_eur_m=80, min_age=21, max_age=28, min_minutes=900)
    candidates = apply_filters(players, f)

    profiles = {
        "High-Press\n(Klopp-style)": {
            "dest": DestinationClub(ppda=13.2, possession_pct=56.0,
                                    directness_idx=0.42, line_height_m=58.0,
                                    league="Premier League"),
            "color": ACCENT,
        },
        "Possession\n(Guardiola-style)": {
            "dest": DestinationClub(ppda=15.1, possession_pct=65.8,
                                    directness_idx=0.30, line_height_m=62.0,
                                    league="Premier League"),
            "color": PURPLE,
        },
        "Low-Block\n(Counter-attack)": {
            "dest": DestinationClub(ppda=14.5, possession_pct=44.0,
                                    directness_idx=0.52, line_height_m=46.0,
                                    league="Premier League"),
            "color": GREEN,
        },
    }

    TOP_N = 8
    results = {}

    for label, info in profiles.items():
        p_X, _ = build_player_features(candidates, cfg)
        dest_df = pd.DataFrame([{
            "club_id": -1, "league": info["dest"].league,
            "ppda": info["dest"].ppda, "possession_pct": info["dest"].possession_pct,
            "directness_idx": info["dest"].directness_idx,
            "line_height_m": info["dest"].line_height_m,
        }])
        c_X_one, _ = build_club_features(dest_df, cfg)
        c_X = np.tile(c_X_one, (len(candidates), 1))
        ctx_df = pd.DataFrame({
            "transfer_age": candidates["age"].values,
            "fee_eur_m": candidates["market_value_eur_m"].values,
        })
        ctx_X, _ = build_transfer_context(ctx_df, cfg)
        with torch.no_grad():
            scores = net(torch.from_numpy(p_X),
                         torch.from_numpy(c_X),
                         torch.from_numpy(ctx_X)).numpy().flatten()
        ranked = candidates.copy()
        ranked["fit_score"] = (scores * 100).round(1)
        ranked = ranked.sort_values("fit_score", ascending=False).head(TOP_N)
        ranked["surname"] = ranked["player_name"].str.split().str[-1]
        results[label] = ranked

    # Compute overlap
    name_sets = {k: set(v["player_name"]) for k, v in results.items()}
    all_names = set().union(*name_sets.values())
    exclusive_counts = {}
    for k, s in name_sets.items():
        others = set().union(*[v for kk, v in name_sets.items() if kk != k])
        exclusive_counts[k] = len(s - others)
    shared_all = set.intersection(*name_sets.values())
    shared_any_two = set()
    ks = list(name_sets.keys())
    for i in range(len(ks)):
        for j in range(i+1, len(ks)):
            shared_any_two |= name_sets[ks[i]] & name_sets[ks[j]]
    shared_any_two -= shared_all

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05)

    for col_idx, (label, info) in enumerate(profiles.items()):
        ax = fig.add_subplot(gs[col_idx])
        color = info["color"]
        ranked = results[label]

        # Column header
        ax.set_facecolor(WHITE)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, TOP_N + 1.2)
        ax.axis("off")

        # Header bar
        header_rect = plt.Rectangle((0, TOP_N + 0.35), 1, 0.85,
                                     facecolor=color, transform=ax.transData,
                                     clip_on=False)
        ax.add_patch(header_rect)
        ax.text(0.5, TOP_N + 0.78, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=WHITE, transform=ax.transData)

        # Profile metrics
        dest = info["dest"]
        ax.text(0.5, TOP_N + 0.15,
                f"PPDA {dest.ppda:.1f}  ·  Poss {dest.possession_pct:.0f}%  ·  Directness {dest.directness_idx:.2f}",
                ha="center", va="center", fontsize=8, color=GRAY,
                transform=ax.transData)

        # Player rows
        for row_idx, (_, player) in enumerate(ranked.iterrows()):
            y = TOP_N - 1 - row_idx
            pname = player["player_name"]

            # Background
            is_shared_all = pname in shared_all
            is_shared_two = pname in shared_any_two
            bg = "#FEF3C7" if is_shared_two else ("#DCFCE7" if is_shared_all else WHITE)
            row_rect = plt.Rectangle((0.01, y - 0.38), 0.98, 0.76,
                                      facecolor=bg, edgecolor=LIGHTGRAY,
                                      linewidth=0.5, transform=ax.transData)
            ax.add_patch(row_rect)

            # Rank number
            ax.text(0.07, y, f"#{row_idx + 1}", ha="center", va="center",
                    fontsize=9.5, color=color, fontweight="bold",
                    transform=ax.transData)

            # Player name
            ax.text(0.17, y + 0.12, player.get("surname", pname.split()[-1]),
                    ha="left", va="center", fontsize=10.5, color=BLUE,
                    fontweight="bold", transform=ax.transData)

            # League + age
            league_short = {
                "Premier League": "PL", "La Liga": "ESP",
                "Bundesliga": "GER", "Serie A": "ITA", "Ligue 1": "FRA",
            }.get(player.get("league", ""), "")
            ax.text(0.17, y - 0.18,
                    f"{league_short}  ·  Age {int(player['age'])}  ·  €{player['market_value_eur_m']:.0f}M",
                    ha="left", va="center", fontsize=8, color=GRAY,
                    transform=ax.transData)

            # Fit score pill
            pill_color = color
            pill_rect = plt.Rectangle((0.77, y - 0.22), 0.2, 0.44,
                                       facecolor=pill_color, alpha=0.15,
                                       edgecolor=pill_color, linewidth=1.0,
                                       transform=ax.transData)
            ax.add_patch(pill_rect)
            ax.text(0.87, y, f"{player['fit_score']:.0f}",
                    ha="center", va="center", fontsize=10, color=pill_color,
                    fontweight="bold", transform=ax.transData)

    # Legend for overlap colouring
    fig.text(0.5, 0.01,
             "  Yellow background = player appears in 2 of 3 shortlists  "
             "  Green background = player appears in all 3 shortlists  "
             f"  Total unique players shown: {len(all_names)}  "
             f"  Shared across all three: {len(shared_all)}",
             ha="center", fontsize=9, color=GRAY,
             bbox=dict(boxstyle="round,pad=0.4", fc=LIGHTGRAY, ec="none"))

    fig.suptitle(
        "Tactical Differentiation — Actual Model Output for Three Club Profiles\n"
        "Top 8 midfielders (age 21–28, budget ≤ €80M, min 900 min), scored per destination style",
        fontsize=13, fontweight="bold", color=BLUE, y=1.01,
    )
    save(fig, "12_tactical_output.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Label component correlation matrix
# ─────────────────────────────────────────────────────────────────────────────
def fig_label_correlations():
    t = pd.read_parquet("data/real_transfers.parquet").copy()
    fs_raw = (t["minutes_share_y1"] - t["apps_pct_season"]).clip(-0.5, 0.5)
    t["fit_surprise"] = (fs_raw - fs_raw.median() + 0.5)

    labels_nice = {
        "minutes_share_y1":    "Playing time\n(minutes share Y1)",
        "goal_contribution_y1":"Goal contribution\n(G+A/90 Y1)",
        "survival_2y":         "Two-year\nretention",
        "fit_surprise":        "Fit surprise\n(centred)",
        "success_score":       "Success score\n(composite)",
    }
    cols = list(labels_nice.keys())
    n = len(cols)

    # Compute Spearman matrix
    mat = np.full((n, n), np.nan)
    sizes = np.full((n, n), 0)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                mat[i, j] = 1.0
                sizes[i, j] = t[c1].notna().sum()
            else:
                valid = t[[c1, c2]].dropna()
                r, _ = spearmanr(valid[c1], valid[c2])
                mat[i, j] = r
                sizes[i, j] = len(valid)

    fig, ax = plt.subplots(figsize=(10, 8.5))

    # Custom diverging colormap
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-0.1, vcenter=0.3, vmax=0.9)
    cmap = plt.cm.RdYlGn

    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([labels_nice[c] for c in cols], fontsize=9.5, color=BLUE)
    ax.set_yticklabels([labels_nice[c] for c in cols], fontsize=9.5, color=BLUE)

    # Cell annotations
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if np.isnan(val):
                continue
            sz = sizes[i, j]
            bright = norm(val)
            txt_color = WHITE if (bright < 0.3 or bright > 0.8) else BLUE
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=13, color=txt_color, fontweight="bold")
            else:
                ax.text(j, i, f"{val:.3f}\nn={sz:,}",
                        ha="center", va="center", fontsize=8.5,
                        color=txt_color, fontweight="bold" if abs(val) > 0.5 else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Spearman rank correlation", fontsize=9.5, color=BLUE)
    cbar.ax.tick_params(labelsize=8.5, colors=GRAY)

    ax.set_title(
        "Label Component Correlation Matrix\n"
        "Spearman rank correlations between success score components and the composite",
        fontsize=12, fontweight="bold", color=BLUE, pad=14,
    )

    # Key insight annotation
    ax.text(1.22, 0.15,
            "Key findings:\n\n"
            "• Minutes share dominates\n  the composite (ρ = 0.884)\n\n"
            "• Fit surprise is genuinely\n  independent: ρ = 0.201\n  with goal contribution\n\n"
            "• Goal contribution and\n  minutes share correlate\n  (ρ = 0.649) — players\n  who play more score\n  more — but both add\n  signal to the composite",
            transform=ax.transAxes,
            fontsize=9, color=BLUE, va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc=LIGHTGRAY, ec="none"))

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    save(fig, "13_label_correlations.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating supplementary thesis figures...")
    fig_pred_vs_actual()
    fig_label_correlations()
    fig_tactical_output()
    print(f"\nDone. Files in {OUT}/")
