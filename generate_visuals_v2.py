"""
Generate new data-driven presentation visuals — v2.
Outputs 4 PNG files to presentation_assets/.
Run with: .venv/bin/python generate_visuals_v2.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import pickle

warnings.filterwarnings("ignore")

OUT = Path("presentation_assets")
OUT.mkdir(exist_ok=True)

# ── Shared palette ────────────────────────────────────────────────────────────
BLUE      = "#1D3557"
ACCENT    = "#2563EB"
RED       = "#DC2626"
GREEN     = "#16A34A"
AMBER     = "#D97706"
GRAY      = "#64748B"
LIGHTGRAY = "#E2E8F0"
WHITE     = "#FFFFFF"

POS_COLORS = {
    "ATT": "#EF4444",
    "MID": "#3B82F6",
    "DEF": "#10B981",
    "GK":  "#9CA3AF",
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
# 1. Feature signal — permutation importance from the trained model
# ─────────────────────────────────────────────────────────────────────────────
def fig_feature_importance():
    from scipy.stats import spearmanr
    from model import TwoTowerFitModel, build_player_features, build_club_features, build_transfer_context, PLAYER_STAT_COLS

    transfers = pd.read_parquet("data/real_transfers.parquet")
    cfg = pickle.loads(Path("models/feature_config.pkl").read_bytes())
    n_p = cfg.player_scaler.n_features_in_
    n_c = cfg.club_scaler.n_features_in_
    n_x = cfg.ctx_scaler.n_features_in_

    net = TwoTowerFitModel(n_p, n_c, n_x)
    net.load_state_dict(torch.load("models/two_tower.pt", weights_only=True))
    net.eval()

    clubs = pd.read_parquet("data/real_clubs.parquet")

    # Build baseline predictions on training transfers
    # (join club features from destination_club_id)
    clubs_dest = clubs.rename(columns={"club_id": "destination_club_id"})
    # transfers has 'league' = player origin league; drop it before merging club league
    t = transfers.drop(columns=["league"], errors="ignore").merge(
        clubs_dest, on="destination_club_id", how="left"
    ).dropna(subset=["ppda"])

    p_X, _  = build_player_features(t, cfg)
    c_X, _  = build_club_features(t, cfg)
    ctx_df  = pd.DataFrame({"transfer_age": t["transfer_age"].values,
                             "fee_eur_m": (t["fee_eur_m"] / 1e6).values})
    ctx_X, _ = build_transfer_context(ctx_df, cfg)

    with torch.no_grad():
        base_preds = net(
            torch.from_numpy(p_X),
            torch.from_numpy(c_X),
            torch.from_numpy(ctx_X)
        ).numpy().flatten()

    base_spear, _ = spearmanr(base_preds, t["success_score"].values)

    # Permutation importance: shuffle each player feature column, measure drop
    feature_names = (
        ["goals_p90", "assists_p90", "yellow_cards_p90",
         "avg_min_per_game", "apps_pct_season",
         "npxg_p90", "key_passes_p90",
         "mv_momentum_12m",
         "injury_days_last_2y", "has_serious_injury"]
        + ["log_value"]
        + ["pos_ATT", "pos_DEF", "pos_MID", "pos_GK"]
    )

    display_names = {
        "goals_p90":          "Goals / 90",
        "assists_p90":        "Assists / 90",
        "yellow_cards_p90":   "Yellow cards / 90",
        "avg_min_per_game":   "Avg minutes / game",
        "apps_pct_season":    "Appearances %",
        "npxg_p90":           "npxG / 90",
        "key_passes_p90":     "Key passes / 90",
        "mv_momentum_12m":    "MV momentum",
        "injury_days_last_2y":"Injury days (2y)",
        "has_serious_injury": "Serious injury flag",
        "log_value":          "Log market value",
        "pos_ATT":            "Position: ATT",
        "pos_DEF":            "Position: DEF",
        "pos_MID":            "Position: MID",
        "pos_GK":             "Position: GK",
    }

    rng = np.random.default_rng(42)
    importances = {}
    n_repeats = 5

    for i, fname in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            p_X_perm = p_X.copy()
            p_X_perm[:, i] = rng.permutation(p_X_perm[:, i])
            with torch.no_grad():
                preds_perm = net(
                    torch.from_numpy(p_X_perm),
                    torch.from_numpy(c_X),
                    torch.from_numpy(ctx_X)
                ).numpy().flatten()
            s, _ = spearmanr(preds_perm, t["success_score"].values)
            drops.append(base_spear - s)
        importances[fname] = np.mean(drops)

    imp_df = pd.DataFrame({
        "feature": list(importances.keys()),
        "importance": list(importances.values()),
        "label": [display_names[k] for k in importances.keys()],
    }).sort_values("importance", ascending=True)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))

    # Continuous color gradient: navy for high importance, light gray for near-zero
    max_imp = imp_df["importance"].max()
    norm_imp = (imp_df["importance"] / max_imp).clip(0, 1)
    bar_colors = [
        # interpolate from LIGHTGRAY (#E2E8F0) to ACCENT (#2563EB)
        "#{:02x}{:02x}{:02x}".format(
            int(0xE2 + (0x25 - 0xE2) * v),
            int(0xE8 + (0x63 - 0xE8) * v),
            int(0xF0 + (0xEB - 0xF0) * v),
        )
        for v in norm_imp
    ]

    bars = ax.barh(imp_df["label"], imp_df["importance"],
                   color=bar_colors, height=0.65, edgecolor="none", zorder=3)

    # Value labels INSIDE the bar for large ones; omit for noise-level features
    x_max = imp_df["importance"].max()
    ax.set_xlim(0, x_max * 1.08)   # tight x-axis — no empty space

    for bar, val, norm_v in zip(bars, imp_df["importance"], norm_imp):
        if val >= 0.005:
            # label inside the bar (white text on dark bar)
            ax.text(
                val - x_max * 0.004,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="right", fontsize=8.5,
                color=WHITE if norm_v > 0.4 else BLUE,
                fontweight="bold",
            )
        elif val >= 0.001:
            # tiny label just to the right of the bar end (gray, small)
            ax.text(
                val + x_max * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="left", fontsize=7.5, color=GRAY,
            )

    # Separator line between "material" and "noise-level" features
    # Find the rank of the last feature above 0.005
    cutoff_rank = (imp_df["importance"] >= 0.005).sum() - 0.5
    ax.axhline(cutoff_rank, color=LIGHTGRAY, lw=1.2, linestyle="--", zorder=2)
    ax.text(x_max * 0.55, cutoff_rank + 0.25,
            "Below this line: near-zero importance",
            fontsize=8, color=GRAY, style="italic")

    # Key insight callout for the top feature
    top_row = imp_df.iloc[-1]
    ax.annotate(
        "Reliability at previous club\npredicts transfer success\nmore than goals or assists",
        xy=(top_row["importance"], len(imp_df) - 1),
        xytext=(x_max * 0.52, len(imp_df) - 1.8),
        fontsize=9, color=NAVY if "NAVY" in dir() else BLUE,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.3,
                        connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.3", fc=LIGHTGRAY, ec="none"),
    )

    ax.set_xlabel(
        "Mean drop in Spearman ρ when feature values are randomly shuffled",
        fontsize=10, color=BLUE, labelpad=8,
    )
    ax.set_title(
        "Player Feature Importance — Permutation Method",
        fontsize=13, fontweight="bold", color=BLUE, pad=12,
    )

    ax.spines["left"].set_color(LIGHTGRAY)
    ax.spines["bottom"].set_color(LIGHTGRAY)
    ax.tick_params(axis="y", labelsize=10, colors=BLUE)
    ax.tick_params(axis="x", labelsize=9, colors=GRAY)

    fig.tight_layout()
    save(fig, "07_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Player embedding space — t-SNE (improved labels)
# ─────────────────────────────────────────────────────────────────────────────
def fig_embedding_space():
    from adjustText import adjust_text

    coords = np.load("presentation_assets/tsne_coords.npy")
    meta   = pd.read_parquet("presentation_assets/tsne_meta.parquet")

    # Curated set: well-known names that are well-spread in the t-SNE space.
    # Verified coordinates (no two within ~7 units of each other).
    notable = [
        # ATT cluster (top-left)
        ("Kylian Mbappé",          "Mbappé"),
        ("Erling Haaland",         "Haaland"),
        # MID attacking (top-centre)
        ("Jude Bellingham",        "Bellingham"),
        ("Pedri",                  "Pedri"),
        # MID deep (centre)
        ("Rodri",                  "Rodri"),
        ("Kevin De Bruyne",        "De Bruyne"),
        # DEF (bottom-left)
        ("Virgil van Dijk",        "van Dijk"),
        # DEF technical / wing-back (centre-bottom)
        ("Trent Alexander-Arnold", "Alexander-Arnold"),
        ("Kieran Trippier",        "Trippier"),
        # GK (far right)
        ("Alisson",                "Alisson"),
        ("Thibaut Courtois",       "Courtois"),
    ]

    name_col = "name"
    found = []   # (x, y, label, pos_key)
    for fullname, shortname in notable:
        # search by last token of full name
        key = fullname.split()[-1]
        matches = meta[meta[name_col].str.contains(key, case=False, na=False)]
        if len(matches):
            idx = matches.index[0]
            found.append((
                float(coords[idx, 0]),
                float(coords[idx, 1]),
                shortname,
                meta.loc[idx, "pos"],
            ))

    fig, ax = plt.subplots(figsize=(13, 9))

    # Background scatter — all players
    for pos, color in POS_COLORS.items():
        mask = meta["pos"] == pos
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, alpha=0.40, s=14, linewidths=0,
                   label=pos, zorder=2, rasterized=True)

    # Highlighted star markers for labelled players
    texts = []
    for (x, y, label, pos_key) in found:
        col = POS_COLORS.get(pos_key, GRAY)
        # Bold gold star marker
        ax.scatter([x], [y], marker="*", s=220, color="gold",
                   edgecolors=col, linewidths=1.2, zorder=5)
        # Text object — adjustText will reposition these
        txt = ax.text(
            x, y, label,
            fontsize=9.5, fontweight="bold", color=col,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=col,
                linewidth=0.8,
                alpha=0.92,
            ),
            zorder=6,
        )
        texts.append(txt)

    # Auto-adjust label positions to eliminate overlap
    adjust_text(
        texts,
        x=coords[:, 0], y=coords[:, 1],
        ax=ax,
        expand=(1.6, 1.8),
        arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8),
        force_text=(0.5, 0.8),
        force_points=(0.2, 0.3),
        lim=300,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "Learned Player Embedding Space — 2,700 Big-Five Players\n"
        "t-SNE projection of 32-dimensional player tower outputs",
        fontsize=13, fontweight="bold", color=BLUE, pad=14,
    )

    legend = ax.legend(
        title="Position", title_fontsize=10, fontsize=10,
        frameon=True, framealpha=0.9, edgecolor=LIGHTGRAY,
        loc="lower left", markerscale=2.0,
    )
    for t_ in legend.get_texts():
        t_.set_color(BLUE)
    legend.get_title().set_color(BLUE)

    ax.text(
        0.01, 0.01,
        "Proximity = similar playing profile as learned from transfer outcomes, "
        "not from raw statistics similarity.   ★ = labelled player.",
        transform=ax.transAxes, fontsize=8, color=GRAY, style="italic",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    save(fig, "08_embedding_space.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transfer success heatmap — age bucket × fee tier
# ─────────────────────────────────────────────────────────────────────────────
def fig_success_heatmap():
    t = pd.read_parquet("data/real_transfers.parquet").copy()
    t["fee_m"] = t["fee_eur_m"] / 1e6

    age_bins   = [17, 21, 24, 27, 30, 38]
    age_labels = ["18–21", "22–24", "25–27", "28–30", "31+"]
    fee_bins   = [-0.01, 0.5, 5, 20, 50, 200]
    fee_labels = ["Free", "< €5M", "€5–20M", "€20–50M", "€50M+"]

    t["age_bucket"] = pd.cut(t["transfer_age"], bins=age_bins, labels=age_labels)
    t["fee_bucket"] = pd.cut(t["fee_m"], bins=fee_bins, labels=fee_labels)

    pivot = t.groupby(["age_bucket", "fee_bucket"], observed=True)["success_score"].agg(
        ["mean", "count"]
    ).reset_index()

    heatmap_mean = pivot.pivot(index="fee_bucket", columns="age_bucket", values="mean")
    heatmap_n    = pivot.pivot(index="fee_bucket", columns="age_bucket", values="count")

    # Reorder
    heatmap_mean = heatmap_mean.reindex(index=fee_labels, columns=age_labels)
    heatmap_n    = heatmap_n.reindex(index=fee_labels, columns=age_labels)

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(heatmap_mean.values, cmap=cmap, aspect="auto",
                   vmin=0.15, vmax=0.60)

    ax.set_xticks(range(len(age_labels)))
    ax.set_yticks(range(len(fee_labels)))
    ax.set_xticklabels(age_labels, fontsize=10.5, color=BLUE)
    ax.set_yticklabels(fee_labels, fontsize=10.5, color=BLUE)

    # Cell annotations
    for i in range(len(fee_labels)):
        for j in range(len(age_labels)):
            val  = heatmap_mean.values[i, j]
            n    = heatmap_n.values[i, j]
            if np.isnan(val):
                continue
            brightness = (val - 0.15) / (0.60 - 0.15)
            txt_color  = WHITE if brightness < 0.45 or brightness > 0.85 else BLUE
            ax.text(j, i, f"{val:.2f}\n(n={int(n) if not np.isnan(n) else 0})",
                    ha="center", va="center", fontsize=8.5,
                    color=txt_color, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean success score", fontsize=9, color=BLUE)
    cbar.ax.tick_params(labelsize=8, colors=GRAY)

    ax.set_xlabel("Transfer age", fontsize=11, color=BLUE, labelpad=10)
    ax.set_ylabel("Transfer fee", fontsize=11, color=BLUE, labelpad=10)
    ax.set_title(
        "Where Do Successful Transfers Come From?\n"
        "Mean success score by age bracket and fee tier  —  n = 3,766 transfers",
        fontsize=12, fontweight="bold", color=BLUE, pad=14
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    save(fig, "09_success_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. fit_surprise signal validation
# ─────────────────────────────────────────────────────────────────────────────
def fig_fit_surprise():
    from scipy.stats import spearmanr
    from scipy.ndimage import gaussian_filter1d

    t = pd.read_parquet("data/real_transfers.parquet").copy()

    # Reconstruct fit_surprise from available columns
    # fit_surprise = actual_minutes - expected (2-season rolling avg)
    # We proxy this using: minutes_share_y1 - apps_pct_season (pre-transfer apps %)
    # Both are proportions in [0,1], so the difference is meaningful
    t["fit_surprise_proxy"] = t["minutes_share_y1"] - t["apps_pct_season"]

    # Centre on the median of the population (as done in real_data.py)
    median_fs = t["fit_surprise_proxy"].median()
    t["fit_surprise_c"] = (t["fit_surprise_proxy"] - median_fs).clip(-0.5, 0.5)

    valid = t.dropna(subset=["fit_surprise_c", "success_score"])
    rho, pval = spearmanr(valid["fit_surprise_c"], valid["success_score"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: scatter with trend ───────────────────────────────────────────
    ax = axes[0]

    # Hex-bin density to avoid overplotting
    hb = ax.hexbin(valid["fit_surprise_c"], valid["success_score"],
                   gridsize=35, cmap="Blues", mincnt=1, alpha=0.85, zorder=3)
    cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Transfer count", fontsize=8, color=GRAY)
    cb.ax.tick_params(labelsize=7.5, colors=GRAY)

    # Binned mean trend line
    bins = np.linspace(-0.5, 0.5, 16)
    valid["fs_bin"] = pd.cut(valid["fit_surprise_c"], bins=bins)
    trend = valid.groupby("fs_bin", observed=True)["success_score"].mean()
    bin_centers = [(iv.left + iv.right) / 2 for iv in trend.index]
    smoothed = gaussian_filter1d(trend.values, sigma=1.0)
    ax.plot(bin_centers, smoothed, color=ACCENT, lw=2.5, zorder=5, label="Binned mean")
    ax.scatter(bin_centers, trend.values, color=ACCENT, s=30, zorder=6)

    ax.axvline(0, color=LIGHTGRAY, lw=1, linestyle="--", zorder=2)
    ax.set_xlabel("Fit surprise (centred)\nActual minutes − expected, centred on population",
                  fontsize=9.5, color=BLUE)
    ax.set_ylabel("Transfer success score", fontsize=9.5, color=BLUE)
    ax.set_title(f"Fit Surprise vs Transfer Success\nSpearman rho = {rho:.3f}  (n = {len(valid):,})",
                 fontsize=11, fontweight="bold", color=BLUE, pad=10)
    ax.spines["left"].set_color(LIGHTGRAY)
    ax.spines["bottom"].set_color(LIGHTGRAY)
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.legend(fontsize=8.5, frameon=False)

    # ── Right: distribution of success by fit_surprise tercile ────────────
    ax2 = axes[1]

    terciles = pd.qcut(valid["fit_surprise_c"], 3,
                       labels=["Bottom third\n(played less\nthan expected)",
                                "Middle third",
                                "Top third\n(played more\nthan expected)"])
    valid_copy = valid.copy()
    valid_copy["tercile"] = terciles

    means = valid_copy.groupby("tercile", observed=True)["success_score"].mean()
    stds  = valid_copy.groupby("tercile", observed=True)["success_score"].std()
    ns    = valid_copy.groupby("tercile", observed=True)["success_score"].count()

    bar_colors = [RED, GRAY, GREEN]
    bars = ax2.bar(range(3), means.values, yerr=stds.values / np.sqrt(ns.values),
                   color=bar_colors, width=0.55, edgecolor="none",
                   capsize=5, error_kw=dict(ecolor=GRAY, lw=1.5), zorder=3)

    for i, (bar, mean_val, n) in enumerate(zip(bars, means.values, ns.values)):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.025,
                 f"{mean_val:.3f}\n(n={n:,})",
                 ha="center", va="bottom", fontsize=9, color=BLUE,
                 fontweight="bold")

    ax2.set_xticks(range(3))
    ax2.set_xticklabels(means.index, fontsize=9, color=BLUE)
    ax2.set_ylim(0, 0.65)
    ax2.set_ylabel("Mean success score", fontsize=9.5, color=BLUE)
    ax2.set_title("Success Score by Fit Surprise Tercile\n"
                  "Players who played more than expected outperform",
                  fontsize=11, fontweight="bold", color=BLUE, pad=10)
    ax2.spines["left"].set_color(LIGHTGRAY)
    ax2.spines["bottom"].set_color(LIGHTGRAY)
    ax2.tick_params(colors=GRAY, labelsize=9)

    fig.tight_layout(pad=3.0)
    save(fig, "10_fit_surprise.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating v2 presentation visuals...")
    fig_feature_importance()
    fig_embedding_space()
    fig_success_heatmap()
    fig_fit_surprise()
    print(f"\nAll done. Files written to: {OUT}/")
