"""
End-to-end training and comparison.

Run: python train.py [--synthetic]

Defaults to real data (cached under data/); pass --synthetic for the
original synthetic baseline.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import os
# Prevent OpenMP deadlock between Intel libiomp5 (torch) and LLVM libomp (lightgbm).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
# lightgbm must still be imported before torch so libomp loads first.
from baselines import evaluate_baselines
import torch
torch.set_num_threads(1)

from model import assemble_training_arrays, train_two_tower

SEED = 0
OUT = Path("models")

USE_SYNTHETIC = "--synthetic" in sys.argv


def main():
    print("=" * 60)
    print("AI FOOTBALL SCOUT v3 — TRAINING & EVAL")
    print("=" * 60)

    # 1. Data
    if USE_SYNTHETIC:
        from synthetic import generate
        print("\n[1] Generating synthetic dataset…")
        data = generate(seed=42)
    else:
        from real_data import generate
        print("\n[1] Loading real dataset (from cache)…")
        data = generate(save_dir="data")
    print(f"    players={len(data['players']):,}  clubs={len(data['clubs']):,}  "
          f"transfers={len(data['transfers']):,}")

    # 2. Features
    print("\n[2] Building feature arrays…")
    if "n_apps" in data["transfers"].columns:
        transfers_clean = data["transfers"][data["transfers"]["n_apps"] > 0].reset_index(drop=True)
    else:
        transfers_clean = data["transfers"].reset_index(drop=True)
    players_train = data["players"]
    print(f"    transfers after quality filter (n_apps>0): {len(transfers_clean):,} / {len(data['transfers']):,}")
    p_X, c_X, ctx_X, y, cfg = assemble_training_arrays(
        players_train, data["clubs"], transfers_clean
    )
    print(f"    player feats: {p_X.shape[1]} | club feats: {c_X.shape[1]} | "
          f"ctx feats: {ctx_X.shape[1]}")

    # 3. Baselines
    print("\n[3] Training baselines (cosine, linear, GBM)…")
    base = evaluate_baselines(p_X, c_X, ctx_X, y, seed=SEED)

    # 4. Two-tower
    print("\n[4] Training two-tower model…")
    model, tt_metrics = train_two_tower(p_X, c_X, ctx_X, y, seed=SEED, verbose=True)

    # 5. Report
    print("\n" + "=" * 60)
    print("RESULTS (test set)")
    print("=" * 60)
    rows = []
    for name in ["cosine", "linear", "gbm"]:
        m = base[name]
        rows.append((name, m["test_mse"], m["test_mae"], m["test_r2"], m["test_spearman"]))
    rows.append(("two_tower", tt_metrics["test_mse"], tt_metrics["test_mae"],
                 tt_metrics["test_r2"], tt_metrics["test_spearman"]))

    print(f"\n{'model':<14} {'MSE':>8} {'MAE':>8} {'R²':>8} {'Spearman ρ':>12}")
    print("-" * 54)
    for name, mse, mae, r2, sp in rows:
        print(f"{name:<14} {mse:>8.4f} {mae:>8.4f} {r2:>8.4f} {sp:>12.4f}")

    # 6. Save
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "two_tower.pt")
    with open(OUT / "feature_config.pkl", "wb") as f:
        pickle.dump(cfg, f)
    metrics_summary = {
        "two_tower": {k: v for k, v in tt_metrics.items() if k not in ("history", "test_pred", "test_y", "test_idx")},
        "baselines": {k: {kk: vv for kk, vv in base[k].items() if kk not in ("model", "test_pred")}
                      for k in ["cosine", "linear", "gbm"]},
    }
    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\n[6] Saved: {OUT}/two_tower.pt, feature_config.pkl, metrics.json")

    # 7. Interpretation
    cos_r2 = base["cosine"]["test_r2"]
    lin_r2 = base["linear"]["test_r2"]
    gbm_r2 = base["gbm"]["test_r2"]
    tt_r2 = tt_metrics["test_r2"]
    print("\n" + "=" * 60)
    print("READ-OUT")
    print("=" * 60)
    print(f"Cosine (v2 emulation): R² = {cos_r2:.3f}")
    print(f"  → captures ~{max(cos_r2, 0)*100:.0f}% of variance. The rest is")
    print(f"    interaction structure that flat-vector similarity cannot reach.")
    print()
    print(f"Linear:                R² = {lin_r2:.3f}")
    print(f"  → main effects only. Gap vs GBM/two-tower = interaction signal.")
    print()
    print(f"GBM (LightGBM):        R² = {gbm_r2:.3f}")
    print(f"  → strong off-the-shelf benchmark. If two-tower beats this, the")
    print(f"    architectural prior (separate player/club encoders) earns its keep.")
    print()
    print(f"Two-tower:             R² = {tt_r2:.3f}")
    if tt_r2 > gbm_r2 + 0.01:
        print(f"  → BEATS GBM by {(tt_r2-gbm_r2)*100:.1f} pp. The two-tower prior helps.")
    elif tt_r2 > gbm_r2 - 0.02:
        print(f"  → COMPARABLE to GBM. Pick based on side-benefits (embeddings).")
    else:
        print(f"  → UNDERPERFORMS GBM by {(gbm_r2-tt_r2)*100:.1f} pp. Either tune more,")
        print(f"    or use GBM for scoring and a separate model for embeddings.")


if __name__ == "__main__":
    main()
