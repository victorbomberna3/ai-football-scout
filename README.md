# AI Football Scout

A data-driven scouting tool that ranks players by **transfer fit** — how well a player's profile matches a destination club's tactical identity — using a two-tower neural network trained on historical transfer outcomes.

Built for sporting directors, analysts and scouts who want a quantitative first filter before manual review.

---

## What it does

Most scouting tools rank players by raw performance stats. This tool ranks by **fit**: a player who excels in a high-press, possession-based system may be a poor fit for a low-block counter-attacking club even if their headline numbers look identical.

The model learns this from ~8,000 labelled historical transfers across the Big-5 leagues (2018–2024), where "success" is a composite of minutes played, market value development and contract survival at the new club.

**Key outputs:**
- Ranked shortlist of candidates by predicted transfer fit (0–100 score)
- Player radar profiles (percentile vs Big-5 peers)
- Market intelligence quadrant — value picks vs premium targets
- Similar player search using the model's learned 32-dimensional player embeddings
- CSV export for further analysis

---

## Scouting interface

```
streamlit run app.py
```

The sidebar lets you configure:

| Filter | What it controls |
|---|---|
| Position | ATT / MID / DEF / GK |
| Age range | Min / max age window |
| Budget | Market value ceiling (€M) |
| Min minutes | Quality filter (only players with real playing time) |
| Club style | PPDA · Possession % · Directness · Line height — or pick a preset |

**Style presets** include Klopp gegenpressing, Pep tiki-taka, Mourinho low-block, counter-attack, and more. Each maps to four tactical parameters that the model uses to score fit.

---

## How the model works

The core idea: **separate encoders for player and club**, then learn their interaction.

```
Player stats (24 features)          Club tactics (9 features)
        │                                     │
  Player Tower                          Club Tower
  24 → 64 → 32-d embedding             9 → 32 → 32-d embedding
        │                                     │
        └──────────────── Head MLP ───────────┘
              input: concat(p_emb ⊙ c_emb,  |p_emb − c_emb|,  context)
                                32-d               32-d            2-d
              ────────────────────────────────────────────────────────────
                                       66-d total
              layers: 66 → 64 → 64 → 1 → sigmoid
                                │
                      Transfer success score (0–1)
```

The 32-dimensional player embedding is reused in the **Similar Players** tab — cosine similarity in this space finds players with comparable style and technical profile, not just similar raw stats.

### Model performance (test set, real Big-5 data)

| Model | R² | MAE | Spearman ρ |
|---|---|---|---|
| Cosine baseline | ~0.00 | 0.19 | ~-0.01 |
| Linear (Ridge) | 0.25 | 0.16 | 0.51 |
| LightGBM | **0.28** | **0.15** | **0.53** |
| Two-tower (this model) | 0.26 | 0.16 | 0.52 |

R² ≈ 0.26–0.28 means the models explain roughly a quarter of variance in real transfer outcomes — strong for this domain given the inherent noise in transfers (injuries, dressing-room dynamics, managerial changes). LightGBM and the two-tower are within 3 pp of each other; the two-tower's main practical advantage is the learned player embedding space used in the Similar Players tab.

---

## Setup

**Requirements**: Python 3.10+

```bash
# Clone the repo
git clone git@github.com:victorbomberna3/ai-football-scout.git
cd ai-football-scout

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option A — Use the pre-trained model (fastest)

The trained model weights and feature config are included in `models/`. You only need to supply the player/club data:

```bash
python train.py        # fetches real data, retrains, saves to models/
streamlit run app.py   # launch the UI
```

### Option B — Synthetic data (no data dependency)

```bash
python train.py --synthetic   # generates synthetic data, trains, evaluates
streamlit run app.py
```

---

## Data

The pipeline uses **Transfermarkt** as its primary data source (Big-5 leagues, 2024–25 season for the player pool; 2018–2024 for transfer labels). Data is fetched and cached locally under `data/` — this directory is excluded from the repo.

Player features include: goals/90, assists/90, progressive passes, progressive carries, take-ons, pass completion %, aerials won, minutes per game, market value, market value momentum (12-month), position group, age.

Club features include: PPDA, possession %, directness index, defensive line height, league.

---

## Project structure

```
ai-football-scout/
├── app.py           — Streamlit UI (4 tabs: Scout · Similar · Data Explorer · Model)
├── scout.py         — Filters, DestinationClub, fit scoring logic
├── model.py         — Two-tower architecture, feature pipeline, training loop
├── real_data.py     — Transfermarkt ETL → player, club and transfer dataframes
├── baselines.py     — Cosine, linear and LightGBM baselines for benchmarking
├── train.py         — End-to-end: load data → train all models → save artifacts
├── synthetic.py     — Synthetic data generator (for testing without real data)
├── models/          — Saved weights (two_tower.pt), feature config, metrics
└── requirements.txt
```

---

## Roadmap

- [ ] Multi-season player history (trend slopes, consistency flags)
- [ ] Manager features (tactical identity beyond club averages)
- [ ] Multi-output prediction head (minutes / value / survival separately)
- [ ] Out-of-time validation (train → 2022-23, test → 2023-24)
- [ ] Action-sequence features from event data (StatsBomb)
