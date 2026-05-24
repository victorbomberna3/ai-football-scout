# AI Football Scout v3 — supervised fit prediction

**v2 → v3 in one line**: replace cosine-similarity-of-text with a two-tower
neural net trained on transfer outcomes. The football intelligence now lives
in the model, not in Claude prompts.

## Why v3 exists

v2 routed the actual reasoning through Claude:
1. Philosophy text → Claude → metric weights
2. Player stats → MiniLM → text embedding
3. Cosine similarity → ranking
4. Top-20 → Claude → rationale

The problem isn't using LLMs. It's that the LLM is the only thing in the
pipeline that knows football. Strip Claude out and you have z-scores + a
generic English embedding model. That's why the result felt thin.

v3 inverts this: train a real model that predicts *transfer success*
conditional on (player profile, destination club style). The LLM keeps its
job — generating the human-readable rationale at the end — but it no longer
does the work of deciding who fits where. That's now a learned function with
test-set R² you can defend.

## Results on synthetic validation data

The synthetic dataset (see `synthetic.py`) encodes a known underlying signal:
player latent traits × club latent style → transfer success, plus age/fee
plausibility. This isolates "does the architecture work in principle" from
"does real-world data carry enough signal" — two failure modes that get
conflated when you start from real data.

Run `python train.py` and you get:

| model                          | MSE    | MAE    | R²     | Spearman ρ |
| ------------------------------ | ------ | ------ | ------ | ---------- |
| cosine (v2 emulation)          | 0.0547 | 0.2011 | 0.014  | 0.131      |
| linear (ridge)                 | 0.0529 | 0.1947 | 0.046  | 0.213      |
| LightGBM                       | 0.0201 | 0.1133 | 0.638  | 0.789      |
| **two-tower neural net**       | 0.0181 | 0.1072 | **0.673** | **0.810**  |

The 1% → 67% R² jump between cosine and two-tower is the entire pitch for
v3. Cosine throws away 66 percentage points of explained variance because
it has no mechanism for interactions — pressing player × pressing club is
exactly the kind of structure flat vector similarity can't represent.

## Style differentiation check

Same 246 candidate forwards, three destination club styles:

| comparison        | top-10 overlap | rank correlation |
| ----------------- | -------------: | ---------------: |
| Klopp ∩ Pep       | 3/10           | +0.76            |
| Klopp ∩ Atlético  | 0/10           | −0.74            |
| Pep ∩ Atlético    | 0/10           | **−0.96**        |

The model treats Pep's and Atléti's player needs as almost perfectly inverse.
A v2 cosine system would produce near-identical rankings across all three
because the candidate features barely change between queries.

## Architecture

```
┌──────────────────┐                    ┌──────────────────┐
│ player features  │                    │ club features    │
│ (per-90 stats,   │                    │ (ppda, poss%,    │
│  age, value,     │                    │  directness,     │
│  position, lg)   │                    │  line height,    │
└────────┬─────────┘                    │  league)         │
         │                              └────────┬─────────┘
         ▼                                       ▼
   ┌───────────┐                          ┌───────────┐
   │ player    │  ───── 16-d player ─┐    │ club      │
   │ tower MLP │       embedding     │    │ tower MLP │
   └───────────┘                     │    └─────┬─────┘
                                     │          │ 16-d club
                                     │          ▼ embedding
                                     │   ┌──────────────┐
                                     └──►│  concat +    │
                                         │  transfer    │◄── fee, age
                                         │  context     │
                                         └──────┬───────┘
                                                ▼
                                         ┌──────────────┐
                                         │  head MLP    │
                                         │  → sigmoid   │
                                         └──────┬───────┘
                                                ▼
                                       fit score ∈ [0,1]
                                       (predicted transfer
                                        success)
```

- **Player tower**: 22 features → 32 → 16. ~1k params.
- **Club tower**: 9 features → 16 → 16. ~400 params.
- **Head**: 34 features → 32 → 32 → 1. ~2k params.
- Total ~4k params. Tiny by ML standards, correct for 8k training transfers.

The architectural prior — separate towers — gives you two things a GBM can't:
1. A learned player embedding space, usable standalone for "find similar
   players" without retraining.
2. Scalable adaptation: add a new club to the directory and you don't need to
   retrain, just compute its tower output.

## Files

```
scout_v3/
├── synthetic.py     ← validation dataset generator (replace with real ETL)
├── model.py         ← feature pipeline + two-tower model + training loop
├── baselines.py     ← cosine, linear, GBM for honest comparison
├── train.py         ← run end-to-end: generate → fit all → save artifacts
├── scout.py         ← drop-in replacement for v2's scoring.py
└── README.md        ← this file
```

## Plugging in real data

`synthetic.py` produces three dataframes the rest of the pipeline consumes:

- `players` — one row per player-season, columns include `player_id`,
  `position_group`, `age`, `minutes`, `league`, `market_value_eur_m`, and
  the per-90 stats listed in `model.PLAYER_STAT_COLS`.
- `clubs` — one row per club-season, columns `club_id`, `league`, `ppda`,
  `possession_pct`, `directness_idx`, `line_height_m`.
- `transfers` — one row per transfer, columns `player_id`,
  `destination_club_id`, `transfer_age`, `fee_eur_m`, `destination_league`,
  plus the outcome columns `minutes_share_y1`, `value_delta_18m`,
  `survival_2y`, `success_score`.

To switch to real data: write a new module (call it `real_data.py`) that
produces dataframes with those exact columns, then import from it in
`train.py` instead of `synthetic.generate()`. Everything downstream is
schema-agnostic.

### Real data sources & labels

**Players & clubs**: you already have the FBref + Transfermarkt pipeline
from v2. Extend it:
- Players: add a `season` column, pull multi-season data (FBref's URL
  pattern supports `/season/2023-2024`, etc).
- Clubs: FBref's team stats pages have all four required columns. PPDA is
  derived from passes per defensive action (pass attempts allowed / sum of
  tackles+interceptions+fouls in opp half).

**Transfers & outcomes**:
- Transfermarkt has the raw transfer events table in their public R2 bucket
  (`dcaribou/transfermarkt-datasets`). It's already in your v2 pipeline.
- Outcomes need joining the *next* season's player stats:
  - `minutes_share_y1` = (player minutes at new club, season N+1) / (max
    possible league minutes that season).
  - `value_delta_18m` = (TM value 18 months later) / (TM value at transfer)
    − 1. TM publishes valuation history.
  - `survival_2y` = 1 if player still at same club 24 months after transfer
    (or transferred for higher fee), else 0.

That gets you a labelled dataset of ~30k transfers across Big-5 leagues from
the last 10 seasons. Probably ~15k after filtering for usable outcome
labels (need full season N+1 stats, value history, etc).

### Labels are noisy

A failed transfer can be down to dressing-room politics, injuries, coach
changes — none of which the model sees. Two ways to handle this:

1. **Drop the worst confounders**: exclude transfers where the destination
   manager changed within 6 months, where the player had a >2-month injury
   in season N+1, etc. Cuts data ~30% but cleans labels significantly.
2. **Multi-task learning**: predict each outcome separately
   (minutes, value, survival) and weight them. If one is dominated by noise,
   the others still inform the shared encoders.

Honest framing for the thesis: the model captures the *systemic* portion of
transfer success — how well a player's profile complements a club's tactical
identity. It doesn't and can't predict the political/personal portion.

## Use in the Streamlit app

`scout.py` exports the same `Filters` dataclass + `apply_filters` /
`compute_fit_score` API as v2. In `app.py`, change:

```python
from scoring import Filters, apply_filters, compute_fit_score
```

to:

```python
from scout import Filters, apply_filters, compute_fit_score, DestinationClub
```

The UI change: instead of five sliders → Claude → weights, the user picks /
configures a `DestinationClub` (their own club's ppda, possession, etc.).
For Big-5 clubs you can pre-fill these from team-stat tables; for custom
queries, four sliders.

Keep Claude for: (a) rationale generation per shortlisted player, (b) parsing
free-text destination descriptions ("a team like Brighton") into a
`DestinationClub` object. That's legitimate LLM use — natural language as
input/output interface, not as the reasoning engine.

## What this needs next (ranked by impact)

1. **Real data backfill.** Until trained on actual transfer outcomes, all
   the numbers above are validation, not validation-set-on-real-task. This
   is the gating step.
2. **Manager features.** Two clubs with identical ppda/possession can play
   very differently. Add a one-hot manager identity feature (or learned
   manager embedding from career trajectory). Probably +5pp R².
3. **Multi-season player history.** Right now player features are
   single-season. Add 3-year rolling means + trend slopes. Catches both
   "consistent" vs "one-season wonder" and "improving" vs "plateaued".
4. **Action-sequence features.** StatsBomb open-data competitions
   (World Cup, Euros, NWSL, some leagues) include full event logs. Encode
   each player as a distribution over action n-grams. This is where the
   project becomes genuinely defensible vs. anyone scraping FBref.
5. **Multi-output head.** Predict minutes, value delta, survival
   separately; let the user weight what they care about (a top-6 club cares
   more about minutes share; a selling club cares about value delta).
6. **Calibration on held-out seasons.** Train through 2022-23, test on
   2023-24 to get a realistic estimate of out-of-time performance. Current
   random-split numbers are upper bounds.

## What you can throw away from v2

- `embeddings.py` (the MiniLM cosine system).
- `philosophy.py` → `llm_weights` Claude call. Keep `philosophy_to_text`
  only if you want LLM-mediated free-text destination parsing as a fallback.

## Honest assessment of where this stands

This is now a real ML project — supervised, evaluable, with a defensible
architectural choice and quantified gains over honest baselines. It is not
yet a real *product* because the real-data training run hasn't happened.
The synthetic results prove the pipeline works; they don't prove the signal
exists at the strength shown above in actual transfer data.

The honest next two weeks of work: backfill the data pipeline to produce
labelled transfers, retrain, see what R² survives. If it lands at 0.30-0.50
on real data, that's a genuinely useful tool and a strong thesis chapter.
If it lands at 0.10, the labels are too noisy and you need richer features
(probably tracking data, which costs).
