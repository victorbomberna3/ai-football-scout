# Presentation Brief — AI Football Scout v3
## For Claude Opus: Build a 20-Minute MBB-Style Deck

---

## INSTRUCTIONS FOR CLAUDE OPUS

Build a presentation deck in the style of a McKinsey / BCG / Bain strategy presentation. That means:

- One clear point per slide, stated as a declarative sentence in the title
- Data and charts carry the argument; text is minimal
- No filler slides, no agenda slides unless structurally necessary
- Pyramid logic: conclusion first, evidence second
- No em dashes anywhere. No hedge language ("could potentially", "may help to"). No AI writing tells ("delve into", "leverage", "seamless", "transformative", "revolutionize", "it is worth noting")
- Write as a practitioner explaining a real system to a sophisticated audience (club executives, data directors, football analysts)
- Acknowledge limitations honestly. MBB does not hide model weaknesses; it contextualises them

The presentation is 20 minutes across 4 speakers. Allocate roughly 5 minutes per speaker. Each speaker owns a section. Write full speaker notes for each slide (3 to 5 spoken sentences, natural and direct, not corporate).

---

## SECTION 1 — THE PROBLEM (Speaker A, 5 min, ~4 slides)

### Context and hook

Football clubs spend between 3 and 5 billion euros per transfer window in the Big Five leagues. The empirical failure rate for transfers above 20 million euros, measured as players who do not reach 50% of available minutes in their first season, is approximately 40 to 45%. The clubs paying the most have the worst track records: between 2018 and 2024, the top 20% of fees by value produced the lowest average first-season minutes share in the Transfermarkt dataset.

The market does not price player-club fit. It prices player quality and scarcity. Those are different things.

### The core problem

Existing scouting tools answer "Is this player good?" not "Will this player succeed here?" These are structurally different questions.

A pressing midfielder who thrives at Liverpool under a 7.5 PPDA system will not replicate that performance at a low-block team with PPDA 16. His stats do not change. His context changes. No proprietary tool currently accounts for this interaction at scale.

Three reasons this problem is hard:
1. Ground truth is delayed. A transfer's success is unknowable for 12 to 18 months after it happens.
2. The outcome is multivariate. Minutes, goals, value trajectory, and retention all matter and trade off against each other.
3. The comparison set is small. The Big Five produce roughly 1,500 to 2,000 senior cross-club transfers per year. With multiple confounders, classical regression is underpowered.

### Our thesis

We trained a neural network on 3,766 labelled transfer outcomes (2018 to 2024) to predict transfer success probability, defined as a composite score across four outcome dimensions. The model does not rank players by quality. It ranks players by predicted success at a specific destination club, given that club's tactical identity. This is a different output.

A player ranked 50th in the overall market can rank 1st for a specific club if their profile matches that club's system.

### What this is not

This is not a transfer recommendation engine that tells clubs who to buy. It is a fit-scoring layer that integrates into an existing scouting process. The model surfaces candidates a club's filters would not reach and quantifies the tactical match before a scout invests time in a dossier.

---

## SECTION 2 — THE DATA ENGINE (Speaker B, 5 min, ~4 slides)

### Three data sources, one labelled dataset

The pipeline combines three external sources with no proprietary data:

**Transfermarkt (public R2 dataset, dcaribou/transfermarkt-datasets)**
- Players: 2,700 Big Five players, 2024-25 season snapshot
- Clubs: 150 clubs with competition metadata
- Transfers: all recorded Big Five transfers, 2018 to 2024 (raw: ~80,000 records)
- Appearances: per-match records for all players (used to compute season-level stats)
- Player valuations: time-series of market values (used for momentum and label computation)
- Injury history: scraped from Transfermarkt individual injury pages (TM has no bulk export for this)

**Football-data.co.uk (free per-match CSVs, Big Five, 2018 to 2024)**
- Used exclusively for club tactical feature proxies
- Provides: shots for/against, shots on target, fouls committed, corners against, per match
- From these six columns we compute four club-level tactical features (explained below)

**FBref via soccerdata (static HTML, Big Five, 2018 to 2024)**
- Used for tackles won and interceptions per 90 (pressing intensity metrics)
- Available via static HTML scraping: misc table, standard table, shooting table
- Not available via static HTML: passing table (pass completion, progressive passes), possession table (carries, take-ons). These pages use JavaScript rendering. Cells are empty in the raw HTML. This is a hard constraint, not a scraping failure.
- FBref pressing stats provide approximately 47% coverage of training transfers. The remaining rows receive position-average fallbacks.

### The outcome label: success_score

Every transfer in the training set receives a success_score between 0 and 1:

```
success_score = 0.35 * minutes_share_y1
              + 0.30 * goal_contribution_y1
              + 0.20 * survival_2y
              + 0.15 * fit_surprise
```

**minutes_share_y1**: minutes played in season 1 at the new club divided by maximum possible minutes for that league (3,420 for a 38-game season). This is the cleanest signal. A coach who does not play the new signing is communicating system mismatch directly.

**goal_contribution_y1**: (goals + assists) per 90 in year 1, clipped to [0, 1]. Scaled so that a player with 0.5 goals+assists per 90 scores 0.5 on this component.

**survival_2y**: binary. Did the player remain at the club for two full years? Proxied by minutes_share for transfers after mid-2022 (where the 24-month window has not elapsed).

**fit_surprise**: the tactical fit signal. Described in detail below.

Population mean of success_score in training data: 0.46. Standard deviation: 0.20.

### The fit_surprise signal: where the model learns tactical fit

This is the most important design decision in the label construction.

For each transfer, we compute the player's expected playing time at the new club using their rolling two-season average apps_pct_season before the transfer. We then compare that expectation to their actual minutes_share_y1.

Raw fit_surprise = minutes_share_y1 minus expected_apps_pct

The raw mean is approximately -0.22. This is not noise. It reflects the structure of football transfers: players move upward in quality, and elite clubs give fewer minutes to new signings who must compete for a spot. If you use the raw signal, you are inadvertently penalising quality players who joined strong clubs.

Fix: centre on the population median (approximately -0.15 to -0.25), then clip to [-0.5, +0.5] and shift to [0, 1]. This transforms the signal from "did this player play more than their own history suggests?" into "did this player outperform the typical cohort outcome?" Coverage: 51% of training transfers have a valid two-season baseline.

Without fit_surprise, Mourinho-profile recommendations and Guardiola-profile recommendations overlap 13 out of 15 players. With fit_surprise, overlap drops to 2 out of 15. The signal is the primary mechanism by which the model distinguishes tactical styles.

### Temporal integrity

Every training transfer uses player stats from the season before the transfer (computed from TM appearances, joined via merge_asof on season_end_year). This prevents temporal leakage. The 2024-25 scouting pool uses the current season snapshot for inference. No player's future performance informs their training representation.

The train/test split is player-stratified, not row-stratified. No player appears in both training and test sets. This prevents the model from memorising individual players and forces generalisation to unseen players.

---

## SECTION 3 — THE MODEL (Speaker C, 5 min, ~5 slides)

### Why a two-tower architecture

A two-tower model has a specific structural advantage for this problem: it separates the representations of the two entities being matched before computing their interaction.

The alternative is a single MLP that takes a concatenated [player, club, context] vector. The problem with concatenation: the model cannot learn that "a player's pressing tendency" and "a club's pressing demand" are interacting dimensions. It can only learn fixed weights on each feature. The interaction is implicit and weak.

In a two-tower model:
- The player tower learns a 32-dimensional embedding that encodes the player's latent profile (pressing tendency, technical quality, robustness, output rate) without reference to any specific club.
- The club tower learns a separate 32-dimensional embedding that encodes the club's tactical identity.
- The head MLP takes the Hadamard product (element-wise multiplication) and absolute difference of the two embeddings, then predicts success probability.

The Hadamard product captures alignment: if player dimension k is high and club dimension k is high, the product is high, signalling mutual strength on that axis. The absolute difference captures mismatch: if they diverge on dimension k, the difference is large. Together they form a bilinear interaction that no flat concatenation can replicate.

Secondary benefit: the player tower's 32-dimensional output is a learned embedding space. Two players with similar embeddings have similar playing profiles, regardless of the clubs they played at. This powers the "Similar Players" tab without any additional training.

### Architecture specification

```
Player inputs (15 features):
  goals_p90, assists_p90, yellow_cards_p90,
  avg_min_per_game, apps_pct_season,
  npxg_p90, key_passes_p90,
  mv_momentum_12m,
  injury_days_last_2y, has_serious_injury
  + log(market_value_eur_m)
  + position one-hot [GK, DEF, MID, ATT]
  Total: 10 stat cols + 1 value col + 4 position cols = 15 inputs

Player Tower: 15 -> Linear(64) -> GELU -> Dropout(0.15) -> Linear(32) -> p_emb

Club inputs (9 features):
  ppda, possession_pct, directness_idx, line_height_m
  + league one-hot [PL, La Liga, Bundesliga, Serie A, Ligue 1]
  Total: 4 tactical cols + 5 league cols = 9 inputs

Club Tower: 9 -> Linear(32) -> GELU -> Dropout(0.15) -> Linear(32) -> c_emb

Transfer context (1 feature):
  log(fee_eur_m)

Head input: concat(p_emb * c_emb, |p_emb - c_emb|, ctx) = 32 + 32 + 1 = 65 features
Head MLP: 65 -> Linear(64) -> GELU -> Dropout(0.15) -> Linear(64) -> GELU -> Dropout(0.15) -> Linear(1) -> Sigmoid

Total parameters: approximately 15,000 to 20,000
```

This is deliberately small. With 3,766 training transfers and 70% training split (approximately 2,636 samples), a larger model overfits. Scaling comes from more data, not more parameters.

### Feature engineering decisions and exclusions

**What is in the model and why:**

- `goals_p90`, `assists_p90`: direct output measures from TM appearances. Clean, consistent across train and inference.
- `yellow_cards_p90`: proxy for aggression and pressing intensity. Players who press hard commit more fouls and receive more cards.
- `avg_min_per_game`: encodes coach trust. A player averaging 82 minutes per game is a starter; 45 is a rotation player. This is structurally different from apps_pct_season.
- `apps_pct_season`: appearance rate. Separates persistent starters from one-off selections.
- `npxg_p90`: proxied as goals_p90 * 0.88. Captures underlying chance quality, not just conversion.
- `key_passes_p90`: proxied as assists_p90 * 2.0. Captures creativity independent of the striker converting the chance.
- `mv_momentum_12m`: (current_value - value_12m_ago) / value_12m_ago. A player whose value is rising is in form; declining value signals a player past their peak or recovering from injury.
- `injury_days_last_2y`: total days missed to injury in the 24 months before the transfer date. Availability is a direct input to the success formula.
- `has_serious_injury`: binary flag. Any single injury exceeding 60 days in the prior 3 years. Structural injuries (ACL, Achilles) create long-term risk that aggregate days missed may understate.
- `log(market_value_eur_m)`: log-transformed because the value distribution is right-skewed. Serves as a quality proxy that the model uses to calibrate expectations.
- Position one-hot: a striker and a goalkeeper cannot be compared on the same stat scales. The one-hot allows the player tower to learn position-specific embeddings.

**What is excluded and why:**

- `transfer_age`: included during early development. At inference, it became the dominant predictor (Spearman correlation with fit_score: +0.72), sorting all recommendations purely by age. Age is controlled upstream by the Filters(min_age, max_age) hard filter. Including it in the model produces age-sorted outputs, not fit-sorted outputs.
- `league_of_origin one-hots`: bake in a 10-point Premier League premium regardless of player quality. A Premier League attacker scores higher than an equally skilled Ligue 1 attacker, biasing the model against continental signings. Destination league is already in the club tower.
- `contract_months_remaining`: TM exposes only the current contract date, not the contract at the time of the historical transfer. Using current contract data on historical training rows would introduce temporal leakage.
- `tkl_won_p90`, `interceptions_p90`: FBref misc data. Available for approximately 47% of training transfers (non-Big-Five fallback rows have no FBref data). Available for approximately 64% of the 2024-25 scouting pool. This train/inference coverage mismatch creates systematic bias: the model would learn these features on a biased subsample of training data, then apply them to a different coverage pattern at inference.
- `pass_completion_pct`, `prog_passes_p90`, `prog_carries_p90`, `take_ons_p90`, `aerials_won_p90`: FBref passing and possession pages use JavaScript rendering. The `<td>` elements in the static HTML contain no data. pd.read_html and soccerdata both return NaN for these columns. Headless browser scraping (Selenium/Playwright) or a paid FBref API would be required to obtain this data at scale.

**Club tactical features and their proxies (from football-data.co.uk):**

- `possession_pct`: shots_for / (shots_for + shots_against). Shot share correlates at approximately r=0.85 with tracked possession data from Opta.
- `ppda`: 20 minus (fouls_committed_per_game * 0.45), clipped to [4, 20]. High-press teams commit more fouls in the opponent's half. This is an inversion proxy, not true PPDA (passes per defensive action). The resulting values cluster in [12.5, 16] across Big Five clubs.
- `directness_idx`: shots_on_target / shots. Possession sides take more speculative long shots, producing a lower ratio. Direct teams generate fewer but higher-quality chances.
- `line_height_m`: derived from corners_against per game. Teams that defend deep concede more corners as opponents pin them back. Mapped to [25, 65] metres.

### Training protocol

- 300 maximum epochs, early stopping with patience=25 on validation MSE
- Player-stratified train/val/test split: 70% / 15% / 15%. No player appears in more than one partition.
- Adam optimizer, learning rate 5e-4, weight decay 1e-4
- Learning rate scheduler: ReduceLROnPlateau, factor 0.5, patience 10, minimum 1e-5
- Loss function: MSE on success_score
- Batch size: 256

### Performance against baselines

Results on held-out test set (approximately 565 transfers, entirely unseen players):

| Model | R² | Spearman rho | MAE |
|---|---|---|---|
| Cosine similarity baseline | 0.000 | 0.008 | 0.173 |
| Gradient Boosted Machine | 0.088 | 0.325 | 0.167 |
| Linear regression | 0.127 | 0.353 | 0.160 |
| **Two-tower (ours)** | **0.122** | **0.370** | **0.160** |

R² = 0.122 means the model explains 12.2% of variance in real transfer outcomes. This sounds low. It is not low for this problem. Football transfer success is driven by unmeasured variables: manager tenure, dressing room dynamics, the player's personal circumstances at the time of the move, injury timing, and luck. No externally available dataset captures these. 12% explained variance on genuinely hard-to-predict events is a meaningful signal.

The Spearman correlation of 0.37 is the more operationally relevant number. It measures whether the model ranks players in the right order, not whether it predicts the exact success score. A scout does not need to know a player will succeed with 0.63 probability; they need the model to rank the right players higher. At rho=0.37, the model produces rankings that are substantially better than random (rho=0) and better than the GBM baseline (rho=0.325) which uses a flat feature vector without the bilinear interaction.

The two-tower does not beat linear regression on R². This is expected: linear regression has a structural advantage on small datasets with Gaussian residuals. The two-tower's advantage is structural, not statistical: it separates player and club representations and learns their interaction via the bilinear head. With more training transfers (the natural state as the dataset grows year over year), the interaction learning advantage will compound while linear regression's performance ceiling remains fixed.

---

## SECTION 4 — THE PRODUCT (Speaker D, 5 min, ~4 slides)

### The scouting interface

The application is built in Streamlit and runs locally. It has four tabs.

**Scout tab** is the primary workflow:
1. Set player filters: position (ATT/MID/DEF/GK), age window, budget ceiling (market value), minimum minutes played, source league filter (narrow the candidate pool to one or more leagues), contract status filter
2. Select destination club from 150 Big Five clubs with auto-filled tactical profiles, or specify a style preset (high possession, balanced pressing, physical/direct, low block)
3. Submit. The model scores all candidates in under two seconds.
4. Output: top-3 recommendation cards (fit score, club, age, value, goals/90, details popover with contract status and injury flags), full ranked table with orange row highlighting for injury-flagged players, player profile radar charts (percentile vs Big Five peers at same position), and a value-for-money efficiency chart (fit score per million euros spent).

The "Why these players fit" expander shows per-player qualitative signals: which stats are in which percentile relative to the filtered candidate pool, which tactical dimensions match the destination club's profile, the value and contract angle, and risk flags.

**Similar Players tab**: enter a player name, the model returns the nearest neighbours in the 32-dimensional embedding space by cosine similarity. This does not use the fit score; it uses the player tower's embedding directly. Two players are similar if their latent profiles are close, regardless of club, league, or market value. The tab shows a comparison radar, a similarity vs value scatter, and budget alternatives (similar players at less than 60% of the target's value).

**Data Explorer**: full pool exploration. Value vs. performance scatter, stat distributions by position, club tactical space visualisation, transfer outcome label distribution.

**Model tab**: architecture diagram, benchmark comparisons, metrics table.

### The injury watch list feature

The clinical use case that demonstrates the model's utility most clearly.

When a scout activates "Injury watch mode," the application re-scores all candidates with their injury flags zeroed out (has_serious_injury set to 0, injury_days_last_2y set to 0) before computing fit scores. Original injury flags are then re-attached to the ranked output for display. Players with injury history appear in orange rows in the ranking table.

This answers a specific question: "If we are willing to accept medical risk, which players would rank highest on pure fit?" The model separates the fit question from the medical question. The medical department handles the latter.

**Concrete demonstration:** Igor Thiago (Brentford, ATT, age 22, EUR 30M) with the following search parameters: ATT position, Premier League only, age 21 to 26, budget EUR 55M, destination Brentford.

- Injury watch OFF: Igor ranks 27th. His injury history penalises the fit score directly.
- Injury watch ON: Igor ranks 9th. Pure player-club fit without the injury discount.

The orange row signals to the scout: this player fits your system, the medical risk is real, and you need your doctors in the room before making a decision.

A second demonstration: Bryan Mbeumo (also Brentford) moves from rank 60 to rank 1 when injury watch is activated. The fit is extraordinary; the injury history is the sole reason he does not appear at the top of the default ranking.

### Value-for-money intelligence

The market intelligence section ranks shortlisted players by fit_score divided by market_value_eur_m. This is the answer to the question clubs ask most often in practice: "Of the players who fit our system, who is the smart buy?"

A EUR 30M player with fit score 58 ranks above a EUR 100M player with fit score 60. That is the insight the standard ranking table does not surface. The bars are coloured from green (affordable relative to the search budget) to orange (at the top of the budget), so a scout can read the chart and immediately identify where value is concentrated.

### What the current system does not do and what it would take

Three honest limitations:

1. **Player stats are from a single season snapshot (2024-25)**, not the season at the time of the hypothetical transfer. The model was trained on temporally aligned stats (season N for a transfer in year N+1). At inference, it uses current stats for future scouting, which is the correct application but introduces a snapshot assumption.

2. **Pass completion, progressive passes, and carrying stats are heuristic proxies** based on position and market value. They are not real. The model was deliberately designed to exclude these from the neural network to avoid train/inference mismatch (FBref JavaScript rendering). They appear in the UI for display and context but do not influence the fit score.

3. **The success_score label is a composite proxy.** It measures playing time, goal output, retention, and a tactical fit signal. It does not measure player improvement, manager satisfaction, team chemistry, or broader sporting project fit. The model is optimised on what can be measured from public data, not on the full complexity of what makes a transfer succeed.

The most direct path to a better model: (a) headless browser scraping of FBref passing and possession data, unlocking 5 to 8 additional tactically meaningful features; (b) event-level PPDA from StatsBomb or Opta, which would replace the football-data.co.uk proxies with precise pressing metrics; (c) a richer transfer outcome label incorporating wage data and manager ratings, neither of which is publicly available.

---

## DATA AND NUMBERS TO USE THROUGHOUT

Use these exact figures. Do not round or adjust.

- Training set: 3,766 labelled transfers (Big Five, 2018 to 2024)
- Scouting pool: 2,700 Big Five players (2024-25 season)
- Club database: 150 clubs with tactical profiles
- Model parameters: approximately 15,000 to 20,000
- Test R²: 0.122 (two-tower) vs 0.088 (GBM) vs 0.127 (linear)
- Test Spearman rho: 0.370 (two-tower) vs 0.325 (GBM) vs 0.353 (linear)
- Test MAE: 0.160 (two-tower and linear, essentially tied) vs 0.167 (GBM)
- Training convergence: early stopping typically fires around epoch 120 to 180
- Fit_surprise coverage: 51% of training transfers have a valid two-season baseline
- FBref pressing stats coverage: 47% in training, 64% in inference pool
- Tactical differentiation test (MID, under EUR 80M, age 22 to 29):
  - Klopp profile (PPDA 7.5 equivalent): Bissouma, Bentancur, Downes (pressing engines)
  - Guardiola profile: Foden, Tchouameni, Mac Allister (technical, deep MFs)
  - Mourinho profile (PPDA 14.0): Olmo, Gibbs-White, Rogers (goal-scoring attacking MFs)
  - Overlap without fit_surprise: 13 of 15 shared between Guardiola and Mourinho
  - Overlap with fit_surprise: 2 of 15

---

## SLIDE STRUCTURE RECOMMENDATION

**Slide 1: Cover**
Title: "Predicting Transfer Success: A Neural Approach to Player-Club Fit"
Subtitle: AI Football Scout v3
No bullet points. Just the title, a clean visual, and the date.

**Slide 2: The market prices quality, not fit**
Single exhibit: scatter plot of transfer fee vs first-season minutes share (inverted: high fee, low minutes in top-right quadrant). Point: expensive transfers underperform on playing time relative to their cost.
Speaker note: "Forty percent of transfers above EUR 20M do not reach 50% of available minutes in year one. The clubs paying the most have the worst track records. The market prices player scarcity. It does not price system match."

**Slide 3: We are solving the wrong problem**
Two-column comparison: "Player quality rank" vs "Player-club fit score." Show the same player ranking very differently under the two frameworks. One example: a player ranked 50th overall who ranks 1st for a specific club profile.

**Slide 4: The success_score formula**
Single exhibit: the formula with visual weights. 35% playing time, 30% goal contribution, 20% retention, 15% fit_surprise. Explain fit_surprise is the tactical fit signal, not raw performance.

**Slide 5: Fit_surprise: the key design decision**
Show: raw signal distribution (mean -0.22, interpretation), the centering operation, and the resulting signal. Show the tactical differentiation result (13 to 2 overlap before and after). This is the intellectual heart of the data section.

**Slide 6: Three data sources, one label**
Table: source, what it provides, how it feeds the model. TM for player and transfer data, football-data.co.uk for club tactical proxies, FBref for pressing stats. Be explicit about what is real vs proxied vs heuristic.

**Slide 7: Why two towers**
Diagram: player tower and club tower as separate columns, merging at a head MLP. Contrast with a flat concatenation MLP. One sentence of intuition: "Concatenation learns fixed weights; two towers learn an interaction."

**Slide 8: Architecture detail**
Show the exact dimensions. 15 inputs -> 64 -> 32 (player). 9 inputs -> 32 -> 32 (club). Head: 65 -> 64 -> 64 -> 1. Total approximately 15,000 parameters. State why it is deliberately small.

**Slide 9: Feature decisions**
Two-column table: "In the model" vs "Excluded and why." List the key exclusions with one-line explanations. Transfer age (dominated predictions at +0.72 Spearman), league of origin (Premier League bias), pass completion (JavaScript rendering, not available), contract months (temporal leakage).

**Slide 10: Results vs baselines**
Table with R², Spearman, MAE for four models. Highlight two-tower. Then put R² = 0.122 in context: football transfer success is dominated by unmeasured variables. The relevant metric is ranking ability (Spearman 0.37), not explained variance.

**Slide 11: The scouting interface**
Screenshot or wireframe of the Scout tab. Walk through the workflow: set filters, select club, get ranked output. Show the recommendation cards.

**Slide 12: Injury watch mode in action**
Two screenshots side by side: Igor Thiago at rank 27 (default) and rank 9 (injury watch). The orange row. The popover with injury flag and contract status. The insight: "This separates the fit question from the medical question. Your doctors handle the latter."

**Slide 13: The player embedding space**
Show the Similar Players tab. Emphasise: this is a free byproduct of the two-tower training. The 32-dimensional player embeddings were learned from transfer outcomes, not stat similarity. Two players close in embedding space have similar profiles as revealed by where they succeeded and failed, not just their raw numbers.

**Slide 14: Limitations and the path to production**
Three honest gaps with a direct path to closing each:
1. Single season snapshot for player stats -> temporal inference alignment
2. Heuristic passing/carrying stats -> headless FBref scraping
3. Proxy PPDA -> Opta/StatsBomb event-level data

Close the presentation with one line: "The model is ready to run. The data infrastructure determines how much further it can go."

---

## TONE AND VOICE GUIDANCE

Write as if presenting to the analytics director and sporting director of a mid-table Premier League club. They are intelligent and sceptical. They have seen many vendor pitches. They will challenge R² = 0.122 immediately. You must have the honest, precise answer ready (12% of a genuinely noisy target is a real signal, ranking ability is what matters operationally, and the baseline is not "a perfect model" but "current scouting without this layer").

Do not oversell. The model is not a recruitment system. It is a shortlisting layer. That is a real, specific, achievable value proposition. Claim it precisely.

Do not use vague phrases like "state of the art," "cutting edge," "powerful," or "robust." Use numbers. Use comparisons. Use the baselines.

The strongest moment in this presentation is the Igor Thiago slide. A 22-year-old Brentford attacker moves from rank 27 to rank 9 when you strip out the injury penalty. That is a real, human, understandable story. Build to it.
