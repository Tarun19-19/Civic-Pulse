# PRD — Smart City Priority Engine ("Where should we fix first?")

**Event:** DATATHON 2026 · Fri 16 Oct 2026 · CB G20, VIT-AP · Track 03 · **Team:** Cyborg (solo/duo)
**Project root:** `C:\SDE Projects\SmartCity` · **Base project:** [God's Eye View](https://github.com/bilawalsidhu/gods-eye-view) (cloned unchanged into the root)

---
## 1. Problem & user question
Cities get reports on potholes, signals, streetlights, drains, flooding and collisions. Crews are limited. Officials need a transparent way to decide **which locations deserve attention first, on what evidence, and what is still missing.**

The product answers exactly the published "useful product question": *"Which locations should be reviewed first, based on what evidence, and what information is still missing?"*

**Users:** municipal operations teams · road/traffic departments · civic-service coordinators.

## 2. Product concept (Problem Statement × God's Eye View)
God's Eye View is a CesiumJS 3D globe "cockpit" for exploring public signals. We keep its stack and visual language and add a **Smart City Command Center** (`/smartcity/index.html`): the 1,000 simulated locations become a ranked 3D map, with every rank explained.

| Problem-statement deliverable | Where it lives |
|---|---|
| 1. Map + ranked list of locations needing attention | Globe markers + 3D columns (top 100) + **QUEUE** tab |
| 2. Complaint/incident trend monitor | **TRENDS** tab (monthly lines by type/category) |
| 3. Severity-and-frequency priority score | `smartcity_pipeline.py` → 6-component score |
| 4. Resolution-time dashboard | **TRENDS** tab (median/p90 days, open share) |
| 5. Recommendation queue with reasons + confidence | Right panel: why, rank interval, confidence, missing info, suggested crew |
| Compare possible interventions | **PLAN** tab: budget slider × 4 allocation strategies |
| Limits & evidence | **EVIDENCE** tab + `architecture.html` |

## 3. Functional requirements
- **FR1** Load 3 CSVs, validate (duplicates, orphans, missing), clean (censored durations, winsorising).
- **FR2** Compute per-location score from six weighted components; weights visible in UI.
- **FR3** Every ranked location shows: score, rank, top-3 reasons with points, full breakdown, bootstrap rank interval (10–90%), P(top 50), confidence label, missing-info flags, dominant problem type → suggested intervention.
- **FR4** Filters: zone, confidence, problem type, ID search.
- **FR5** Trend monitor + resolution-time dashboard.
- **FR6** Intervention planner comparing: Highest score · Score×confidence · Backlog first · Zone-balanced. Shows historical load covered, zones touched, non-low-confidence sites. Effectiveness is a user-editable **assumption**.
- **FR7** Evidence tab: time-split validation vs baselines, data-quality report, weights, extra-data disclosure.
- **FR8** Architecture sub-page (`architecture.html`) with animated diagram.
- **FR9** Synthetic-data banner always visible.

## 4. Data
`smartcity/data/raw/`: `locations.csv` (1,000) · `incidents.csv` (10,000) · `public_complaints.csv` (8,000) · `README.txt`. Join key `location_id`. All 2025, synthetic, Guntur–Amaravati-area simulated coordinates.
**Extra data used: none.** OpenStreetMap raster tiles = basemap only.

## 5. Method
1. **Clean** – `resolved=0` rows carry `resolution_hours` = time open so far (right-censored): excluded from fix-time stats, counted as backlog. Winsorise durations at 1/99 pct. Ordinal weights: severity L1/M2/H4/C8, complaint priority L1/N2/H4.
2. **Features at an as-of date** (events strictly before it): decayed severity load (half-life 120 d), injuries + 2×critical, unresolved backlog, slow-fix excess vs type median, priority-weighted complaints (open & decayed), recent-90-day vs earlier trend, context exposure.
3. **Regularisation:** empirical-Bayes shrinkage `(n·raw + 3·zone_mean)/(n+3)`; percentile normalisation; L2 / L1+L2 / depth-3 GBM with L2 in the models.
4. **Score** = 0.30 severity + 0.20 backlog + 0.15 complaints + 0.15 context + 0.10 injuries + 0.10 trend (percentile-scaled 0–100).
5. **Confidence** = 50% evidence volume + 40% rank stability (200× Poisson bootstrap) + 10% incident↔complaint agreement → Low (<0.55) / Medium / High (>0.75).
6. **Validation:** features at cutoff → target = severity load in next 92 days. Train on earlier cutoffs, test on later (Jul-1 and Oct-1). Metrics: Spearman, precision@100 (chance = 0.10), MAE.

## 6. Honest findings (must appear in the pitch)
- Incidents per location: mean 10.0, sd 3.25 vs Poisson sd 3.16. No relationship with road condition, type, zone, density, school/hospital (all Kruskal p > 0.5).
- On held-out quarters **every method, including ours, scores about chance** (precision@100 0.05–0.17 vs 0.10). Regularised models correctly refuse to invent signal.
- Therefore: the score is **evidence triage**, not a forecast; we claim no accuracy. Result distribution: 12 High / 136 Medium / 852 Low confidence — the low share is the correct reading of ~10 incidents + ~8 complaints per site.
- The value of the product is *transparency*: reasons, intervals, missing-info, and the planner baseline ("random pick covers K/1000").

## 7. Setup (Windows)
```powershell
cd "C:\SDE Projects\SmartCity"
# Option A — extract the delivered zip here, then:
powershell -ExecutionPolicy Bypass -File setup.ps1
# Option B — manual
git clone --depth 1 https://github.com/bilawalsidhu/gods-eye-view.git .   # base project at the root
python -m pip install -r smartcity/requirements.txt
python smartcity/pipeline/smartcity_pipeline.py      # CSV -> public/smartcity/data/*.json
python smartcity/notebooks/make_notebooks.py         # builds + executes both notebooks
npm ci                                               # PUPPETEER_SKIP_DOWNLOAD=1 on slow links
npm run dev
```
Open **http://localhost:5173/smartcity/index.html** (use the full `index.html`; dev server returns the main globe for `/smartcity/`).
Notebooks: `jupyter lab smartcity/notebooks` → `01_preprocessing_eda.ipynb`, `02_modeling_regularization.ipynb`.
Rules copy: `rules.md`.

## 8. Deploy free
**Vercel (recommended):** import repo → uses `vercel.json` (`npx vite build` → `dist`). No env vars, no keys. Page URL: `https://<app>.vercel.app/smartcity/index.html`.
Netlify/Cloudflare Pages: build `npx vite build`, publish `dist`. The Smart City pages are static, so `public/smartcity/` alone can also be hosted anywhere.

## 9. Demo script (rule 12: real input → useful output)
1. Open QUEUE → click #1 → read reasons, interval, missing info.
2. Filter *High confidence* → show how few sites are trustworthy.
3. PLAN → set 50 sites → compare strategies against the random baseline.
4. TRENDS → slowest-resolving categories.
5. EVIDENCE → show the chance-level test and say why we don't claim accuracy.
6. Architecture page → data → pipeline → JSON → globe.

## 10. Rules compliance
| Rule | How |
|---|---|
| 02/04 core data used | Entire product is built from the 3 CSVs |
| 03 extra data | None; OSM tiles basemap disclosed |
| 05/12 product, workflow | Interactive app + pipeline + notebooks |
| 06 claims supported | Method, metrics, baselines shown; no accuracy claim |
| 07 leakage | As-of features, future targets, time splits |
| 08 explain | Reasons/breakdown/intervals/missing-info |
| 09 synthetic limits | Banner + Evidence tab |
| 10 responsible | No real people, no live deployment claims |
| 11 tools | No API keys. Disclosed: CesiumJS, OSM tiles, scikit-learn (trained from scratch, no pretrained models) |
| 15 integrity | Borrowed: God's Eye View (see its LICENSE / THIRD_PARTY_NOTICES) |

## 11. Out of scope / next steps
Real data with ground-truth outcomes; calibrated forecasting; crew routing; auth; live feeds.
