# ARCHITECTURE & EXPLANATION

Visual version: `/smartcity/architecture.html` (animated diagram).

## 1. Flow
```
locations.csv ┐
incidents.csv ├─► smartcity_pipeline.py ─► priorities.json ┐
complaints.csv┘   clean → features(as-of) →  trends.json    ├─► Smart City Command Center (CesiumJS)
                  shrink → score → bootstrap validation.json┘     queue · map · trends · plan · evidence
                  → reasons/confidence/missing
```
Build-time Python, run-time static JS. No server: the browser fetches three JSON files.

## 2. Components
- **`clean()`** validates IDs, flags orphans/duplicates, marks unresolved incidents as censored, winsorises durations, adds ordinal weights.
- **`location_features(asof)`** one row per location from events before `asof` (leakage guard). Accepts bootstrap weights.
- **`score()`** six percentile-scaled components → weighted 0–100 score.
- **`bootstrap_rank()`** 200 Poisson-reweighted recomputations → 10th/90th percentile rank, P(top 50).
- **`build()`** adds corroboration (top incident type vs top complaint type via category map), sensor share, confidence.
- **`reasons()` / `missing_info()`** turn numbers into the sentences officials read.
- **`temporal_validation()`** expanding-window time splits; Ridge, ElasticNet, regularised GBM vs last-90d, lifetime, our score, random.
- **Front end** reads JSON; renders entities (points for all, columns for top 100), tabs, planner math in-browser.

## 3. Why a location ranks high (example #1 = L0642)
Score 97.5, interval #1–#107, 84% chance of staying top-50. Reasons: severity load (98th pct), backlog (98th), exposure (98th). Missing info: complaints say *Pothole*, incidents say *Streetlight outage* → "verify on site". This is the intended output: a rank plus the doubt attached to it.

## 4. Score
`score = 0.30·severity + 0.20·backlog + 0.15·complaints + 0.15·context + 0.10·injuries + 0.10·trend` (each a 0–100 percentile).
Shrinkage `(n·raw + 3·zone_mean)/(n+3)` stops a site with 2 lucky/unlucky events from dominating.

## 5. Regularisation story (for judges)
Shrinkage (data level) + percentile scaling (feature level) + L2/L1+L2/GBM-L2 (model level) + bootstrap (rank level). On this synthetic data they all agree there is no forecastable signal; ElasticNet collapses to a constant. We report this rather than tuning until a metric looks good.

## 6. Planner math
`load covered = Σ(selected sites' weighted historical load) / total`; `assumed cut = effectiveness × load covered`. Baseline: random K sites cover K/1000. Strategies close to that baseline = data cannot discriminate.

## 7. Integration with God's Eye View
Static module in `public/` → served by Vite dev and copied to `dist/` on build. Same CesiumJS engine, same cockpit palette, cross-links both ways. Upstream code untouched except one link in `index.html`.

## 8. Failure modes & mitigations
| Risk | Mitigation |
|---|---|
| Noise mistaken for signal | Intervals, confidence, Evidence tab |
| CDN/tiles offline | Message in console; data still loads; list works without map tiles |
| Misread as real places | Banner + README notice |
| Weights contested | Visible; editable in `COMPONENTS` |
