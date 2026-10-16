# CONSTRAINTS

## Competition (from rules.md)
- One track: **03 Where should we fix first?** Time-boxed: 9:00–18:00, 16 Oct 2026. Solo/duo.
- Provided data is foundational; extra data must be disclosed; no replacing core data with an API.
- Must be a **product**, not only a notebook/model. Model claims need evaluation method + metrics. No leakage. Explain every result. Acknowledge synthetic limits.
- Disclose every API key/pretrained model/external service. From-scratch models earn more than API solutions.
- Plagiarism / fabricated evaluation → disqualification. Base repo must be credited.

## Data
- Synthetic: counts per location ≈ Poisson(10); no real structure vs attributes. **Do not claim predictive accuracy or real-world validity.**
- Coordinates simulated — never describe them as real places.
- Unresolved incidents carry elapsed hours (censored); handled explicitly.
- Only 2025 (12 months): no multi-year seasonality; forecasting horizon limited to one 92-day window.

## Technical
- **No paid services, no API keys, no backend.** Everything static; free-tier hostable.
- CesiumJS loaded from jsDelivr CDN; OSM tiles used only as light basemap (respect OSM tile usage policy: low traffic demo, attribution shown by Cesium).
- Needs internet at view time for CDN + tiles. Offline fallback: `npm run dev` uses the repo's bundled Cesium for the main app only.
- Windows 11 dev; Node ≥ 20 (tested 22), Python ≥ 3.10 (tested 3.12).
- Base repo is large (≈117 MB with history excluded); `npm ci` needs ~1–2 GB disk; use `PUPPETEER_SKIP_DOWNLOAD=1`.
- Dev-server quirk: open `/smartcity/index.html` (not `/smartcity/`).
- JSON payload ≈ 0.9 MB (1,000 locations with explanations) — fine for static hosting.

## Product / ethics
- Output is decision **support** for review, never automatic dispatch. Not for actual public-safety decisions.
- Effectiveness figures in the planner are user assumptions, labelled as such.
- No personal data. No live deployment claims.

## Not done / known gaps
- No automated browser UI test was run in the build sandbox (build, route smoke test and JS syntax check passed). Open the page once and check the console.
- GitHub Pages not configured (root app uses absolute paths; needs a `base` change).
- Weights are expert-set, not learned (learning them on noise would be overfitting).
