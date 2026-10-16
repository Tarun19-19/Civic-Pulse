# TECH STACK

| Layer | Choice | Why | Cost |
|---|---|---|---|
| Base app | **God's Eye View** (Vite 6, CesiumJS ≥1.124, vanilla JS modules) | Required base; 3D globe cockpit | Free (see LICENSE) |
| Smart City UI | Vanilla HTML/CSS/JS, **CesiumJS** via jsDelivr CDN | No build step, matches base stack, themeable (light/dark) | Free |
| Basemap | OpenStreetMap raster tiles | Keyless | Free |
| Charts | Inline SVG (hand-rolled) | Zero dependencies | Free |
| Data pipeline | **Python 3.12**, pandas, numpy, scipy | Reproducible CSV→JSON | Free |
| Models | **scikit-learn**: RidgeCV, ElasticNetCV, HistGradientBoosting (L2) | Regularised, trained from scratch, no pretrained weights | Free |
| Notebooks | nbformat + nbconvert + ipykernel, matplotlib | Generated & executed by `make_notebooks.py` | Free |
| Hosting | **Vercel** Hobby (static `dist`), or Netlify / Cloudflare Pages | No server, no env vars | Free |
| CI | Existing `.github/workflows/ci.yml` from base | — | Free |

**APIs / keys / pretrained models used: none.** (Rule 11 disclosure.)
**Borrowed code:** God's Eye View (upstream license applies). Everything under `smartcity/` and `public/smartcity/` is new.

## Files added to the base repo
```
PRD.md  CONSTRAINTS.md  TECH_STACK.md  ARCHITECTURE.md  rules.md
setup.ps1  setup.sh  vercel.json
index.html                      (one link added: SMART CITY PRIORITIES)
smartcity/
  requirements.txt
  data/raw/{locations,incidents,public_complaints}.csv  README.txt
  pipeline/smartcity_pipeline.py
  notebooks/{01_preprocessing_eda,02_modeling_regularization}.ipynb  make_notebooks.py
public/smartcity/
  index.html  architecture.html
  data/{priorities,trends,validation}.json
```

## Commands
| Task | Command |
|---|---|
| Rebuild JSON | `python smartcity/pipeline/smartcity_pipeline.py` |
| Rebuild notebooks | `python smartcity/notebooks/make_notebooks.py` |
| Dev | `npm run dev` → `/smartcity/index.html` |
| Production build | `npx vite build` (output in `dist/`, includes `dist/smartcity/`) |
