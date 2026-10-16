"""Generates + executes the two notebooks. Run: python smartcity/notebooks/make_notebooks.py"""
import nbformat as nbf
from pathlib import Path
from nbconvert.preprocessors import ExecutePreprocessor
HERE = Path(__file__).parent
md = nbf.v4.new_markdown_cell; code = nbf.v4.new_code_cell
SETUP = '''import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path.cwd().resolve().parents[1] if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "smartcity" / "pipeline"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import smartcity_pipeline as sp
pd.set_option("display.width", 160)'''

nb1 = nbf.v4.new_notebook(); nb1.cells = [
 md("# 01 - Data audit, cleaning & regularised feature pipeline\n**Synthetic educational data** - nothing here proves real-world performance (rule 09)."),
 code(SETUP),
 md("## 1. Load + validate\nChecks: duplicate IDs, orphan `location_id`s, missing values, censored durations."),
 code("loc, inc, com = sp.load_raw()\nloc, inc, com, q = sp.clean(loc, inc, com)\nq"),
 md("**Cleaning decisions**\n- `resolved == 0` rows still carry `resolution_hours` -> *time open so far* (right-censored). They are excluded from fix-time medians and counted as backlog.\n- Durations are winsorised at the 1st/99th percentile so outliers cannot dominate rankings.\n- Categorical severity/priority mapped to ordinal weights (Low 1, Medium 2, High 4, Critical 8)."),
 code("print(inc.groupby('severity').resolution_hours.describe().round(1))\nfig, ax = plt.subplots(1, 2, figsize=(11, 3))\ninc.resolution_hours.hist(bins=50, ax=ax[0]); ax[0].set_title('incident hours (raw)')\ninc.resolution_hours_w.hist(bins=50, ax=ax[1]); ax[1].set_title('winsorised'); plt.show()"),
 md("## 2. Is there signal? (honest EDA)\nIf incident counts per location look like pure Poisson noise, rankings are mostly noise and the product must show that uncertainty."),
 code("n = inc.groupby('location_id').size().reindex(loc.location_id, fill_value=0)\nprint('mean %.2f  std %.2f  | Poisson std would be %.2f' % (n.mean(), n.std(), n.mean()**.5))\nfrom scipy.stats import kruskal\nl2 = loc.assign(n=n.values)\nfor c in ['road_condition','road_type','pedestrian_activity','zone','near_school','near_hospital']:\n    print(f'{c:22s} Kruskal p = {kruskal(*[g.n.values for _, g in l2.groupby(c)]).pvalue:.3f}')"),
 md("Counts are ~Poisson(10) and unrelated to context attributes. **Consequence:** we (a) shrink sparse locations toward their zone mean, (b) bootstrap rank intervals, (c) label confidence, and (d) never claim forecasting skill."),
 md("## 3. Regularisation set-up\n1. **Empirical-Bayes shrinkage**: `load = (n*raw + k*zone_mean)/(n+k)`, k=3.\n2. **Recency decay** half-life 120 days.\n3. **Percentile normalisation** of each component (robust to scale/outliers).\n4. **Poisson bootstrap** (200x) for rank stability.\n5. L2 / L1+L2 regularisation in the models of notebook 02."),
 code("asof = pd.Timestamp('2026-01-01')\nf = sp.location_features(loc, inc, com, asof)\ns = sp.score(f)\nd = f.merge(s, on='location_id')\nprint('raw vs shrunk load spread:', f.sev_load_raw.std().round(2), '->', f.sev_load.std().round(2))\nd.sort_values('score', ascending=False)[['location_id','zone','n_inc','n_com','score']].head(10)"),
 code("plt.figure(figsize=(5,4)); plt.scatter(f.sev_load_raw, f.sev_load, s=6, alpha=.5); plt.xlabel('raw decayed load'); plt.ylabel('shrunk load'); plt.title('Shrinkage pulls sparse locations to zone mean'); plt.show()"),
 md("## 4. Leakage guard\n`location_features(..., asof)` only uses events strictly before `asof`; targets in notebook 02 come strictly after the cutoff.")]
nb2 = nbf.v4.new_notebook(); nb2.cells = [
 md("# 02 - Time-split validation of regularised models vs baselines\nTarget: severity-weighted incidents in the **next 92 days**. Train on earlier cutoffs only; test on a later cutoff (rule 07)."),
 code(SETUP),
 code("loc, inc, com, q = sp.clean(*sp.load_raw())\nval = sp.temporal_validation(loc, inc, com)\nfor k, v in val.items():\n    print('\\n', k, '(train rows:', v['_train_rows'], ')')\n    print(pd.DataFrame({m: x for m, x in v.items() if not m.startswith('_')}).T)"),
 md("### Reading the result\n- Every method sits near the **Random** baseline (Spearman ~ 0, precision@100 ~ 0.10 expected by chance).\n- Regularisation (Ridge/ElasticNet/GBM-L2) correctly refuses to invent signal: ElasticNet shrinks to a constant.\n- **We therefore do not claim predictive accuracy.** The priority score is a transparent *triage of current evidence*, not a forecast. On real data the same harness would quantify any genuine lift."),
 code("m = pd.DataFrame({k: {n: x['precision_at_100'] for n, x in v.items() if not n.startswith('_')} for k, v in val.items()})\nm.plot.barh(figsize=(8,4)); plt.axvline(.10, c='k', ls='--', label='chance'); plt.legend(); plt.title('precision@100 on held-out quarter'); plt.show()")]
for name, nb in [("01_preprocessing_eda.ipynb", nb1), ("02_modeling_regularization.ipynb", nb2)]:
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
    nbf.write(nb, HERE / name); print("wrote", name)
