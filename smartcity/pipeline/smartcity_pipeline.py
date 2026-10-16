"""Smart City - "Where should we fix first?"  Reproducible pipeline.

raw CSVs -> clean/validate -> features -> shrinkage-regularised priority score
-> bootstrap confidence -> reasons / missing-info flags -> JSON for the web app.
Run:  python smartcity/pipeline/smartcity_pipeline.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "smartcity" / "data" / "raw"
OUT = ROOT / "public" / "smartcity" / "data"
SEV_W = {"Low": 1, "Medium": 2, "High": 4, "Critical": 8}
PRI_W = {"Low": 1, "Normal": 2, "High": 4}
COND_W = {"Good": 0, "Fair": 1, "Poor": 2, "Critical": 3}
ROAD_W = {"Local": 0.2, "Residential": 0.4, "Collector": 0.6, "Arterial": 0.85, "Highway": 1.0}
PED_W = {"Low": 0, "Medium": 0.5, "High": 1}
CAT2TYPE = {"Road damage": "Pothole", "Lighting": "Streetlight outage", "Traffic": "Signal failure",
            "Drainage": "Blocked drain", "Waste": "Blocked drain", "Pedestrian safety": "Collision"}
COMPONENTS = {  # name: (weight, human label)
    "severity": (0.30, "Severity-weighted incident load"),
    "injuries": (0.10, "Injuries & critical events"),
    "backlog": (0.20, "Unresolved backlog / slow fixes"),
    "complaints": (0.15, "Public complaint pressure"),
    "context": (0.15, "Exposure (road, density, school, hospital, pedestrians)"),
    "trend": (0.10, "Worsening trend (last 90 days)"),
}
SHRINK_K = 3.0      # empirical-Bayes pseudo-events pulling a location toward its zone mean
HALF_LIFE_D = 120   # recency half-life for event weighting
N_BOOT = 200


def load_raw():
    loc = pd.read_csv(RAW / "locations.csv")
    inc = pd.read_csv(RAW / "incidents.csv", parse_dates=["timestamp"])
    com = pd.read_csv(RAW / "public_complaints.csv", parse_dates=["created_at"])
    return loc, inc, com


def clean(loc, inc, com):
    """Validation + cleaning. Returns cleaned frames and a data-quality report."""
    q = {"rows": {"locations": len(loc), "incidents": len(inc), "complaints": len(com)}}
    q["duplicate_ids"] = {"incidents": int(inc.incident_id.duplicated().sum()),
                          "complaints": int(com.complaint_id.duplicated().sum())}
    inc = inc.drop_duplicates("incident_id"); com = com.drop_duplicates("complaint_id")
    q["orphan_location_ids"] = int((~inc.location_id.isin(loc.location_id)).sum() + (~com.location_id.isin(loc.location_id)).sum())
    inc = inc[inc.location_id.isin(loc.location_id)].copy(); com = com[com.location_id.isin(loc.location_id)].copy()
    q["missing_values"] = int(loc.isna().sum().sum() + inc.isna().sum().sum() + com.isna().sum().sum())
    # unresolved incidents still carry resolution_hours: time open so far (right-censored), not a fix time
    inc["censored"] = (inc.resolved == 0)
    q["unresolved_with_hours_censored"] = int(inc.censored.sum())
    com["is_open"] = ~com.status.isin({"Resolved", "Closed"})
    q["complaints_open"] = int(com.is_open.sum())
    # winsorise heavy-tailed durations (1/99 pct) so outliers cannot drive rankings
    for df, col in [(inc, "resolution_hours"), (com, "days_to_resolution")]:
        lo, hi = df[col].quantile([.01, .99]); df[col + "_w"] = df[col].clip(lo, hi)
    inc["sev_w"] = inc.severity.map(SEV_W); com["pri_w"] = com.priority_reported.map(PRI_W)
    inc["month"] = inc.timestamp.dt.month; com["month"] = com.created_at.dt.month
    q["synthetic"] = True
    return loc, inc.reset_index(drop=True), com.reset_index(drop=True), q


def pct(s):  # percentile rank 0..100
    return s.rank(pct=True) * 100


def location_features(loc, inc, com, asof, w_boot=None):
    """Aggregate to one row per location using events strictly before `asof` (no leakage)."""
    mi = (inc.timestamp < asof).values; mc = (com.created_at < asof).values
    I = inc[mi]; C = com[mc]
    wi = np.ones(len(I)) if w_boot is None else w_boot[0][mi]
    wc = np.ones(len(C)) if w_boot is None else w_boot[1][mc]
    age_i = (asof - I.timestamp).dt.days.values; age_c = (asof - C.created_at).dt.days.values
    decay_i = 0.5 ** (age_i / HALF_LIFE_D); decay_c = 0.5 ** (age_c / HALF_LIFE_D)
    recent_i = age_i <= 90
    ids = loc.location_id; idx = {k: i for i, k in enumerate(ids)}; n = len(ids)
    ii = I.location_id.map(idx).values; ci = C.location_id.map(idx).values
    bc = lambda ix, w: np.bincount(ix, weights=w, minlength=n)
    f = pd.DataFrame({"location_id": ids.values})
    f["n_inc"] = bc(ii, wi)
    f["sev_load_raw"] = bc(ii, wi * I.sev_w.values * decay_i)
    f["inj_raw"] = bc(ii, wi * (I.injuries.values + 2 * (I.severity == "Critical").values))
    f["backlog_inc"] = bc(ii, wi * I.censored.values * I.sev_w.values)
    med = I.groupby("incident_type").resolution_hours_w.transform("median").values
    f["slow_excess"] = bc(ii, wi * np.clip(I.resolution_hours_w.values / med - 1, 0, 3))
    f["n_com"] = bc(ci, wc)
    f["com_open"] = bc(ci, wc * C.is_open.values * C.pri_w.values)
    f["com_load"] = bc(ci, wc * C.pri_w.values * decay_c)
    f["recent_load"] = bc(ii, wi * recent_i * I.sev_w.values)
    f["older_load"] = bc(ii, wi * (~recent_i) * I.sev_w.values)
    for t in sorted(inc.incident_type.unique()):
        f["type_" + t] = bc(ii, wi * (I.incident_type == t).values * I.sev_w.values)
    f = f.merge(loc, on="location_id")
    f["zone_mean"] = f.groupby("zone").sev_load_raw.transform("mean")
    f["sev_load"] = (f.n_inc * f.sev_load_raw + SHRINK_K * f.zone_mean) / (f.n_inc + SHRINK_K)  # shrinkage
    f["ctx_raw"] = (0.30 * f.road_condition.map(COND_W) / 3 + 0.20 * pct(np.log(f.population_density_per_km2)) / 100
                    + 0.15 * f.near_school + 0.10 * f.near_hospital + 0.15 * f.pedestrian_activity.map(PED_W)
                    + 0.10 * f.road_type.map(ROAD_W))
    span = max(1, (asof - inc.timestamp.min()).days - 90)
    f["trend_raw"] = (f.recent_load / 90.0 - f.older_load / span) * 90   # recent daily rate vs earlier daily rate
    return f


def score(f):
    s = pd.DataFrame({"location_id": f.location_id})
    s["severity"] = pct(f.sev_load); s["injuries"] = pct(f.inj_raw)
    s["backlog"] = pct(f.backlog_inc + f.slow_excess + 0.25 * f.com_open)
    s["complaints"] = pct(f.com_load); s["context"] = pct(f.ctx_raw); s["trend"] = pct(f.trend_raw)
    s["score"] = sum(w * s[k] for k, (w, _) in COMPONENTS.items())
    return s


def bootstrap_rank(loc, inc, com, asof, seed=7):
    rng = np.random.default_rng(seed); ranks = []
    for _ in range(N_BOOT):
        wb = (rng.poisson(1, len(inc)).astype(float), rng.poisson(1, len(com)).astype(float))
        ranks.append(score(location_features(loc, inc, com, asof, wb)).score.rank(ascending=False).values)
    R = np.array(ranks)
    return np.percentile(R, 10, axis=0), np.percentile(R, 90, axis=0), (R <= 50).mean(axis=0)


def build(asof=None):
    loc, inc, com, q = clean(*load_raw())
    asof = asof or (max(inc.timestamp.max(), com.created_at.max()).normalize() + pd.Timedelta(days=1))
    f = location_features(loc, inc, com, asof); s = score(f)
    lo, hi, p50 = bootstrap_rank(loc, inc, com, asof)
    d = f.merge(s, on="location_id"); d["rank"] = d.score.rank(ascending=False, method="first").astype(int)
    d["rank_lo"], d["rank_hi"], d["p_top50"] = lo, hi, p50
    top_mode = lambda x: x.value_counts().idxmax()
    d["top_type"] = d.location_id.map(inc.groupby("location_id").incident_type.agg(top_mode))
    d["com_type"] = d.location_id.map(com.assign(t=com.category.map(CAT2TYPE)).groupby("location_id").t.agg(top_mode))
    d["corroborated"] = d.top_type == d.com_type
    d["sensor_share"] = d.location_id.map(inc.assign(s=inc.reported_by == "Sensor").groupby("location_id").s.mean()).fillna(0)
    # confidence = evidence volume (50%) + rank stability under resampling (40%) + cross-source agreement (10%)
    vol = np.clip((d.n_inc + d.n_com) / 20, 0, 1); stab = 1 - np.clip((d.rank_hi - d.rank_lo) / 400, 0, 1)
    d["confidence"] = (0.5 * vol + 0.4 * stab + 0.1 * d.corroborated).round(3)
    d["conf_label"] = pd.cut(d.confidence, [-1, .55, .75, 2], labels=["Low", "Medium", "High"]).astype(str)
    return loc, inc, com, q, d, asof


def reasons(r):
    parts = sorted(((k, w * r[k]) for k, (w, _) in COMPONENTS.items()), key=lambda x: -x[1])[:3]
    return [{"key": k, "label": COMPONENTS[k][1], "points": round(float(v), 1), "pctile": round(float(r[k]))} for k, v in parts]


def missing_info(r):
    m = []
    if r.n_inc + r.n_com < 12: m.append("Few reports - ranking may be noise; schedule an inspection")
    if r.sensor_share == 0: m.append("No sensor-reported incidents - severity is human-reported only")
    if not r.corroborated: m.append("Complaints and incidents disagree on problem type - verify on site")
    if r.p_top50 < 0.5 and r["rank"] <= 50: m.append("Rank unstable under resampling - borderline for the top 50")
    if r.backlog_inc > 0 and r.com_open == 0: m.append("Open incidents but no open complaints - may be unknown to the public")
    return m


def trends(inc, com):
    t = {"months": [f"2025-{m:02d}" for m in range(1, 13)]}
    t["incidents_by_type"] = {k: inc[inc.incident_type == k].groupby("month").size().reindex(range(1, 13), fill_value=0).tolist() for k in sorted(inc.incident_type.unique())}
    t["complaints_by_cat"] = {k: com[com.category == k].groupby("month").size().reindex(range(1, 13), fill_value=0).tolist() for k in sorted(com.category.unique())}
    t["incident_resolution_hours"] = {k: round(float(v), 1) for k, v in inc[~inc.censored].groupby("incident_type").resolution_hours.median().items()}
    t["complaint_days"] = {k: {"median": round(float(g.days_to_resolution.median()), 1), "p90": round(float(g.days_to_resolution.quantile(.9)), 1),
                              "open_share": round(float(g.is_open.mean()), 3)} for k, g in com.groupby("category")}
    t["severity_mix"] = inc.severity.value_counts().to_dict(); t["status_mix"] = com.status.value_counts().to_dict()
    return t


def temporal_validation(loc, inc, com):
    """Features from events < cutoff; target = severity load in the next 92 days (strictly future).
    Train on earlier cutoffs only, test on a later one. Regularised models vs naive baselines."""
    from sklearn.linear_model import RidgeCV, ElasticNetCV
    from sklearn.ensemble import HistGradientBoostingRegressor
    base = location_features(loc, inc, com, pd.Timestamp("2025-07-01"))
    feats = ["n_inc", "sev_load_raw", "inj_raw", "backlog_inc", "slow_excess", "n_com", "com_open", "com_load", "recent_load", "ctx_raw", "trend_raw"] + \
            [c for c in base.columns if c.startswith("type_")]
    def table(cut):
        f = location_features(loc, inc, com, cut)
        y = inc[(inc.timestamp >= cut) & (inc.timestamp < cut + pd.Timedelta(days=92))].groupby("location_id").sev_w.sum()
        f["y"] = f.location_id.map(y).fillna(0); f["cut"] = cut; return f
    def p_at_k(y, p, k=100): return float(len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k])) / k)
    res = {}
    for te_cut in [pd.Timestamp("2025-07-01"), pd.Timestamp("2025-10-01")]:
        tr_cuts = [c for c in pd.date_range("2025-03-01", te_cut, freq="MS") if c + pd.Timedelta(days=92) <= te_cut]
        tr = pd.concat([table(c) for c in tr_cuts]); te = table(te_cut)
        Xtr, ytr, Xte, yte = tr[feats].values, tr.y.values, te[feats].values, te.y.values
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Z = lambda X: (X - mu) / sd
        out = {"Baseline: last-90d load": te.recent_load.values, "Baseline: lifetime load": te.sev_load_raw.values,
               "Our priority score": score(location_features(loc, inc, com, te_cut)).score.values}
        rng = np.random.default_rng(0); out["Random"] = rng.random(len(te))
        models = {"Ridge (L2)": RidgeCV(alphas=np.logspace(-1, 4, 30)),
                  "ElasticNet (L1+L2)": ElasticNetCV(l1_ratio=[.2, .5, .8], cv=4, alphas=30, max_iter=5000),
                  "HistGB (L2, depth 3)": HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=150, l2_regularization=10, min_samples_leaf=40, random_state=0)}
        for name, m in models.items():
            m.fit(Z(Xtr), ytr); out[name] = m.predict(Z(Xte))
        res[f"test_{te_cut.date()}"] = {k: {"spearman": None if np.ptp(v) == 0 else round(float(spearmanr(v, yte)[0]), 3), "precision_at_100": round(p_at_k(yte, v), 3),
                                          "mae": None if (k.startswith("Baseline") or k in ("Random", "Our priority score")) else round(float(np.mean(np.abs(v - yte))), 3)}
                                      for k, v in out.items()}
        res[f"test_{te_cut.date()}"]["_train_rows"] = int(len(tr))
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loc, inc, com, q, d, asof = build()
    d = d.sort_values("rank"); rows = []
    for _, r in d.iterrows():
        rows.append({"id": r.location_id, "rank": int(r["rank"]), "zone": r.zone, "lat": round(float(r.latitude), 5), "lon": round(float(r.longitude), 5),
                     "score": round(float(r.score), 1), "rank_range": [int(r.rank_lo), int(r.rank_hi)], "p_top50": round(float(r.p_top50), 2),
                     "confidence": float(r.confidence), "conf_label": r.conf_label, "reasons": reasons(r),
                     "components": {k: round(float(r[k]), 1) for k in COMPONENTS}, "missing": missing_info(r),
                     "top_type": r.top_type, "complaint_type": r.com_type, "n_inc": int(r.n_inc), "n_com": int(r.n_com),
                     "road_type": r.road_type, "road_condition": r.road_condition, "density": int(r.population_density_per_km2),
                     "school": int(r.near_school), "hospital": int(r.near_hospital), "ped": r.pedestrian_activity,
                     "by_type": {c[5:]: round(float(r[c]), 1) for c in d.columns if c.startswith("type_")}})
    zone = d.groupby("zone").agg(score=("score", "mean"), n=("location_id", "size"), top100=("rank", lambda x: int((x <= 100).sum()))).round(1).reset_index().sort_values("score", ascending=False)
    meta = {"asof": str(asof.date()), "weights": {k: {"w": w, "label": l} for k, (w, l) in COMPONENTS.items()}, "shrink_k": SHRINK_K,
            "half_life_days": HALF_LIFE_D, "bootstrap": N_BOOT, "quality": q, "zones": zone.to_dict("records"),
            "synthetic_notice": "Synthetic educational data. Coordinates are simulated. Not for real public-safety decisions."}
    (OUT / "priorities.json").write_text(json.dumps({"meta": meta, "locations": rows}, separators=(",", ":")))
    (OUT / "trends.json").write_text(json.dumps(trends(inc, com), separators=(",", ":")))
    val = temporal_validation(loc, inc, com)
    (OUT / "validation.json").write_text(json.dumps(val, indent=1))
    print("top5:", [(x["id"], x["score"], x["conf_label"]) for x in rows[:5]])
    print("conf mix:", pd.Series([x["conf_label"] for x in rows]).value_counts().to_dict())
    print(json.dumps(q)); print(json.dumps(val, indent=1))


if __name__ == "__main__":
    sys.exit(main())
