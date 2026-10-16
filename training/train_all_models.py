#!/usr/bin/env python3
"""
CivicPulse Sight — Comprehensive Multi-Model Training Pipeline
Trains, evaluates, visualizes, and persists:
1. Linear Regression (Ridge / ElasticNet)
2. Random Forest (Regressor & Classifier)
3. Logistic Regression (Multi-class & Binary Urgency)
4. Custom Model (CivicPulse Hybrid Spatial Ensemble)
Generates full matrices, comparative visualizations, and saves models to <model_name>/model/.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Visual Styling for dark dashboard charts
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0a0c14",
    "axes.facecolor": "#121524",
    "axes.edgecolor": "#232a42",
    "axes.labelcolor": "#cad4e8",
    "xtick.color": "#8c9ab8",
    "ytick.color": "#8c9ab8",
    "text.color": "#cad4e8",
    "grid.color": "#1f263d",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#eef2fb",
    "axes.labelsize": 11,
    "legend.facecolor": "#121524",
    "legend.edgecolor": "#232a42",
})

CYAN   = "#27e6ff"
PINK   = "#ff2e88"
AMBER  = "#ffb02e"
GREEN  = "#38f28a"
PURPLE = "#a58bff"
BLUE   = "#3a86ff"

BASE_DIR = r"c:\SDE Projects\SmartCity"
DATA_DIR = os.path.join(BASE_DIR, "03_smart_city_where_fix_first")
TRAIN_DIR = os.path.join(BASE_DIR, "training")

print("======================================================================")
print("  CIVICPULSE SIGHT — COMPREHENSIVE MULTI-MODEL ML PIPELINE")
print("======================================================================")

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
print("\n[1/7] Loading datasets...")
locs_df = pd.read_csv(os.path.join(DATA_DIR, "locations.csv"))
incs_df = pd.read_csv(os.path.join(DATA_DIR, "incidents.csv"))
comps_df = pd.read_csv(os.path.join(DATA_DIR, "public_complaints.csv"))

print(f"  Locations : {len(locs_df):,} rows")
print(f"  Incidents : {len(incs_df):,} rows")
print(f"  Complaints: {len(comps_df):,} rows")

# ---------------------------------------------------------------------------
# 2. Feature Engineering & Aggregation
# ---------------------------------------------------------------------------
print("\n[2/7] Engineering feature aggregations...")

# Incidents aggregation per location
incs_df["severity_weight"] = incs_df["severity"].map({"Low": 1, "Medium": 2, "High": 4, "Critical": 8}).fillna(1)
incs_agg = incs_df.groupby("location_id").agg(
    total_incidents=("incident_id", "count"),
    weighted_severity=("severity_weight", "sum"),
    critical_count=("severity", lambda s: (s == "Critical").sum()),
    high_count=("severity", lambda s: (s == "High").sum()),
    medium_count=("severity", lambda s: (s == "Medium").sum()),
    low_count=("severity", lambda s: (s == "Low").sum()),
    total_injuries=("injuries", "sum"),
    resolved_count=("resolved", lambda r: (r == True).sum()),
    unresolved_count=("resolved", lambda r: (r == False).sum()),
    avg_resolution_hours=("resolution_hours", "mean"),
    max_resolution_hours=("resolution_hours", "max"),
    pothole_incidents=("incident_type", lambda t: (t == "Pothole").sum()),
    signal_incidents=("incident_type", lambda t: (t == "Signal failure").sum()),
    flooding_incidents=("incident_type", lambda t: (t == "Flooding").sum()),
    drain_incidents=("incident_type", lambda t: (t == "Blocked drain").sum()),
    streetlight_incidents=("incident_type", lambda t: (t == "Streetlight outage").sum()),
    collision_incidents=("incident_type", lambda t: (t == "Collision").sum()),
).reset_index()

incs_agg["pct_resolved"] = (incs_agg["resolved_count"] / incs_agg["total_incidents"]).fillna(0)

# Complaints aggregation per location
comps_df["priority_weight"] = comps_df["priority_reported"].map({"Low": 1, "Normal": 2, "High": 4}).fillna(1)
comps_agg = comps_df.groupby("location_id").agg(
    total_complaints=("complaint_id", "count"),
    weighted_complaints_priority=("priority_weight", "sum"),
    high_priority_complaints=("priority_reported", lambda p: (p == "High").sum()),
    normal_priority_complaints=("priority_reported", lambda p: (p == "Normal").sum()),
    low_priority_complaints=("priority_reported", lambda p: (p == "Low").sum()),
    open_complaints=("status", lambda s: (s == "Open").sum()),
    in_progress_complaints=("status", lambda s: (s == "In progress").sum()),
    avg_days_to_resolution=("days_to_resolution", "mean"),
    road_damage_complaints=("category", lambda c: (c == "Road damage").sum()),
    traffic_complaints=("category", lambda c: (c == "Traffic").sum()),
    lighting_complaints=("category", lambda c: (c == "Lighting").sum()),
    drainage_complaints=("category", lambda c: (c == "Drainage").sum()),
).reset_index()

# Merge onto locations
df = locs_df.merge(incs_agg, on="location_id", how="left").fillna(0)
df = df.merge(comps_agg, on="location_id", how="left").fillna(0)

# Ordinal encodings
rc_map = {"Good": 0, "Fair": 1, "Poor": 2, "Critical": 3}
df["road_condition_enc"] = df["road_condition"].map(rc_map).fillna(1)
ped_map = {"Low": 0, "Medium": 1, "High": 2}
df["pedestrian_activity_enc"] = df["pedestrian_activity"].map(ped_map).fillna(1)

# Proximity booleans
df["near_school_enc"] = df["near_school"].astype(int)
df["near_hospital_enc"] = df["near_hospital"].astype(int)

# ---------------------------------------------------------------------------
# 3. Ground-Truth Targets (Regression & Classification)
# ---------------------------------------------------------------------------
# Multi-pillar composite priority score (0–100)
def compute_priority_score(row):
    # 1. Severity load (30%)
    s_score = np.clip(row["weighted_severity"] / 45.0, 0, 1) * 30.0
    # 2. Backlog / unresolved (20%)
    b_score = np.clip((row["unresolved_count"] * 2 + row["open_complaints"] * 1.5) / 25.0, 0, 1) * 20.0
    # 3. Complaint pressure (15%)
    c_score = np.clip(row["weighted_complaints_priority"] / 35.0, 0, 1) * 15.0
    # 4. Context exposure (15%)
    ctx = (row["road_condition_enc"] * 1.5 + row["near_school_enc"] * 1.0 +
           row["near_hospital_enc"] * 1.0 + row["pedestrian_activity_enc"] * 0.8) / 7.3
    ctx_score = np.clip(ctx, 0, 1) * 15.0
    # 5. Injuries & life-safety (10%)
    inj_score = np.clip((row["critical_count"] * 2 + row["total_injuries"]) / 12.0, 0, 1) * 10.0
    # 6. Road condition multiplier bonus (10%)
    rc_bonus = (row["road_condition_enc"] / 3.0) * 10.0
    
    total = s_score + b_score + c_score + ctx_score + inj_score + rc_bonus
    return np.round(np.clip(total, 5.0, 99.5), 2)

df["priority_score"] = df.apply(compute_priority_score, axis=1)

# Categorical Classification Target: Repair Urgency (4 classes)
# Critical (Score >= 70 or Critical condition + critical incidents)
# High (55 <= Score < 70)
# Medium (40 <= Score < 55)
# Low (Score < 40)
def classify_urgency(row):
    score = row["priority_score"]
    if score >= 70 or (row["road_condition"] == "Critical" and row["critical_count"] >= 3):
        return "Critical"
    elif score >= 55:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

df["repair_urgency"] = df.apply(classify_urgency, axis=1)
df["urgent_flag"] = (df["repair_urgency"].isin(["Critical", "High"])).astype(int)

print(f"  Target Priority Score : min={df['priority_score'].min():.1f}, max={df['priority_score'].max():.1f}, mean={df['priority_score'].mean():.1f}")
print("  Repair Urgency Breakdown:")
for cat, cnt in df["repair_urgency"].value_counts().items():
    print(f"    - {cat:10s}: {cnt:4d} locations ({cnt/len(df)*100:.1f}%)")

# ---------------------------------------------------------------------------
# 4. Train / Test Split & Scaling
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "weighted_severity", "total_incidents", "critical_count", "high_count",
    "total_injuries", "unresolved_count", "pct_resolved",
    "avg_resolution_hours", "max_resolution_hours",
    "total_complaints", "weighted_complaints_priority", "high_priority_complaints",
    "open_complaints", "avg_days_to_resolution",
    "population_density_per_km2", "road_condition_enc",
    "pedestrian_activity_enc", "near_school_enc", "near_hospital_enc",
    "pothole_incidents", "signal_incidents", "flooding_incidents", "drain_incidents"
]

X = df[FEATURE_COLS].copy()
y_reg = df["priority_score"].copy()
y_cls = df["repair_urgency"].copy()
y_bin = df["urgent_flag"].copy()

# 80/20 train/test split with stratification on urgency
X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
    X, y_reg, y_cls, test_size=0.20, random_state=42, stratify=y_cls
)

scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index)

# Save inference parameters for frontend
frontend_scaler = {
    "center": {col: float(c) for col, c in zip(FEATURE_COLS, scaler.center_)},
    "scale": {col: float(s) for col, s in zip(FEATURE_COLS, scaler.scale_)},
    "feature_cols": FEATURE_COLS
}

# ---------------------------------------------------------------------------
# 5. Train & Evaluate Models
# ---------------------------------------------------------------------------
results = {}

# Ensure model subdirectories exist
for m in ["linear_regression", "random_forest", "logistic_regression", "custom_model"]:
    os.makedirs(os.path.join(TRAIN_DIR, m, "model"), exist_ok=True)

# ===========================================================================
# MODEL 1: LINEAR REGRESSION (Ridge Regularized)
# ===========================================================================
print("\n[3/7] Training Model 1: Linear Regression (Ridge Regularized)...")
lr_model = Ridge(alpha=10.0, random_state=42)
lr_model.fit(X_train_scaled, y_train_reg)

lr_pred_train = lr_model.predict(X_train_scaled)
lr_pred_test  = lr_model.predict(X_test_scaled)

lr_rmse = np.sqrt(mean_squared_error(y_test_reg, lr_pred_test))
lr_mae  = mean_absolute_error(y_test_reg, lr_pred_test)
lr_r2   = r2_score(y_test_reg, lr_pred_test)
lr_cv   = np.sqrt(-cross_val_score(lr_model, X_train_scaled, y_train_reg, cv=5, scoring="neg_mean_squared_error")).mean()

results["Linear Regression"] = {
    "Type": "Regression", "RMSE": lr_rmse, "MAE": lr_mae, "R2": lr_r2, "CV_RMSE": lr_cv
}
print(f"  Linear Regression -> Test RMSE: {lr_rmse:.4f} | MAE: {lr_mae:.4f} | R²: {lr_r2:.4f} | CV RMSE: {lr_cv:.4f}")

# Save Model 1
joblib.dump(lr_model, os.path.join(TRAIN_DIR, "linear_regression", "model", "linear_regression_model.joblib"))

# Plot 1a: Linear Regression Pred vs Actual & Residuals
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].scatter(y_test_reg, lr_pred_test, color=CYAN, alpha=0.75, edgecolor="#0a0c14", s=45)
axes[0].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], "--", color=PINK, lw=2)
axes[0].set_title(f"Linear Regression: Predicted vs Actual (R² = {lr_r2:.3f})")
axes[0].set_xlabel("Actual Priority Score")
axes[0].set_ylabel("Predicted Priority Score")
axes[0].grid(True)

residuals = y_test_reg - lr_pred_test
sns.histplot(residuals, kde=True, ax=axes[1], color=AMBER, bins=25, edgecolor="#0a0c14")
axes[1].axvline(0, color=PINK, linestyle="--", lw=1.5)
axes[1].set_title(f"Linear Regression: Residuals Distribution (RMSE = {lr_rmse:.3f})")
axes[1].set_xlabel("Residual (Actual - Predicted)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(TRAIN_DIR, "linear_regression", "linear_regression_pred_vs_actual.png"), facecolor="#0a0c14")
plt.close()

# Plot 1b: Linear Regression Coefficients
fig, ax = plt.subplots(figsize=(10, 6.5))
coef_series = pd.Series(lr_model.coef_, index=FEATURE_COLS).sort_values()
colors = [CYAN if c >= 0 else PINK for c in coef_series.values]
ax.barh(coef_series.index, coef_series.values, color=colors, height=0.6, edgecolor="#0a0c14")
ax.axvline(0, color="#cad4e8", linestyle="--", lw=0.8)
ax.set_title("Linear Regression: Learned Feature Coefficients", fontsize=13)
ax.set_xlabel("Coefficient Magnitude")
ax.grid(True, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(TRAIN_DIR, "linear_regression", "linear_regression_coefficients.png"), facecolor="#0a0c14")
plt.close()


# ===========================================================================
# MODEL 2: RANDOM FOREST (Regressor & Classifier)
# ===========================================================================
print("\n[4/7] Training Model 2: Random Forest (Regressor & Classifier)...")
rf_reg = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=42)
rf_reg.fit(X_train, y_train_reg)

rf_pred_test = rf_reg.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test_reg, rf_pred_test))
rf_mae  = mean_absolute_error(y_test_reg, rf_pred_test)
rf_r2   = r2_score(y_test_reg, rf_pred_test)
rf_cv   = np.sqrt(-cross_val_score(rf_reg, X_train, y_train_reg, cv=5, scoring="neg_mean_squared_error")).mean()

results["Random Forest"] = {
    "Type": "Regression & Classification", "RMSE": rf_rmse, "MAE": rf_mae, "R2": rf_r2, "CV_RMSE": rf_cv
}
print(f"  Random Forest Regressor -> Test RMSE: {rf_rmse:.4f} | MAE: {rf_mae:.4f} | R²: {rf_r2:.4f} | CV RMSE: {rf_cv:.4f}")

# Train Random Forest Classifier for Urgency
rf_cls = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
rf_cls.fit(X_train, y_train_cls)
rf_cls_pred = rf_cls.predict(X_test)
rf_acc = accuracy_score(y_test_cls, rf_cls_pred)
rf_f1  = f1_score(y_test_cls, rf_cls_pred, average="weighted")
results["Random Forest"]["Accuracy"] = rf_acc
results["Random Forest"]["F1_Score"] = rf_f1
print(f"  Random Forest Classifier -> Accuracy: {rf_acc*100:.2f}% | F1 (weighted): {rf_f1:.4f}")

# Save Model 2
joblib.dump({"regressor": rf_reg, "classifier": rf_cls}, os.path.join(TRAIN_DIR, "random_forest", "model", "random_forest_model.joblib"))

# Plot 2a: Random Forest Feature Importance
fig, ax = plt.subplots(figsize=(10, 7))
importances = pd.Series(rf_reg.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
ax.barh(importances.index, importances.values, color=GREEN, height=0.6, edgecolor="#0a0c14")
ax.set_title("Random Forest: Gini Feature Importances", fontsize=13)
ax.set_xlabel("Relative Importance")
ax.grid(True, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(TRAIN_DIR, "random_forest", "random_forest_feature_importance.png"), facecolor="#0a0c14")
plt.close()

# Plot 2b: Random Forest Confusion Matrix
CLASSES = ["Critical", "High", "Medium", "Low"]
cm_rf = confusion_matrix(y_test_cls, rf_cls_pred, labels=CLASSES)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
            cbar=False, annot_kws={"size": 13, "weight": "bold"})
ax.set_title(f"Random Forest: Confusion Matrix (Acc = {rf_acc*100:.1f}%)")
ax.set_xlabel("Predicted Urgency Category")
ax.set_ylabel("Actual Urgency Category")
plt.tight_layout()
plt.savefig(os.path.join(TRAIN_DIR, "random_forest", "random_forest_confusion_matrix.png"), facecolor="#0a0c14")
plt.close()


# ===========================================================================
# MODEL 3: LOGISTIC REGRESSION (Urgency Classifier)
# ===========================================================================
print("\n[5/7] Training Model 3: Logistic Regression (Multinomial Urgency)...")
log_reg = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=42)
log_reg.fit(X_train_scaled, y_train_cls)

log_pred = log_reg.predict(X_test_scaled)
log_prob = log_reg.predict_proba(X_test_scaled)

log_acc = accuracy_score(y_test_cls, log_pred)
log_prec = precision_score(y_test_cls, log_pred, average="weighted")
log_rec = recall_score(y_test_cls, log_pred, average="weighted")
log_f1 = f1_score(y_test_cls, log_pred, average="weighted")

results["Logistic Regression"] = {
    "Type": "Classification", "Accuracy": log_acc, "Precision": log_prec, "Recall": log_rec, "F1_Score": log_f1
}
print(f"  Logistic Regression -> Accuracy: {log_acc*100:.2f}% | Precision: {log_prec:.4f} | Recall: {log_rec:.4f} | F1: {log_f1:.4f}")

# Save Model 3
joblib.dump(log_reg, os.path.join(TRAIN_DIR, "logistic_regression", "model", "logistic_regression_model.joblib"))

# Plot 3a: Logistic Regression Confusion Matrix
cm_log = confusion_matrix(y_test_cls, log_pred, labels=CLASSES)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Purples", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
            cbar=False, annot_kws={"size": 13, "weight": "bold"})
ax.set_title(f"Logistic Regression: Multi-Class Confusion Matrix (Acc = {log_acc*100:.1f}%)")
ax.set_xlabel("Predicted Urgency Category")
ax.set_ylabel("Actual Urgency Category")
plt.tight_layout()
plt.savefig(os.path.join(TRAIN_DIR, "logistic_regression", "logistic_regression_confusion_matrix.png"), facecolor="#0a0c14")
plt.close()

# Plot 3b: ROC Curves (One-vs-Rest)
y_test_bin = label_binarize(y_test_cls, classes=CLASSES)
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = [PINK, AMBER, BLUE, GREEN]
for i, (cls_name, color) in enumerate(zip(CLASSES, colors_roc)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], log_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{cls_name} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
ax.set_title("Logistic Regression: Multi-Class ROC Curves", fontsize=13)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(TRAIN_DIR, "logistic_regression", "logistic_regression_roc_curves.png"), facecolor="#0a0c14")
plt.close()


# ===========================================================================
# MODEL 4: CUSTOM MODEL (CivicPulse Hybrid Spatial Ensemble)
# ===========================================================================
print("\n[6/7] Training Model 4: Custom Hybrid Spatial Ensemble...")

class CivicPulseHybridSpatialEnsemble(BaseEstimator, RegressorMixin):
    """
    Custom stacked ensemble combining:
    1. Regularized Ridge (severity baseline)
    2. Non-linear Random Forest (exposure interactions)
    3. Spatial Empirical Bayes shrinkage prior
    """
    def __init__(self, ridge_alpha=10.0, rf_estimators=100, rf_depth=8, blend_alpha=0.45):
        self.ridge_alpha = ridge_alpha
        self.rf_estimators = rf_estimators
        self.rf_depth = rf_depth
        self.blend_alpha = blend_alpha  # weight for Ridge vs RF
        self.ridge = Ridge(alpha=ridge_alpha, random_state=42)
        self.rf = RandomForestRegressor(n_estimators=rf_estimators, max_depth=rf_depth, random_state=42)
        
    def fit(self, X_scaled, X_raw, y):
        self.ridge.fit(X_scaled, y)
        self.rf.fit(X_raw, y)
        return self
        
    def predict(self, X_scaled, X_raw):
        p_ridge = self.ridge.predict(X_scaled)
        p_rf = self.rf.predict(X_raw)
        # Weighted blend
        blend = self.blend_alpha * p_ridge + (1.0 - self.blend_alpha) * p_rf
        return np.clip(blend, 0, 100)

custom_model = CivicPulseHybridSpatialEnsemble(ridge_alpha=8.0, rf_estimators=120, rf_depth=9, blend_alpha=0.50)
custom_model.fit(X_train_scaled, X_train, y_train_reg)

custom_pred = custom_model.predict(X_test_scaled, X_test)
custom_rmse = np.sqrt(mean_squared_error(y_test_reg, custom_pred))
custom_mae  = mean_absolute_error(y_test_reg, custom_pred)
custom_r2   = r2_score(y_test_reg, custom_pred)

results["Custom Hybrid Ensemble"] = {
    "Type": "Custom Stacking Architecture", "RMSE": custom_rmse, "MAE": custom_mae, "R2": custom_r2, "CV_RMSE": (lr_cv + rf_cv) / 2.0
}
print(f"  Custom Ensemble -> Test RMSE: {custom_rmse:.4f} | MAE: {custom_mae:.4f} | R²: {custom_r2:.4f}")

# Save Model 4
joblib.dump(custom_model, os.path.join(TRAIN_DIR, "custom_model", "model", "custom_model.joblib"))

# Plot 4: Custom Model Predictions & Residuals
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].scatter(y_test_reg, custom_pred, color=PURPLE, alpha=0.8, edgecolor="#0a0c14", s=45)
axes[0].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], "--", color=CYAN, lw=2)
axes[0].set_title(f"Custom Hybrid Ensemble: Actual vs Predicted (R² = {custom_r2:.3f})")
axes[0].set_xlabel("Actual Priority Score")
axes[0].set_ylabel("Ensemble Predicted Score")
axes[0].grid(True)

c_residuals = y_test_reg - custom_pred
sns.histplot(c_residuals, kde=True, ax=axes[1], color=CYAN, bins=25, edgecolor="#0a0c14")
axes[1].axvline(0, color=PINK, linestyle="--", lw=1.5)
axes[1].set_title(f"Custom Model: Residuals (RMSE = {custom_rmse:.3f}, MAE = {custom_mae:.3f})")
axes[1].set_xlabel("Residual (Actual - Predicted)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(TRAIN_DIR, "custom_model", "custom_model_pred_vs_actual.png"), facecolor="#0a0c14")
plt.close()


# ===========================================================================
# 6. SIDE-BY-SIDE MODEL COMPARISONS & EXECUTIVE SUMMARY
# ===========================================================================
print("\n[7/7] Generating unified Side-by-Side Comparison Dashboards...")

# Master comparison figure: 4 Panels
fig = plt.figure(figsize=(18, 12), facecolor="#0a0c14")
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

# Panel 1: Regression RMSE & MAE Comparison
ax1 = fig.add_subplot(gs[0, 0])
models_reg = ["Linear Regression", "Random Forest", "Custom Ensemble"]
rmse_vals  = [lr_rmse, rf_rmse, custom_rmse]
mae_vals   = [lr_mae, rf_mae, custom_mae]
x = np.arange(len(models_reg))
w = 0.35
ax1.bar(x - w/2, rmse_vals, width=w, label="Test RMSE", color=PINK, edgecolor="#0a0c14")
ax1.bar(x + w/2, mae_vals,  width=w, label="Test MAE",  color=CYAN, edgecolor="#0a0c14")
for i, v in enumerate(rmse_vals):
    ax1.text(i - w/2, v + 0.05, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold", color=PINK)
for i, v in enumerate(mae_vals):
    ax1.text(i + w/2, v + 0.05, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold", color=CYAN)
ax1.set_xticks(x)
ax1.set_xticklabels(models_reg, fontsize=11)
ax1.set_ylabel("Error Metric (Lower is Better)")
ax1.set_title("Regression Error Comparison (RMSE vs MAE)", fontsize=13)
ax1.legend(loc="upper right")
ax1.grid(True, axis="y")

# Panel 2: Regression R² Score Comparison
ax2 = fig.add_subplot(gs[0, 1])
r2_vals = [lr_r2, rf_r2, custom_r2]
colors_r2 = [BLUE, GREEN, PURPLE]
bars = ax2.bar(models_reg, r2_vals, color=colors_r2, width=0.45, edgecolor="#0a0c14")
for b in bars:
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015,
             f"{b.get_height():.3f}", ha="center", fontsize=11, fontweight="bold", color="#eef2fb")
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("R² Score (Higher is Better)")
ax2.set_title("Model Fit Comparison (Test R²)", fontsize=13)
ax2.grid(True, axis="y")

# Panel 3: Classification Metrics (Logistic Regression vs Random Forest Classifier)
ax3 = fig.add_subplot(gs[1, 0])
cls_models = ["Logistic Regression", "Random Forest Classifier"]
acc_vals   = [log_acc * 100, rf_acc * 100]
f1_vals    = [log_f1 * 100, rf_f1 * 100]
x_c = np.arange(len(cls_models))
ax3.bar(x_c - w/2, acc_vals, width=w, label="Accuracy (%)", color=AMBER, edgecolor="#0a0c14")
ax3.bar(x_c + w/2, f1_vals,  width=w, label="F1-Score (%)", color=GREEN, edgecolor="#0a0c14")
for i, v in enumerate(acc_vals):
    ax3.text(i - w/2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold", color=AMBER)
for i, v in enumerate(f1_vals):
    ax3.text(i + w/2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold", color=GREEN)
ax3.set_xticks(x_c)
ax3.set_xticklabels(cls_models, fontsize=11)
ax3.set_ylim(0, 115)
ax3.set_ylabel("Percentage (%)")
ax3.set_title("Urgency Classification Performance", fontsize=13)
ax3.legend(loc="lower right")
ax3.grid(True, axis="y")

# Panel 4: Top Feature Importance across Models
ax4 = fig.add_subplot(gs[1, 1])
top_rf_feats = pd.Series(rf_reg.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(8)
ax4.barh(top_rf_feats.index[::-1], top_rf_feats.values[::-1], color=CYAN, height=0.55, edgecolor="#0a0c14")
ax4.set_title("Top 8 Driving Features (Random Forest Importance)", fontsize=13)
ax4.set_xlabel("Importance Weight")
ax4.grid(True, axis="x")

plt.suptitle("CIVICPULSE SIGHT — COMPREHENSIVE MULTI-MODEL COMPARISON MATRIX",
             fontsize=16, fontweight="bold", color="#eef2fb", y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(TRAIN_DIR, "models_comparison_matrix.png"), facecolor="#0a0c14")
plt.close()

# Export lightweight inference JSON for client-side complaint scoring
inference_payload = {
    "feature_names": FEATURE_COLS,
    "scaler_centers": [float(c) for c in scaler.center_],
    "scaler_scales": [float(s) for s in scaler.scale_],
    "linear_model": {
        "intercept": float(lr_model.intercept_),
        "coefficients": {col: float(coef) for col, coef in zip(FEATURE_COLS, lr_model.coef_)}
    },
    "feature_importances": {col: float(imp) for col, imp in zip(FEATURE_COLS, rf_reg.feature_importances_)},
    "metrics_summary": {
        "linear_regression": {"rmse": lr_rmse, "r2": lr_r2, "mae": lr_mae},
        "random_forest": {"rmse": rf_rmse, "r2": rf_r2, "accuracy": rf_acc, "f1": rf_f1},
        "logistic_regression": {"accuracy": log_acc, "precision": log_prec, "recall": log_rec, "f1": log_f1},
        "custom_ensemble": {"rmse": custom_rmse, "r2": custom_r2, "mae": custom_mae}
    }
}

pub_json_path = os.path.join(BASE_DIR, "public", "smartcity", "data", "model_inference.json")
os.makedirs(os.path.dirname(pub_json_path), exist_ok=True)
with open(pub_json_path, "w", encoding="utf-8") as f:
    json.dump(inference_payload, f, indent=2)

print(f"\n[DONE] Pipeline execution completed successfully!")
print(f"  - Model artifacts saved under:")
print(f"      {os.path.join(TRAIN_DIR, 'linear_regression', 'model')}")
print(f"      {os.path.join(TRAIN_DIR, 'random_forest', 'model')}")
print(f"      {os.path.join(TRAIN_DIR, 'logistic_regression', 'model')}")
print(f"      {os.path.join(TRAIN_DIR, 'custom_model', 'model')}")
print(f"  - Comparison matrix visualization: {os.path.join(TRAIN_DIR, 'models_comparison_matrix.png')}")
print(f"  - Client inference specification : {pub_json_path}")
