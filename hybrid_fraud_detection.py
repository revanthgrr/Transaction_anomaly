# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # 🧠 Hybrid Adaptive Fraud Detection System
#
# **Intelligent fusion of Rule-Based, Unsupervised, and Supervised Learning**
#
# This notebook runs the complete hybrid fraud detection pipeline:
# 1. **Data Understanding** — Auto-profile any dataset (schema-agnostic)
# 2. **Feature Engineering** — 15+ derived features (z-scores, velocity, geo, temporal)
# 3. **Rule Engine** — Dynamic statistical rules (P95, IQR, z-score thresholds)
# 4. **Unsupervised Models** — Isolation Forest + HDBSCAN + LOF
# 5. **Supervised Models** — XGBoost + LightGBM + CatBoost (GPU⚡) with Optuna tuning
# 6. **Fusion Layer** — Meta-classifier stacking for optimal combination
# 7. **Explainability** — SHAP values + rule trigger explanations
#
# ---
#
# **⚠️ IMPORTANT**: Set Runtime → Change runtime type → **GPU** for best performance!

# %% [markdown]
# ## 1. Environment Setup

# %%
# Install all dependencies (run this cell in Colab)
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "xgboost", "lightgbm", "catboost", "optuna", "shap", "hdbscan",
    "imbalanced-learn", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "pyyaml", "scipy"])

# %%
# GPU Check
import subprocess
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ GPU DETECTED:")
        print(result.stdout[:500])
    else:
        print("⚠️ No GPU detected. Running in CPU mode (still works, just slower)")
except FileNotFoundError:
    print("⚠️ No GPU detected. Running in CPU mode.")

# %% [markdown]
# ## 2. Get the Code
#
# **Option A**: Clone from GitHub (recommended)
#
# **Option B**: Upload files manually

# %%
# === OPTION A: Clone from GitHub ===
# Uncomment and replace with your actual repo URL:

# !git clone https://github.com/YOUR_USERNAME/ha.git /content/ha 2>/dev/null || echo "Repo already exists"

# === OPTION B: If files are already uploaded or mounted ===
# (Skip this cell if you cloned)

# %%
# Set up Python paths
import sys
import os

# Try common locations
for path in ["/content/ha/models", "/content/models", "./models"]:
    if os.path.isdir(path):
        if path not in sys.path:
            sys.path.insert(0, path)
        print(f"✅ Found models at: {path}")
        break
else:
    print("⚠️ models/ directory not found. Upload or clone the repo first.")

# %% [markdown]
# ## 3. Upload Dataset
#
# Upload your CSV file (e.g., `fraudTest.csv`), or mount Google Drive.

# %%
# === OPTION A: Upload via file picker ===
from google.colab import files

print("📁 Select your CSV file to upload...")
uploaded = files.upload()
DATASET_PATH = list(uploaded.keys())[0] if uploaded else None
print(f"✅ Uploaded: {DATASET_PATH}") if DATASET_PATH else None

# %%
# === OPTION B: Mount Google Drive ===
# Uncomment if your data is on Drive:

# from google.colab import drive
# drive.mount('/content/drive')
# DATASET_PATH = "/content/drive/MyDrive/path/to/fraudTest.csv"

# %%
# === OPTION C: Direct path (if data is already in Colab) ===
# DATASET_PATH = "/content/ha/data/fraudTest.csv"

# %% [markdown]
# ## 4. Run Hybrid Detection Pipeline

# %%
import pandas as pd
import numpy as np
import time

# Load dataset
print(f"📊 Loading: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH, low_memory=False)

# Drop unnamed index columns
unnamed = [c for c in df.columns if "Unnamed" in c]
if unnamed:
    df.drop(columns=unnamed, inplace=True)

print(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
print(f"📋 Columns: {list(df.columns)}")
df.head()

# %%
# Initialize and run the Hybrid Fraud Detector
from hybrid_engine import HybridFraudDetector

detector = HybridFraudDetector(
    n_optuna_trials=20,       # Optuna trials per model (20 = standard)
    shap_sample_size=5000,    # SHAP subsample size
    random_state=42,
)

# 🚀 RUN THE FULL PIPELINE
result = detector.detect(df)

# %% [markdown]
# ## 5. Results & Metrics Dashboard

# %%
# === PERFORMANCE METRICS ===
print("=" * 60)
print("  📊 PERFORMANCE METRICS")
print("=" * 60)

if result.metrics:
    for key, val in result.metrics.items():
        if isinstance(val, float):
            print(f"  {key:<24s} {val:>10.4f}")
        elif isinstance(val, int):
            print(f"  {key:<24s} {val:>10,}")
else:
    print("  No ground truth labels found — metrics unavailable")

print(f"\n  Best Model: {result.best_model_name}")
print(f"  GPU Used: {result.gpu_used}")
print(f"  Total Time: {result.total_time_seconds:.1f}s")

# %%
# === MODEL COMPARISON TABLE ===
if result.supervised_result and result.supervised_result.all_model_scores:
    print("\n" + "=" * 60)
    print("  🏆 MODEL COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<12s} {'ROC-AUC':>10s} {'PR-AUC':>10s} {'F1':>10s}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    for name, scores in result.supervised_result.all_model_scores.items():
        is_best = " ⭐" if name == result.best_model_name else ""
        print(f"  {name:<12s} {scores['roc_auc']:>10.4f} "
              f"{scores['pr_auc']:>10.4f} {scores['f1']:>10.4f}{is_best}")

# %%
# === FUSION WEIGHTS ===
print("\n" + "=" * 60)
print("  ⚖️  FUSION WEIGHTS")
print("=" * 60)
for key, val in result.fusion_weights.items():
    bar = "█" * int(val * 30) if isinstance(val, float) else ""
    val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
    print(f"  {key:<20s} {val_str:>8s}  {bar}")

# %% [markdown]
# ## 6. Visualizations

# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Hybrid Fraud Detection Results", fontsize=16, fontweight="bold")

# 1. Score Distribution
ax = axes[0, 0]
scores = result.final_scores
ax.hist(scores[result.final_predictions == 0], bins=50, alpha=0.7,
        label="Normal", color="#2ecc71")
ax.hist(scores[result.final_predictions == 1], bins=50, alpha=0.7,
        label="Fraud", color="#e74c3c")
ax.set_xlabel("Final Fraud Score")
ax.set_ylabel("Count")
ax.set_title("Score Distribution")
ax.legend()

# 2. Confusion Matrix (if labels exist)
ax = axes[0, 1]
label_col = result.profile.detected_label_col
if label_col and label_col in result.scored_df.columns:
    from sklearn.metrics import confusion_matrix
    y_true = result.scored_df[label_col].astype(int)
    cm = confusion_matrix(y_true, result.final_predictions)
    sns.heatmap(cm, annot=True, fmt=",d", cmap="RdYlGn_r",
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
else:
    ax.text(0.5, 0.5, "No ground truth\nlabels available",
            ha="center", va="center", fontsize=12)
    ax.set_title("Confusion Matrix")

# 3. ROC Curve
ax = axes[0, 2]
if label_col and label_col in result.scored_df.columns:
    from sklearn.metrics import roc_curve
    y_true = result.scored_df[label_col].astype(int)
    fpr, tpr, _ = roc_curve(y_true, result.final_scores)
    roc_auc = result.metrics.get("roc_auc", 0)
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
else:
    ax.text(0.5, 0.5, "No ground truth", ha="center", va="center")
    ax.set_title("ROC Curve")

# 4. Subsystem Score Comparison
ax = axes[1, 0]
subsystem_scores = {
    "Rule Engine": result.scored_df["_anomaly_score"].mean(),
    "Unsupervised": result.scored_df["_unsupervised_score"].mean(),
    "Supervised": result.scored_df["_supervised_prob"].mean(),
    "Fusion": result.scored_df["_final_fraud_score"].mean(),
}
colors = ["#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
ax.bar(subsystem_scores.keys(), subsystem_scores.values(), color=colors)
ax.set_ylabel("Mean Score")
ax.set_title("Mean Scores by Subsystem")
plt.setp(ax.get_xticklabels(), rotation=15)

# 5. Feature Importance (top 10)
ax = axes[1, 1]
if result.supervised_result and result.supervised_result.feature_importances:
    imp = dict(list(result.supervised_result.feature_importances.items())[:10])
    ax.barh(list(imp.keys()), list(imp.values()), color="#3498db")
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Feature Importance")
    ax.invert_yaxis()
else:
    ax.text(0.5, 0.5, "No feature importance", ha="center", va="center")
    ax.set_title("Feature Importance")

# 6. Fusion Weight Pie Chart
ax = axes[1, 2]
# Get core weights only (rule, unsupervised, supervised)
pie_weights = {}
for k, v in result.fusion_weights.items():
    if k in ("rule", "unsupervised", "supervised") and isinstance(v, (int, float)):
        pie_weights[k] = abs(v)
if pie_weights:
    ax.pie(pie_weights.values(), labels=pie_weights.keys(), autopct="%1.1f%%",
           colors=["#3498db", "#9b59b6", "#e67e22"])
    ax.set_title("Fusion Weights")
else:
    ax.text(0.5, 0.5, "Meta-classifier\n(non-linear weights)",
            ha="center", va="center")
    ax.set_title("Fusion Strategy")

plt.tight_layout()
plt.savefig("hybrid_results_visualization.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Visualization saved!")

# %% [markdown]
# ## 7. Top Suspicious Records

# %%
# Show top 20 most suspicious records
top = result.scored_df.nlargest(20, "_final_fraud_score")
display_cols = ["_final_fraud_score", "_anomaly_score",
                "_unsupervised_score", "_supervised_prob",
                "_final_prediction"]

# Add original columns that exist
for c in [result.profile.detected_entity_col,
          result.profile.detected_amount_col,
          result.profile.detected_label_col]:
    if c and c in top.columns:
        display_cols.insert(0, c)

top[display_cols].head(20)

# %% [markdown]
# ## 8. Explainability — Sample Row Explanations

# %%
# Show detailed explanation for top 5 flagged records
print("=" * 60)
print("  🔍 DETAILED EXPLANATIONS (Top 5)")
print("=" * 60)

top_indices = result.scored_df.nlargest(5, "_final_fraud_score").index

for rank, idx in enumerate(top_indices, 1):
    if idx < len(result.row_explanations):
        expl = result.row_explanations[idx]
        print(f"\n{'─'*50}")
        print(f"  Record #{rank}  |  Final Score: {expl.final_score:.4f}  |  "
              f"Confidence: {expl.confidence}")
        print(f"{'─'*50}")
        print(f"  Rule:         {expl.rule_contribution.get('score', 0):.4f} "
              f"× {expl.rule_contribution.get('weight', 0):.3f} = "
              f"{expl.rule_contribution.get('weighted_contribution', 0):.4f}")
        print(f"  Unsupervised: {expl.unsupervised_contribution.get('score', 0):.4f} "
              f"× {expl.unsupervised_contribution.get('weight', 0):.3f} = "
              f"{expl.unsupervised_contribution.get('weighted_contribution', 0):.4f}")
        print(f"  Supervised:   {expl.supervised_contribution.get('probability', 0):.4f} "
              f"× {expl.supervised_contribution.get('weight', 0):.3f} = "
              f"{expl.supervised_contribution.get('weighted_contribution', 0):.4f}")

        if expl.rule_triggers:
            print(f"\n  Rule Triggers:")
            for t in expl.rule_triggers[:3]:
                if isinstance(t, dict):
                    print(f"    • {t.get('reason', 'N/A')}")

        if expl.top_shap_features:
            print(f"\n  SHAP Features:")
            for f in expl.top_shap_features[:3]:
                print(f"    • {f.get('feature', '?')}: "
                      f"{f.get('shap_value', 0):.4f} ({f.get('direction', '?')})")

# %% [markdown]
# ## 9. Save & Download Results

# %%
# Save all results
OUTPUT_DIR = "results/hybrid_output"
detector.save_results(OUTPUT_DIR)
print(f"\n✅ All results saved to: {OUTPUT_DIR}")

# %%
# Download results (Colab only)
from google.colab import files

for f in os.listdir(OUTPUT_DIR):
    filepath = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(filepath):
        files.download(filepath)
        print(f"📥 Downloaded: {f}")

# %% [markdown]
# ## 10. Summary

# %%
import json
print(json.dumps(result.summary(), indent=2, default=str))

# %% [markdown]
# ---
# *Hybrid Adaptive Fraud Detection System — Rule-Based × Unsupervised × Supervised × Fusion*
