"""Run the Dynamic Rule Engine on the full fraudTest.csv dataset with accuracy metrics."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

import pandas as pd
import numpy as np
from rule_engine.engine import RuleEngine

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "fraudTest.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "engine_output")

print(f"[LOAD] Loading {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Drop unnamed index column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print(f"[LOAD] Loaded {len(df)} rows × {len(df.columns)} columns")

engine = RuleEngine()
result = engine.run(df)
engine.save_results(OUTPUT_DIR)

# ── Accuracy Metrics ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  ACCURACY & PERFORMANCE METRICS")
print("=" * 60)

label_col = result.profile.detected_label_col
if label_col and label_col in result.result_df.columns:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score,
        matthews_corrcoef
    )

    y_true = result.result_df[label_col].astype(int)
    y_pred = result.result_df["_is_anomaly"].astype(int)
    y_scores = result.result_df["_anomaly_score"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = None

    print(f"\n  Ground Truth Column: '{label_col}'")
    print(f"  Total Samples:      {len(y_true):,}")
    print(f"  Actual Frauds:      {y_true.sum():,} ({y_true.mean()*100:.2f}%)")
    print(f"  Predicted Anomalies:{y_pred.sum():,} ({y_pred.mean()*100:.2f}%)")
    print()
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │  Accuracy:          {acc*100:>7.2f}%        │")
    print(f"  │  Precision:         {prec*100:>7.2f}%        │")
    print(f"  │  Recall (TPR):      {rec*100:>7.2f}%        │")
    print(f"  │  F1 Score:          {f1:>7.4f}         │")
    print(f"  │  MCC:               {mcc:>7.4f}         │")
    if auc is not None:
        print(f"  │  ROC-AUC:           {auc:>7.4f}         │")
    print(f"  └─────────────────────────────────────┘")

    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Normal", "Fraud"],
                                zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix:")
    print(f"                    Predicted Normal  Predicted Fraud")
    print(f"  Actual Normal     {tn:>14,}   {fp:>14,}")
    print(f"  Actual Fraud      {fn:>14,}   {tp:>14,}")
    print()
    print(f"  True Positives:  {tp:,}   (Correctly caught frauds)")
    print(f"  False Positives: {fp:,}   (False alarms)")
    print(f"  True Negatives:  {tn:,}   (Correctly cleared)")
    print(f"  False Negatives: {fn:,}   (Missed frauds)")

else:
    print("\n  ⚠ No ground truth label column detected.")
    print("    Cannot compute accuracy metrics.")
    print(f"    Detected label col: {label_col}")

print("\n" + "=" * 60)

print(f"\nTop 15 anomalous rows:")
top = result.result_df.nlargest(15, "_anomaly_score")
for _, row in top.iterrows():
    expls = row["_explanations"]
    reasons = [e["reason"] for e in expls] if expls else ["—"]
    print(f"  Score={row['_anomaly_score']:.3f} | {'; '.join(reasons[:2])}")

print(f"\nDone! Results saved to: {OUTPUT_DIR}")
