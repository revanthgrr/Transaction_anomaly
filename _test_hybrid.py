"""Quick integration test for the hybrid system."""
import sys
import os
import io
import traceback

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

import numpy as np
import pandas as pd

try:
    from hybrid_engine.hybrid_detector import HybridFraudDetector

    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "txn_id": range(n),
        "customer_id": np.random.randint(1, 20, n),
        "amount": np.concatenate([
            np.random.exponential(50, 475),
            np.random.exponential(500, 25),
        ]),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="30min"),
        "lat": np.random.uniform(30, 45, n),
        "lon": np.random.uniform(-120, -70, n),
        "category": np.random.choice(["food", "gas", "electronics", "travel"], n),
        "is_fraud": np.array([0] * 475 + [1] * 25),
    })

    print("Starting hybrid detection test (n=500, optuna=3 trials)...")
    detector = HybridFraudDetector(n_optuna_trials=3, shap_sample_size=100)
    result = detector.detect(df, verbose=True)

    print("\n" + "=" * 50)
    print("  INTEGRATION TEST RESULTS")
    print("=" * 50)
    print(f"  Rows:        {len(result.scored_df)}")
    print(f"  Flagged:     {int(result.final_predictions.sum())}")
    print(f"  Best Model:  {result.best_model_name}")
    print(f"  ROC-AUC:     {result.metrics.get('roc_auc', 'N/A')}")
    print(f"  PR-AUC:      {result.metrics.get('pr_auc', 'N/A')}")
    print(f"  F1:          {result.metrics.get('f1', 'N/A')}")
    print(f"  Recall:      {result.metrics.get('recall', 'N/A')}")
    print(f"  Precision:   {result.metrics.get('precision', 'N/A')}")
    print(f"  Fusion:      {result.fusion_weights}")
    print(f"  Time:        {result.total_time_seconds:.1f}s")
    print(f"  Explanations:{len(result.row_explanations)}")
    print(f"\n  SUCCESS!")

except Exception as e:
    print(f"\n  FAILED: {e}")
    traceback.print_exc()
