"""
Run the Hybrid Adaptive Fraud Detection System.

Usage:
    python run_hybrid.py                    # Auto-detect data/fraudTest.csv
    python run_hybrid.py path/to/data.csv   # Custom CSV
"""
import sys
import os
import time

# Ensure models/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from hybrid_engine import HybridFraudDetector

# ── Configuration ─────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "fraudTest.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "hybrid_output")
N_OPTUNA_TRIALS = 20
SHAP_SAMPLE_SIZE = 5000


def main():
    # Allow CLI path override
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = DATA_PATH

    if not os.path.isfile(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        print("Usage: python run_hybrid.py [path_to_csv]")
        sys.exit(1)

    print(f"[CONFIG] Data: {data_path}")
    print(f"[CONFIG] Output: {OUTPUT_DIR}")
    print(f"[CONFIG] Optuna trials: {N_OPTUNA_TRIALS}")
    print(f"[CONFIG] SHAP sample: {SHAP_SAMPLE_SIZE}")
    print()

    # Initialize detector
    detector = HybridFraudDetector(
        n_optuna_trials=N_OPTUNA_TRIALS,
        shap_sample_size=SHAP_SAMPLE_SIZE,
        model_dir=os.path.join(OUTPUT_DIR, "models"),
    )

    # Run detection
    result = detector.detect_from_file(data_path)

    # Save everything
    print("\n[SAVE] Saving results...")
    detector.save_results(OUTPUT_DIR)

    # Print top anomalous rows
    print(f"\n{'='*60}")
    print("  TOP 15 MOST SUSPICIOUS RECORDS")
    print(f"{'='*60}")
    top = result.scored_df.nlargest(15, "_final_fraud_score")
    for i, (_, row) in enumerate(top.iterrows(), 1):
        score = row["_final_fraud_score"]
        rule_s = row["_anomaly_score"]
        unsup_s = row.get("_unsupervised_score", 0)
        sup_s = row.get("_supervised_prob", 0)
        print(f"\n  #{i}  Final={score:.3f} | "
              f"Rule={rule_s:.3f} | Unsup={unsup_s:.3f} | Sup={sup_s:.3f}")

        # Show top explanation
        if i <= 5 and result.row_explanations:
            idx = top.index[i - 1]
            if idx < len(result.row_explanations):
                expl = result.row_explanations[idx]
                if expl.rule_triggers:
                    triggers = expl.rule_triggers[:2]
                    for t in triggers:
                        if isinstance(t, dict):
                            print(f"       ↳ {t.get('reason', 'N/A')}")
                if expl.top_shap_features:
                    feat = expl.top_shap_features[0]
                    print(f"       ↳ SHAP: {feat.get('feature', '?')} "
                          f"({feat.get('direction', '?')})")

    print(f"\n{'='*60}")
    print(f"  Done! Total time: {result.total_time_seconds:.1f}s")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
