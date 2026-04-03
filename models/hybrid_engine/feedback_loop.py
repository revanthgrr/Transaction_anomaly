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
# # Feedback Loop (Self-Improving System)
#
# Enables the hybrid system to:
# - Retrain with new labeled data
# - Recalibrate rule thresholds
# - Detect concept drift via KS tests
# - Persist and reload model state

# %%
import numpy as np
import pandas as pd
import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


# %% [markdown]
# ## Drift Detection

# %%
@dataclass
class DriftReport:
    """Concept drift detection report."""
    drifted: bool = False
    drift_score: float = 0.0
    drifted_features: List[str] = field(default_factory=list)
    feature_ks_scores: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "drifted": self.drifted,
            "drift_score": round(self.drift_score, 4),
            "drifted_features": self.drifted_features,
            "feature_ks_scores": {k: round(v, 4)
                                   for k, v in self.feature_ks_scores.items()},
            "recommendation": self.recommendation,
        }


# %% [markdown]
# ## FeedbackLoop Class

# %%
class FeedbackLoop:
    """
    Layer 8: Self-improving feedback loop.

    - Retrains supervised model with new labeled data
    - Recalibrates rule thresholds by re-profiling
    - Detects concept drift via Kolmogorov-Smirnov test
    - Persists model state via joblib
    """

    def __init__(self, model_dir: str = "results/hybrid_output/models"):
        self.model_dir = model_dir
        self._training_history = []

    def detect_drift(self, original_df: pd.DataFrame,
                     new_df: pd.DataFrame,
                     feature_cols: List[str],
                     threshold: float = 0.05,
                     verbose: bool = True) -> DriftReport:
        """
        Detect concept drift between original and new data distributions.

        Uses Kolmogorov-Smirnov test on numeric features.
        A feature is considered drifted if KS p-value < threshold.

        Args:
            original_df: Training data
            new_df: New incoming data
            feature_cols: Numeric features to compare
            threshold: KS p-value threshold (default 0.05)
            verbose: Print details

        Returns:
            DriftReport
        """
        from scipy.stats import ks_2samp

        if verbose:
            print("\n[DRIFT] Running concept drift detection...")

        report = DriftReport()
        ks_scores = {}
        drifted = []

        for col in feature_cols:
            if col not in original_df.columns or col not in new_df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(original_df[col]):
                continue

            orig_vals = original_df[col].dropna().values
            new_vals = new_df[col].dropna().values

            if len(orig_vals) < 10 or len(new_vals) < 10:
                continue

            # Subsample for efficiency
            if len(orig_vals) > 10000:
                rng = np.random.RandomState(42)
                orig_vals = rng.choice(orig_vals, 10000, replace=False)
            if len(new_vals) > 10000:
                rng = np.random.RandomState(42)
                new_vals = rng.choice(new_vals, 10000, replace=False)

            stat, pvalue = ks_2samp(orig_vals, new_vals)
            ks_scores[col] = float(pvalue)

            if pvalue < threshold:
                drifted.append(col)

        report.feature_ks_scores = ks_scores
        report.drifted_features = drifted
        report.drifted = len(drifted) > 0
        report.drift_score = len(drifted) / max(len(ks_scores), 1)

        if report.drifted:
            if report.drift_score > 0.5:
                report.recommendation = (
                    "CRITICAL: Major drift detected in >50% of features. "
                    "Full retraining strongly recommended."
                )
            else:
                report.recommendation = (
                    "Moderate drift detected. Consider incremental retraining "
                    "or adjusting rule thresholds."
                )
        else:
            report.recommendation = "No significant drift detected. Model is current."

        if verbose:
            print(f"  → Drift detected: {report.drifted}")
            print(f"  → Drifted features: {len(drifted)} / {len(ks_scores)}")
            if drifted:
                print(f"  → Features: {drifted[:10]}")
            print(f"  → Recommendation: {report.recommendation}")

        return report

    def save_model_state(self, state: dict, verbose: bool = True):
        """
        Save model state to disk via joblib.

        Args:
            state: Dictionary containing model objects and metadata
            verbose: Print progress
        """
        import joblib

        os.makedirs(self.model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model objects
        model_path = os.path.join(self.model_dir, f"model_state_{timestamp}.joblib")
        joblib.dump(state, model_path)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "model_path": model_path,
            "training_samples": state.get("n_samples", 0),
            "best_model": state.get("best_model_name", ""),
            "best_roc_auc": state.get("best_roc_auc", 0),
            "fusion_weights": state.get("fusion_weights", {}),
        }
        meta_path = os.path.join(self.model_dir, f"metadata_{timestamp}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self._training_history.append(metadata)

        if verbose:
            print(f"\n[SAVE] Model state saved:")
            print(f"  → Model: {model_path}")
            print(f"  → Metadata: {meta_path}")

    def load_latest_model(self, verbose: bool = True) -> Optional[dict]:
        """
        Load the most recent model state from disk.

        Returns:
            Model state dictionary, or None if not found
        """
        import joblib

        if not os.path.isdir(self.model_dir):
            if verbose:
                print("[LOAD] No model directory found")
            return None

        # Find most recent .joblib file
        joblib_files = sorted(
            [f for f in os.listdir(self.model_dir) if f.endswith(".joblib")],
            reverse=True,
        )

        if not joblib_files:
            if verbose:
                print("[LOAD] No saved model states found")
            return None

        model_path = os.path.join(self.model_dir, joblib_files[0])
        state = joblib.load(model_path)

        if verbose:
            print(f"[LOAD] Loaded model state from: {model_path}")

        return state

    def get_training_history(self) -> List[dict]:
        """Return training history metadata."""
        # Also check disk for past runs
        if os.path.isdir(self.model_dir):
            meta_files = sorted(
                [f for f in os.listdir(self.model_dir)
                 if f.startswith("metadata_") and f.endswith(".json")],
                reverse=True,
            )
            history = []
            for mf in meta_files:
                with open(os.path.join(self.model_dir, mf), "r") as f:
                    history.append(json.load(f))
            return history

        return self._training_history


# %%
if __name__ == "__main__":
    np.random.seed(42)

    # Simulate original vs drifted data
    orig_df = pd.DataFrame({
        "amount": np.random.exponential(50, 1000),
        "frequency": np.random.poisson(5, 1000),
    })
    new_df = pd.DataFrame({
        "amount": np.random.exponential(80, 500),  # Drifted!
        "frequency": np.random.poisson(5, 500),     # Not drifted
    })

    fl = FeedbackLoop()
    report = fl.detect_drift(orig_df, new_df, ["amount", "frequency"])
    print(f"\nDrift report: {report.to_dict()}")
