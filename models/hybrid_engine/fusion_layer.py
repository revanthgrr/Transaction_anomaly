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
# # Fusion Layer
#
# Combines outputs from all three subsystems:
# - Rule-based anomaly scores
# - Unsupervised anomaly scores
# - Supervised fraud probabilities
#
# Two strategies:
# 1. **Weighted Ensemble** — scipy.optimize to find optimal weights
# 2. **Meta-Classifier (Stacking)** — Logistic Regression on subsystem outputs

# %%
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import warnings

warnings.filterwarnings("ignore")

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from scipy.optimize import minimize


# %% [markdown]
# ## Result Dataclass

# %%
@dataclass
class FusionResult:
    """Results from the fusion layer."""
    strategy: str = ""  # "weighted_ensemble" or "meta_classifier"
    weights: Dict[str, float] = field(default_factory=dict)
    final_scores: np.ndarray = None
    final_predictions: np.ndarray = None
    threshold: float = 0.5
    meta_model: object = None  # If stacking was used
    metrics: Dict[str, float] = field(default_factory=dict)


# %% [markdown]
# ## FusionLayer Class

# %%
class FusionLayer:
    """
    Layer 6 & 7: Fusion + Adaptive Weight Optimization.

    Combines rule-based, unsupervised, and supervised outputs into
    a single fraud score. Automatically selects the best fusion
    strategy based on data availability.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fuse(self, rule_scores: np.ndarray,
             unsupervised_scores: np.ndarray,
             supervised_probs: np.ndarray,
             y_true: Optional[np.ndarray] = None,
             rule_counts: Optional[np.ndarray] = None,
             verbose: bool = True) -> FusionResult:
        """
        Combine all subsystem outputs.

        Args:
            rule_scores: Anomaly scores from rule engine (0-1)
            unsupervised_scores: Anomaly scores from unsupervised models (0-1)
            supervised_probs: Fraud probabilities from supervised models (0-1)
            y_true: Ground truth labels (if available)
            rule_counts: Number of rules triggered per row (optional)
            verbose: Print progress

        Returns:
            FusionResult with final scores and predictions
        """
        if verbose:
            print("\n" + "=" * 60)
            print("  LAYER 6: FUSION LAYER")
            print("=" * 60)

        # Ensure all arrays are proper numpy
        rule_scores = np.asarray(rule_scores, dtype=float)
        unsupervised_scores = np.asarray(unsupervised_scores, dtype=float)
        supervised_probs = np.asarray(supervised_probs, dtype=float)

        # Replace NaN with 0
        rule_scores = np.nan_to_num(rule_scores, 0)
        unsupervised_scores = np.nan_to_num(unsupervised_scores, 0)
        supervised_probs = np.nan_to_num(supervised_probs, 0)

        if y_true is not None and len(np.unique(y_true)) > 1:
            if verbose:
                print("\n[6.1] Labels available → using Meta-Classifier (Stacking)")
            result = self._meta_classifier(
                rule_scores, unsupervised_scores, supervised_probs,
                y_true, rule_counts, verbose
            )
        else:
            if verbose:
                print("\n[6.1] No labels → using Weighted Ensemble")
            result = self._weighted_ensemble(
                rule_scores, unsupervised_scores, supervised_probs,
                y_true, verbose
            )

        # Determine optimal threshold
        if y_true is not None and len(np.unique(y_true)) > 1:
            result.threshold = self._find_optimal_threshold(
                result.final_scores, y_true, verbose
            )
        else:
            result.threshold = 0.5

        result.final_predictions = (result.final_scores >= result.threshold).astype(int)

        # Compute metrics if labels available
        if y_true is not None and len(np.unique(y_true)) > 1:
            result.metrics = self._compute_metrics(
                result.final_scores, result.final_predictions, y_true
            )
            if verbose:
                print(f"\n[6.✓] Fusion complete.")
                print(f"  → Strategy: {result.strategy}")
                print(f"  → Threshold: {result.threshold:.4f}")
                print(f"  → ROC-AUC:  {result.metrics.get('roc_auc', 0):.4f}")
                print(f"  → PR-AUC:   {result.metrics.get('pr_auc', 0):.4f}")
                print(f"  → F1:       {result.metrics.get('f1', 0):.4f}")
                print(f"  → Recall:   {result.metrics.get('recall', 0):.4f}")
                print(f"  → Precision:{result.metrics.get('precision', 0):.4f}")
        elif verbose:
            print(f"\n[6.✓] Fusion complete (no labels for metrics).")
            print(f"  → Strategy: {result.strategy}")
            print(f"  → Weights: {result.weights}")

        return result

    # ------------------------------------------------------------------
    # Weighted Ensemble
    # ------------------------------------------------------------------
    def _weighted_ensemble(self, rule_scores, unsupervised_scores,
                           supervised_probs, y_true, verbose
                           ) -> FusionResult:
        """Optimize weights via scipy to maximize ROC-AUC (or heuristic)."""
        result = FusionResult(strategy="weighted_ensemble")

        if y_true is not None and len(np.unique(y_true)) > 1:
            # Optimize weights
            def objective(w):
                w = np.abs(w)
                w = w / w.sum()
                combined = (w[0] * rule_scores +
                            w[1] * unsupervised_scores +
                            w[2] * supervised_probs)
                try:
                    return -roc_auc_score(y_true, combined)
                except ValueError:
                    return 0.0

            # Multiple random starts
            best_result = None
            best_obj = 0.0
            rng = np.random.RandomState(self.random_state)

            for _ in range(20):
                w0 = rng.dirichlet([1, 1, 1])
                res = minimize(objective, w0, method="Nelder-Mead",
                               options={"maxiter": 500})
                if best_result is None or res.fun < best_obj:
                    best_obj = res.fun
                    best_result = res

            w = np.abs(best_result.x)
            w = w / w.sum()

            if verbose:
                print(f"  → Optimized weights: rule={w[0]:.3f}, "
                      f"unsupervised={w[1]:.3f}, supervised={w[2]:.3f}")
                print(f"  → Optimized ROC-AUC: {-best_obj:.4f}")
        else:
            # Heuristic weights: supervised is usually strongest
            w = np.array([0.2, 0.2, 0.6])
            if verbose:
                print(f"  → Heuristic weights: rule=0.2, "
                      f"unsupervised=0.2, supervised=0.6")

        result.weights = {
            "rule": round(float(w[0]), 4),
            "unsupervised": round(float(w[1]), 4),
            "supervised": round(float(w[2]), 4),
        }

        result.final_scores = (w[0] * rule_scores +
                                w[1] * unsupervised_scores +
                                w[2] * supervised_probs)

        return result

    # ------------------------------------------------------------------
    # Meta-Classifier (Stacking)
    # ------------------------------------------------------------------
    def _meta_classifier(self, rule_scores, unsupervised_scores,
                         supervised_probs, y_true, rule_counts,
                         verbose) -> FusionResult:
        """Train a Logistic Regression meta-model on subsystem outputs."""
        result = FusionResult(strategy="meta_classifier")

        # Build meta-features
        meta_features = np.column_stack([
            rule_scores,
            unsupervised_scores,
            supervised_probs,
        ])

        if rule_counts is not None:
            meta_features = np.column_stack([
                meta_features,
                np.asarray(rule_counts, dtype=float),
            ])

        # Interaction features
        meta_features = np.column_stack([
            meta_features,
            rule_scores * supervised_probs,  # Rule × Supervised interaction
            unsupervised_scores * supervised_probs,  # Unsupervised × Supervised
        ])

        if verbose:
            print(f"  → Meta-features: {meta_features.shape[1]} dimensions")

        # Out-of-fold predictions for fair evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=self.random_state)

        meta_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
            C=1.0,
        )

        # Cross-validated predictions
        oof_probs = cross_val_predict(
            meta_model, meta_features, y_true, cv=cv, method="predict_proba"
        )[:, 1]

        # Train final model on all data
        meta_model.fit(meta_features, y_true)
        result.meta_model = meta_model
        result.final_scores = oof_probs

        # Extract weights from logistic regression coefficients
        coefs = meta_model.coef_[0]
        feature_names = ["rule", "unsupervised", "supervised"]
        if rule_counts is not None:
            feature_names.append("rule_count")
        feature_names.extend(["rule×supervised", "unsup×supervised"])

        result.weights = {
            name: round(float(c), 4)
            for name, c in zip(feature_names, coefs)
        }

        if verbose:
            print(f"  → Meta-classifier coefficients: {result.weights}")

        return result

    # ------------------------------------------------------------------
    # Threshold Selection
    # ------------------------------------------------------------------
    def _find_optimal_threshold(self, scores: np.ndarray,
                                 y_true: np.ndarray,
                                 verbose: bool) -> float:
        """Find threshold that maximizes F1 score."""
        precision, recall, thresholds = precision_recall_curve(y_true, scores)

        # F1 at each threshold
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0,
        )

        if len(thresholds) == 0:
            return 0.5

        best_idx = np.argmax(f1_scores[:-1])  # Exclude last element
        best_threshold = float(thresholds[best_idx])

        # Clamp to reasonable range
        best_threshold = max(0.1, min(0.9, best_threshold))

        if verbose:
            print(f"  → Optimal threshold: {best_threshold:.4f} "
                  f"(F1={f1_scores[best_idx]:.4f})")

        return best_threshold

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _compute_metrics(self, scores: np.ndarray,
                          predictions: np.ndarray,
                          y_true: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive fraud detection metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            matthews_corrcoef, confusion_matrix,
        )

        metrics = {}

        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 4)
        except ValueError:
            metrics["roc_auc"] = 0.0

        try:
            precision, recall, _ = precision_recall_curve(y_true, scores)
            metrics["pr_auc"] = round(float(auc(recall, precision)), 4)
        except ValueError:
            metrics["pr_auc"] = 0.0

        metrics["accuracy"] = round(float(accuracy_score(y_true, predictions)), 4)
        metrics["precision"] = round(float(precision_score(y_true, predictions, zero_division=0)), 4)
        metrics["recall"] = round(float(recall_score(y_true, predictions, zero_division=0)), 4)
        metrics["f1"] = round(float(f1_score(y_true, predictions, zero_division=0)), 4)
        metrics["mcc"] = round(float(matthews_corrcoef(y_true, predictions)), 4)

        cm = confusion_matrix(y_true, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["false_positives"] = int(fp)
            metrics["true_negatives"] = int(tn)
            metrics["false_negatives"] = int(fn)

        return metrics


# %%
if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    y = np.array([0]*950 + [1]*50)

    rule = np.random.uniform(0, 0.3, n)
    rule[y == 1] += np.random.uniform(0.2, 0.5, 50)

    unsup = np.random.uniform(0, 0.3, n)
    unsup[y == 1] += np.random.uniform(0.1, 0.4, 50)

    sup = np.random.uniform(0, 0.2, n)
    sup[y == 1] += np.random.uniform(0.3, 0.7, 50)

    fusion = FusionLayer()
    result = fusion.fuse(rule, unsup, sup, y)
    print(f"\nStrategy: {result.strategy}")
    print(f"Metrics: {result.metrics}")
