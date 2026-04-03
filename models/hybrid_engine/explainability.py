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
# # Explainability Layer
#
# Provides interpretable explanations for every prediction:
# - **SHAP values** from the best supervised model
# - **Rule triggers** from the rule engine
# - **Decision breakdown** combining all subsystem contributions

# %%
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import warnings

warnings.filterwarnings("ignore")


# %% [markdown]
# ## Explanation Dataclasses

# %%
@dataclass
class RowExplanation:
    """Per-row fraud detection explanation."""
    final_score: float = 0.0
    confidence: str = "low"  # low, medium, high
    rule_contribution: dict = field(default_factory=dict)
    unsupervised_contribution: dict = field(default_factory=dict)
    supervised_contribution: dict = field(default_factory=dict)
    top_shap_features: List[dict] = field(default_factory=list)
    rule_triggers: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "final_score": round(self.final_score, 4),
            "confidence": self.confidence,
            "rule_contribution": self.rule_contribution,
            "unsupervised_contribution": self.unsupervised_contribution,
            "supervised_contribution": self.supervised_contribution,
            "top_shap_features": self.top_shap_features,
            "rule_triggers": self.rule_triggers,
        }


@dataclass
class GlobalExplanation:
    """Global-level explainability summary."""
    shap_feature_importance: Dict[str, float] = field(default_factory=dict)
    rule_trigger_summary: Dict[str, int] = field(default_factory=dict)
    model_contribution_weights: Dict[str, float] = field(default_factory=dict)


# %% [markdown]
# ## ExplainabilityLayer Class

# %%
class ExplainabilityLayer:
    """
    Layer 9: Explainability.

    Computes SHAP values for supervised model predictions and
    combines with rule engine explanations to produce comprehensive,
    human-readable fraud detection explanations.
    """

    def __init__(self, shap_sample_size: int = 5000):
        self.shap_sample_size = shap_sample_size
        self._shap_values = None
        self._global_shap = None

    def explain(self, df: pd.DataFrame,
                final_scores: np.ndarray,
                rule_scores: np.ndarray,
                unsupervised_scores: np.ndarray,
                supervised_probs: np.ndarray,
                fusion_weights: Dict[str, float],
                supervised_model: Any = None,
                feature_names: List[str] = None,
                rule_explanations: List = None,
                rule_trigger_counts: Dict[str, int] = None,
                anomaly_mask: np.ndarray = None,
                verbose: bool = True) -> tuple:
        """
        Generate explanations for all predictions.

        Args:
            df: Source DataFrame
            final_scores: Final fusion scores
            rule_scores, unsupervised_scores, supervised_probs: Subsystem scores
            fusion_weights: Learned weights for each subsystem
            supervised_model: Best supervised model (for SHAP)
            feature_names: Feature names used by supervised model
            rule_explanations: Per-row rule explanations list
            rule_trigger_counts: Rule ID → trigger count
            anomaly_mask: Boolean mask of flagged rows
            verbose: Print progress

        Returns:
            (list_of_RowExplanation, GlobalExplanation)
        """
        if verbose:
            print("\n" + "=" * 60)
            print("  LAYER 9: EXPLAINABILITY")
            print("=" * 60)

        n = len(df)

        # --- SHAP computation ---
        shap_values = None
        global_shap = {}

        if supervised_model is not None and feature_names:
            if verbose:
                print("\n[9.1] Computing SHAP values...")
            shap_values, global_shap = self._compute_shap(
                df, supervised_model, feature_names, anomaly_mask, verbose
            )

        # --- Build per-row explanations ---
        if verbose:
            print("\n[9.2] Building per-row explanations...")

        row_explanations = []
        for i in range(n):
            expl = RowExplanation(
                final_score=float(final_scores[i]),
                confidence=self._score_to_confidence(final_scores[i]),
            )

            # Rule contribution
            rule_w = fusion_weights.get("rule", 0.33)
            expl.rule_contribution = {
                "score": round(float(rule_scores[i]), 4),
                "weight": round(rule_w, 4),
                "weighted_contribution": round(float(rule_scores[i] * rule_w), 4),
            }

            # Unsupervised contribution
            unsup_w = fusion_weights.get("unsupervised", 0.33)
            expl.unsupervised_contribution = {
                "score": round(float(unsupervised_scores[i]), 4),
                "weight": round(unsup_w, 4),
                "weighted_contribution": round(
                    float(unsupervised_scores[i] * unsup_w), 4
                ),
            }

            # Supervised contribution
            sup_w = fusion_weights.get("supervised", 0.33)
            expl.supervised_contribution = {
                "probability": round(float(supervised_probs[i]), 4),
                "weight": round(sup_w, 4),
                "weighted_contribution": round(
                    float(supervised_probs[i] * sup_w), 4
                ),
            }

            # SHAP features for this row (if computed)
            if shap_values is not None and i < len(shap_values):
                row_shap = shap_values[i]
                if row_shap is not None:
                    top_idx = np.argsort(np.abs(row_shap))[::-1][:5]
                    expl.top_shap_features = [
                        {
                            "feature": feature_names[j],
                            "shap_value": round(float(row_shap[j]), 4),
                            "direction": "increases fraud" if row_shap[j] > 0
                                         else "decreases fraud",
                        }
                        for j in top_idx
                        if j < len(feature_names)
                    ]

            # Rule triggers for this row
            if rule_explanations and i < len(rule_explanations):
                triggers = rule_explanations[i]
                if isinstance(triggers, list):
                    expl.rule_triggers = triggers[:5]  # Top 5

            row_explanations.append(expl)

        # --- Global explanation ---
        global_expl = GlobalExplanation(
            shap_feature_importance=global_shap,
            rule_trigger_summary=rule_trigger_counts or {},
            model_contribution_weights=fusion_weights,
        )

        if verbose:
            n_flagged = int(anomaly_mask.sum()) if anomaly_mask is not None else 0
            print(f"\n[9.✓] Explainability complete.")
            print(f"  → {n:,} row explanations generated")
            print(f"  → {n_flagged:,} flagged rows with detailed SHAP")
            if global_shap:
                top3 = list(global_shap.items())[:3]
                print(f"  → Top SHAP features: "
                      f"{', '.join(f'{k}={v:.3f}' for k, v in top3)}")

        return row_explanations, global_expl

    # ------------------------------------------------------------------
    # SHAP Computation
    # ------------------------------------------------------------------
    def _compute_shap(self, df: pd.DataFrame, model: Any,
                      feature_names: List[str],
                      anomaly_mask: Optional[np.ndarray],
                      verbose: bool) -> tuple:
        """Compute SHAP values for supervised model."""
        try:
            import shap
        except ImportError:
            if verbose:
                print("  → SHAP not installed, skipping")
            return None, {}

        # Determine which rows to explain
        valid_features = [f for f in feature_names if f in df.columns]
        if not valid_features:
            if verbose:
                print("  → No valid feature columns found for SHAP")
            return None, {}

        X = df[valid_features].copy()
        # Impute for safety
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(
            imputer.fit_transform(X), columns=valid_features, index=X.index
        )

        # Global importance: use subsample
        sample_size = min(self.shap_sample_size, len(X_imp))
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(X_imp), sample_size, replace=False)
        X_sample = X_imp.iloc[sample_idx]

        if verbose:
            print(f"  → Computing SHAP on {sample_size:,} sample rows...")

        try:
            explainer = shap.TreeExplainer(model, feature_names=valid_features)
            shap_values_sample = explainer.shap_values(X_sample)

            # Handle binary classification output
            if isinstance(shap_values_sample, list):
                shap_values_sample = shap_values_sample[1]  # Positive class

            # Global importance: mean absolute SHAP
            mean_abs_shap = np.abs(shap_values_sample).mean(axis=0)
            global_shap = {
                name: round(float(val), 4)
                for name, val in zip(valid_features, mean_abs_shap)
            }
            global_shap = dict(sorted(global_shap.items(),
                                       key=lambda x: x[1], reverse=True))

            # Per-row SHAP for anomalous rows only
            n = len(df)
            all_shap = [None] * n

            # Map sample SHAP back
            for local_i, global_i in enumerate(sample_idx):
                all_shap[global_i] = shap_values_sample[local_i]

            # Also compute for flagged non-sampled rows
            if anomaly_mask is not None:
                flagged_not_sampled = np.where(
                    anomaly_mask & ~np.isin(np.arange(n), sample_idx)
                )[0]
                if len(flagged_not_sampled) > 0:
                    extra_limit = min(len(flagged_not_sampled), 1000)
                    X_extra = X_imp.iloc[flagged_not_sampled[:extra_limit]]
                    if verbose:
                        print(f"  → Computing SHAP for {extra_limit} "
                              f"additional flagged rows...")
                    shap_extra = explainer.shap_values(X_extra)
                    if isinstance(shap_extra, list):
                        shap_extra = shap_extra[1]
                    for local_i, global_i in enumerate(
                            flagged_not_sampled[:extra_limit]):
                        all_shap[global_i] = shap_extra[local_i]

            self._shap_values = all_shap
            self._global_shap = global_shap
            return all_shap, global_shap

        except Exception as e:
            if verbose:
                print(f"  → SHAP computation failed: {e}")
            return None, {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _score_to_confidence(score: float) -> str:
        """Convert final score to confidence label."""
        if score >= 0.8:
            return "very_high"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        return "very_low"


# %%
if __name__ == "__main__":
    print("Explainability layer loaded successfully.")
    print("Requires a trained model and feature data to run.")
