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
# # Hybrid Fraud Detector — Master Orchestrator
#
# Single entry point that chains all layers:
# 1. Data Understanding
# 2. Feature Engineering (rule_engine.FeatureGenerator)
# 3. Rule Engine (rule_engine.RuleGenerator + RuleEvaluator)
# 4. Unsupervised Module (IForest + HDBSCAN + LOF)
# 5. Supervised Module (XGBoost + LightGBM + CatBoost w/ Optuna)
# 6. Fusion Layer (weighted ensemble or stacking)
# 7. Explainability (SHAP + rule triggers)
#
# **Colab-ready**: auto-detects GPU, path-agnostic, no hardcoded columns.

# %%
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import warnings

warnings.filterwarnings("ignore")

# %%
# Ensure models/ is on path for both local and Colab execution
_this_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_models_dir = os.path.join(_this_dir, "..")
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)

# Import all layers
from hybrid_engine.data_understanding import DataUnderstanding, DataUnderstandingReport
from hybrid_engine.unsupervised_module import UnsupervisedModule, UnsupervisedResult
from hybrid_engine.supervised_module import SupervisedModule, SupervisedResult
from hybrid_engine.fusion_layer import FusionLayer, FusionResult
from hybrid_engine.feedback_loop import FeedbackLoop, DriftReport
from hybrid_engine.explainability import ExplainabilityLayer, RowExplanation, GlobalExplanation

# Import from existing rule engine
from rule_engine.data_profiler import DataProfiler, DataProfile
from rule_engine.feature_generator import FeatureGenerator
from rule_engine.rule_generator import RuleGenerator, Rule
from rule_engine.rule_evaluator import RuleEvaluator, EvaluationResult


# %% [markdown]
# ## HybridResult Dataclass

# %%
@dataclass
class HybridResult:
    """Complete output from the hybrid fraud detection system."""
    # Main output
    scored_df: pd.DataFrame = None
    final_scores: np.ndarray = None
    final_predictions: np.ndarray = None

    # Sub-results
    understanding_report: DataUnderstandingReport = None
    profile: DataProfile = None
    rule_evaluation: EvaluationResult = None
    unsupervised_result: UnsupervisedResult = None
    supervised_result: SupervisedResult = None
    fusion_result: FusionResult = None

    # Explainability
    row_explanations: List[RowExplanation] = field(default_factory=list)
    global_explanation: GlobalExplanation = None

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    best_model_name: str = ""

    # Metadata
    total_time_seconds: float = 0.0
    gpu_used: bool = False

    def summary(self) -> dict:
        """JSON-safe summary of the hybrid detection run."""
        return {
            "total_rows": len(self.scored_df) if self.scored_df is not None else 0,
            "total_flagged": int(self.final_predictions.sum()) if self.final_predictions is not None else 0,
            "metrics": self.metrics,
            "fusion_weights": self.fusion_weights,
            "best_supervised_model": self.best_model_name,
            "gpu_used": self.gpu_used,
            "total_time_seconds": round(self.total_time_seconds, 1),
            "unsupervised_models": (self.unsupervised_result.models_used
                                    if self.unsupervised_result else []),
            "fusion_strategy": (self.fusion_result.strategy
                                if self.fusion_result else ""),
        }


# %% [markdown]
# ## HybridFraudDetector Class

# %%
class HybridFraudDetector:
    """
    Master orchestrator for the hybrid adaptive fraud detection system.

    Usage:
        detector = HybridFraudDetector()
        result = detector.detect(df)

        # or from file:
        result = detector.detect_from_file("data.csv")

        # Access:
        result.scored_df       # DataFrame with all scores
        result.metrics         # Performance metrics
        result.fusion_weights  # Learned weights
    """

    def __init__(self, n_optuna_trials: int = 20,
                 shap_sample_size: int = 5000,
                 random_state: int = 42,
                 model_dir: str = "results/hybrid_output/models"):
        self.n_optuna_trials = n_optuna_trials
        self.shap_sample_size = shap_sample_size
        self.random_state = random_state
        self.model_dir = model_dir

        # Initialize all layers
        self.data_understanding = DataUnderstanding()
        self.feature_generator = FeatureGenerator()
        self.rule_generator = RuleGenerator()
        self.rule_evaluator = RuleEvaluator()
        self.unsupervised = UnsupervisedModule(random_state=random_state)
        self.supervised = SupervisedModule(
            n_optuna_trials=n_optuna_trials,
            random_state=random_state,
        )
        self.fusion = FusionLayer(random_state=random_state)
        self.feedback = FeedbackLoop(model_dir=model_dir)
        self.explainability = ExplainabilityLayer(
            shap_sample_size=shap_sample_size,
        )

        self._last_result: Optional[HybridResult] = None

    def detect(self, df: pd.DataFrame, verbose: bool = True) -> HybridResult:
        """
        Run the full hybrid fraud detection pipeline.

        Args:
            df: Input DataFrame (any schema)
            verbose: Print progress

        Returns:
            HybridResult with all scores, predictions, and explanations
        """
        start_time = time.time()

        if verbose:
            print("╔" + "═" * 58 + "╗")
            print("║  HYBRID ADAPTIVE FRAUD DETECTION SYSTEM                  ║")
            print("║  Rule-Based × Unsupervised × Supervised × Fusion         ║")
            print("╚" + "═" * 58 + "╝")
            print(f"\nInput: {len(df):,} rows × {len(df.columns)} columns")

        result = HybridResult()

        # ══════════════════════════════════════════════════════
        # LAYER 1: DATA UNDERSTANDING
        # ══════════════════════════════════════════════════════
        understanding = self.data_understanding.analyze(df, verbose=verbose)
        result.understanding_report = understanding
        result.profile = understanding.profile
        profile = understanding.profile

        # ══════════════════════════════════════════════════════
        # LAYER 2 & 3: FEATURE ENGINEERING + RULE ENGINE
        # ══════════════════════════════════════════════════════
        if verbose:
            print("\n" + "=" * 60)
            print("  LAYER 2 & 3: FEATURE ENGINEERING + RULE ENGINE")
            print("=" * 60)

        # Feature generation
        if verbose:
            print("\n[2] Generating derived features...")
        enriched_df = self.feature_generator.generate(df, profile)

        # Rule generation
        if verbose:
            print("\n[3.1] Generating detection rules...")
        rules = self.rule_generator.generate(profile)

        # Rule evaluation
        if verbose:
            print("\n[3.2] Evaluating rules...")
        scored_df, eval_result = self.rule_evaluator.evaluate(
            enriched_df, rules, profile
        )
        result.rule_evaluation = eval_result

        rule_scores = scored_df["_anomaly_score"].values.copy()
        rule_counts = scored_df["_rules_triggered_count"].values.copy()
        rule_explanations = scored_df["_explanations"].tolist()

        if verbose:
            print(f"  → Rule anomalies: {eval_result.total_anomalies:,} / "
                  f"{eval_result.total_rows:,}")

        # ══════════════════════════════════════════════════════
        # LAYER 4: UNSUPERVISED ANOMALY DETECTION
        # ══════════════════════════════════════════════════════
        # Select numeric features (exclude label, exclude rule-generated scores)
        label_col = profile.detected_label_col
        numeric_feature_cols = [
            c for c in scored_df.columns
            if pd.api.types.is_numeric_dtype(scored_df[c])
            and c != label_col
            and c not in ("_anomaly_score", "_is_anomaly",
                          "_rules_triggered_count")
            and not c.startswith("_ml_")
        ]

        unsup_scores, unsup_result = self.unsupervised.detect(
            scored_df, numeric_feature_cols, verbose=verbose
        )
        result.unsupervised_result = unsup_result

        # Add unsupervised scores to the df for supervised module input
        scored_df["_unsupervised_score"] = unsup_scores

        # ══════════════════════════════════════════════════════
        # LAYER 5: SUPERVISED LEARNING
        # ══════════════════════════════════════════════════════
        # Build supervised feature set: all numeric + rule score + unsupervised score
        sup_feature_cols = [
            c for c in scored_df.columns
            if pd.api.types.is_numeric_dtype(scored_df[c])
            and c != label_col
            and c not in ("_is_anomaly", "_rules_triggered_count")
            and not c.startswith("_ml_")
        ]

        if label_col and label_col in scored_df.columns:
            sup_probs, sup_result = self.supervised.train_and_predict(
                scored_df, sup_feature_cols, label_col, verbose=verbose
            )
            result.supervised_result = sup_result
            result.best_model_name = sup_result.best_model_name
            result.gpu_used = sup_result.gpu_used
        else:
            if verbose:
                print("\n[5] No label column → skipping supervised module")
            sup_probs = np.zeros(len(scored_df))
            sup_result = SupervisedResult()
            result.supervised_result = sup_result

        scored_df["_supervised_prob"] = sup_probs

        # ══════════════════════════════════════════════════════
        # LAYER 6 & 7: FUSION
        # ══════════════════════════════════════════════════════
        y_true = None
        if label_col and label_col in scored_df.columns:
            y_true = scored_df[label_col].astype(int).values

        fusion_result = self.fusion.fuse(
            rule_scores=rule_scores,
            unsupervised_scores=unsup_scores,
            supervised_probs=sup_probs,
            y_true=y_true,
            rule_counts=rule_counts,
            verbose=verbose,
        )
        result.fusion_result = fusion_result
        result.final_scores = fusion_result.final_scores
        result.final_predictions = fusion_result.final_predictions
        result.fusion_weights = fusion_result.weights
        result.metrics = fusion_result.metrics

        # Add to DataFrame
        scored_df["_final_fraud_score"] = fusion_result.final_scores
        scored_df["_final_prediction"] = fusion_result.final_predictions

        # ══════════════════════════════════════════════════════
        # LAYER 9: EXPLAINABILITY
        # ══════════════════════════════════════════════════════
        anomaly_mask = fusion_result.final_predictions.astype(bool)

        # Build simpler weights dict for explainability
        explain_weights = {
            "rule": fusion_result.weights.get("rule", 0.33),
            "unsupervised": fusion_result.weights.get("unsupervised", 0.33),
            "supervised": fusion_result.weights.get("supervised", 0.33),
        }

        row_explanations, global_expl = self.explainability.explain(
            df=scored_df,
            final_scores=fusion_result.final_scores,
            rule_scores=rule_scores,
            unsupervised_scores=unsup_scores,
            supervised_probs=sup_probs,
            fusion_weights=explain_weights,
            supervised_model=sup_result.best_model,
            feature_names=sup_result.feature_names,
            rule_explanations=rule_explanations,
            rule_trigger_counts=eval_result.rule_trigger_counts,
            anomaly_mask=anomaly_mask,
            verbose=verbose,
        )
        result.row_explanations = row_explanations
        result.global_explanation = global_expl

        # Final DataFrame
        result.scored_df = scored_df

        # Total time
        result.total_time_seconds = time.time() - start_time

        # ══════════════════════════════════════════════════════
        # FINAL REPORT
        # ══════════════════════════════════════════════════════
        if verbose:
            self._print_final_report(result, y_true)

        self._last_result = result
        return result

    def detect_from_file(self, path: str, verbose: bool = True) -> HybridResult:
        """Load CSV/JSON and run hybrid detection."""
        if verbose:
            print(f"[LOAD] Loading file: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            df = pd.read_json(path)
        else:
            df = pd.read_csv(path, low_memory=False)

        # Drop unnamed index columns
        unnamed = [c for c in df.columns if "Unnamed" in c]
        if unnamed:
            df.drop(columns=unnamed, inplace=True)

        if verbose:
            print(f"[LOAD] Loaded {len(df):,} rows × {len(df.columns)} columns\n")
        return self.detect(df, verbose=verbose)

    def save_results(self, output_dir: str = "results/hybrid_output"):
        """Save all results to an output directory."""
        if self._last_result is None:
            raise RuntimeError("No detection run yet. Call detect() first.")

        os.makedirs(output_dir, exist_ok=True)
        result = self._last_result

        # 1. Full scored CSV
        csv_path = os.path.join(output_dir, "hybrid_results_full.csv")
        # Drop non-serializable columns
        save_df = result.scored_df.copy()
        for col in save_df.columns:
            if save_df[col].dtype == object:
                try:
                    save_df[col] = save_df[col].astype(str)
                except Exception:
                    save_df.drop(columns=[col], inplace=True)
        save_df.to_csv(csv_path, index=False)
        print(f"[SAVE] Full results → {csv_path}")

        # 2. Flagged anomalies only
        flagged = save_df[save_df["_final_prediction"] == 1].sort_values(
            "_final_fraud_score", ascending=False
        )
        flagged_path = os.path.join(output_dir, "hybrid_flagged_anomalies.csv")
        flagged.to_csv(flagged_path, index=False)
        print(f"[SAVE] Flagged ({len(flagged):,}) → {flagged_path}")

        # 3. Summary JSON
        summary_path = os.path.join(output_dir, "hybrid_summary.json")
        with open(summary_path, "w") as f:
            json.dump(result.summary(), f, indent=2, default=str)
        print(f"[SAVE] Summary → {summary_path}")

        # 4. Metrics JSON
        metrics_path = os.path.join(output_dir, "hybrid_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(result.metrics, f, indent=2, default=str)
        print(f"[SAVE] Metrics → {metrics_path}")

        # 5. Sample explanations
        explanations_path = os.path.join(output_dir, "hybrid_explanations_sample.json")
        # Save top 50 anomalous row explanations
        if result.row_explanations:
            top_indices = np.argsort(result.final_scores)[::-1][:50]
            sample_expls = [
                result.row_explanations[i].to_dict() for i in top_indices
            ]
            with open(explanations_path, "w") as f:
                json.dump(sample_expls, f, indent=2, default=str)
            print(f"[SAVE] Sample explanations (top 50) → {explanations_path}")

        # 6. Save model state for feedback loop
        self.feedback.save_model_state({
            "best_model_name": result.best_model_name,
            "best_model": (result.supervised_result.best_model
                           if result.supervised_result else None),
            "best_roc_auc": result.metrics.get("roc_auc", 0),
            "fusion_weights": result.fusion_weights,
            "feature_names": (result.supervised_result.feature_names
                              if result.supervised_result else []),
            "n_samples": len(result.scored_df),
        }, verbose=True)

    # ------------------------------------------------------------------
    # Final Report
    # ------------------------------------------------------------------
    def _print_final_report(self, result: HybridResult,
                             y_true: Optional[np.ndarray]):
        """Print comprehensive final report."""
        print("\n")
        print("╔" + "═" * 58 + "╗")
        print("║              HYBRID DETECTION — FINAL REPORT              ║")
        print("╠" + "═" * 58 + "╣")

        n = len(result.scored_df)
        n_flagged = int(result.final_predictions.sum())

        print(f"║  Total Records:        {n:>10,}                         ║")
        print(f"║  Flagged as Fraud:     {n_flagged:>10,} "
              f"({n_flagged/n*100:>6.2f}%)             ║")
        print(f"║  Best Model:           {result.best_model_name:>10s}"
              f"                         ║")
        print(f"║  GPU Used:             {'Yes':>10s}" if result.gpu_used
              else f"║  GPU Used:             {'No':>10s}"
              f"                         ║")
        print(f"║  Fusion Strategy:      "
              f"{result.fusion_result.strategy:>10s}"
              f"                         ║")
        print(f"║  Time:                 "
              f"{result.total_time_seconds:>8.1f}s"
              f"                           ║")

        if result.metrics:
            print("╠" + "═" * 58 + "╣")
            print("║  PERFORMANCE METRICS                                     ║")
            print("╠" + "─" * 58 + "╣")
            for key, val in result.metrics.items():
                if isinstance(val, float):
                    print(f"║  {key:<24s} {val:>10.4f}"
                          f"                        ║")
                elif isinstance(val, int):
                    print(f"║  {key:<24s} {val:>10,}"
                          f"                        ║")

        print("╠" + "═" * 58 + "╣")
        print("║  FUSION WEIGHTS                                          ║")
        print("╠" + "─" * 58 + "╣")
        for key, val in result.fusion_weights.items():
            print(f"║  {key:<24s} {val:>10.4f}"
                  f"                        ║")

        print("╚" + "═" * 58 + "╝")


# %%
if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n = 2000
    test_df = pd.DataFrame({
        "txn_id": range(n),
        "customer_id": np.random.randint(1, 30, n),
        "amount": np.concatenate([
            np.random.exponential(50, 1900),
            np.random.exponential(500, 100),  # Anomalous amounts
        ]),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "lat": np.random.uniform(30, 45, n),
        "lon": np.random.uniform(-120, -70, n),
        "category": np.random.choice(
            ["food", "gas", "electronics", "travel"], n
        ),
        "is_fraud": np.array([0] * 1900 + [1] * 100),
    })

    detector = HybridFraudDetector(n_optuna_trials=3)
    result = detector.detect(test_df)
    print(f"\nSummary: {json.dumps(result.summary(), indent=2, default=str)}")
