# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Rule Engine — Orchestrator
#
# Top-level class that chains the four pipeline stages:
#
# 1. **Profile** → Infer schema & statistics
# 2. **Generate Features** → Create derived columns
# 3. **Generate Rules** → Build detection rules from statistics
# 4. **Evaluate** → Apply rules and score every row
#
# Supports file-based input, rule export/import, and optional ML integration.

# %%
import pandas as pd
import json
import os
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# %%
from .data_profiler import DataProfiler, DataProfile
from .feature_generator import FeatureGenerator
from .rule_generator import RuleGenerator, Rule
from .rule_evaluator import RuleEvaluator, EvaluationResult

# %% [markdown]
# ## EngineResult Dataclass

# %%
@dataclass
class EngineResult:
    """Container for all outputs of a single engine run."""
    profile: DataProfile
    rules: list                       # list[Rule]
    evaluation: EvaluationResult
    result_df: pd.DataFrame
    enriched_df: pd.DataFrame

    def summary(self) -> dict:
        """Return a JSON-safe summary of the run."""
        return {
            "profile": self.profile.to_dict(),
            "rules": [r.to_dict() for r in self.rules],
            "evaluation": self.evaluation.to_dict(),
            "columns_original": [c for c in self.result_df.columns if not c.startswith("_")],
            "columns_generated": [c for c in self.result_df.columns if c.startswith("_")],
        }

# %% [markdown]
# ## RuleEngine Class

# %%
class RuleEngine:
    """
    Schema-agnostic, dynamic rule-based anomaly detection engine.

    Usage:
        engine = RuleEngine()
        result = engine.run(df)
        # or
        result = engine.run_from_file("path/to/data.csv")

    The engine uses zero hardcoded column names or thresholds.
    """

    def __init__(self):
        self.profiler = DataProfiler()
        self.feature_gen = FeatureGenerator()
        self.rule_gen = RuleGenerator()
        self.evaluator = RuleEvaluator()
        self._last_result: Optional[EngineResult] = None

    def run(self, df: pd.DataFrame,
            custom_rules: Optional[list] = None) -> EngineResult:
        """
        Run the full pipeline on a DataFrame.

        Args:
            df: Input DataFrame (any schema)
            custom_rules: Optional list of Rule objects to use instead of
                          auto-generated rules. Can also be combined with
                          generated rules by calling add_rule() after.

        Returns:
            EngineResult with profile, rules, evaluation, and scored DataFrame
        """
        print("=" * 60)
        print("  DYNAMIC RULE ENGINE — Starting Analysis")
        print("=" * 60)

        # Stage 1: Profile
        print("\n[1/4] Profiling dataset...")
        profile = self.profiler.profile(df)
        print(f"  → {profile.row_count} rows, {len(profile.columns)} columns")
        print(f"  → Entity: {profile.detected_entity_col}")
        print(f"  → Amount: {profile.detected_amount_col}")
        print(f"  → Time:   {profile.detected_time_col}")
        print(f"  → Geo:    {profile.detected_geo_cols}")
        print(f"  → Labels: {profile.detected_label_col}")
        print(f"  → Categories: {profile.detected_category_cols}")

        # Stage 2: Feature Generation
        print("\n[2/4] Generating derived features...")
        enriched_df = self.feature_gen.generate(df, profile)

        # Stage 3: Rule Generation
        print("\n[3/4] Generating detection rules...")
        if custom_rules is not None:
            rules = custom_rules
            print(f"  → Using {len(rules)} custom rules")
        else:
            rules = self.rule_gen.generate(profile)

        # Stage 4: Evaluation
        print("\n[4/4] Evaluating rules...")
        result_df, eval_result = self.evaluator.evaluate(enriched_df, rules, profile)

        # Ground truth comparison if available
        if profile.detected_label_col:
            self._compare_ground_truth(result_df, profile.detected_label_col)

        engine_result = EngineResult(
            profile=profile,
            rules=rules,
            evaluation=eval_result,
            result_df=result_df,
            enriched_df=enriched_df,
        )
        self._last_result = engine_result

        print("\n" + "=" * 60)
        print("  ANALYSIS COMPLETE")
        print(f"  Anomalies: {eval_result.total_anomalies} / {eval_result.total_rows} "
              f"({eval_result.anomaly_rate*100:.2f}%)")
        print("=" * 60)

        return engine_result

    def run_quiet(self, df: pd.DataFrame,
                  custom_rules: Optional[list] = None) -> EngineResult:
        """Run the full pipeline silently (no print output). Used by hybrid system."""
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            return self.run(df, custom_rules)

    def run_from_file(self, path: str,
                      custom_rules: Optional[list] = None) -> EngineResult:
        """Load CSV or JSON file and run the engine."""
        print(f"[LOAD] Loading file: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            df = pd.read_json(path)
        elif ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        else:
            df = pd.read_csv(path)  # Try CSV as fallback
        print(f"[LOAD] Loaded {len(df)} rows × {len(df.columns)} columns")
        return self.run(df, custom_rules)

    # ------------------------------------------------------------------
    # Rule Management
    # ------------------------------------------------------------------
    def export_rules(self, path: str, fmt: str = "json"):
        """Export the last run's generated rules to file."""
        if self._last_result is None:
            raise RuntimeError("No engine run yet. Call run() first.")
        if fmt == "yaml":
            RuleGenerator.rules_to_yaml(self._last_result.rules, path)
        else:
            RuleGenerator.rules_to_json(self._last_result.rules, path)

    def import_rules(self, path: str) -> list:
        """Import rules from a JSON or YAML file."""
        if path.endswith(".yaml") or path.endswith(".yml"):
            return RuleGenerator.rules_from_yaml(path)
        return RuleGenerator.rules_from_json(path)

    def add_rule(self, rule_dict: dict):
        """Add a rule to the last run's rule set."""
        if self._last_result is None:
            raise RuntimeError("No engine run yet.")
        self._last_result.rules.append(Rule.from_dict(rule_dict))

    def remove_rule(self, rule_id: str):
        """Remove a rule by ID from the last run's rule set."""
        if self._last_result is None:
            raise RuntimeError("No engine run yet.")
        self._last_result.rules = [
            r for r in self._last_result.rules if r.rule_id != rule_id
        ]

    def rerun_evaluation(self) -> EngineResult:
        """Re-evaluate with the current (possibly modified) rule set."""
        if self._last_result is None:
            raise RuntimeError("No engine run yet.")
        result_df, eval_result = self.evaluator.evaluate(
            self._last_result.enriched_df,
            self._last_result.rules,
            self._last_result.profile,
        )
        self._last_result.result_df = result_df
        self._last_result.evaluation = eval_result
        return self._last_result

    # ------------------------------------------------------------------
    # Ground Truth Comparison
    # ------------------------------------------------------------------
    def _compare_ground_truth(self, df: pd.DataFrame, label_col: str):
        """Compare engine predictions with ground truth labels."""
        if "_is_anomaly" not in df.columns or label_col not in df.columns:
            return

        try:
            from sklearn.metrics import classification_report, confusion_matrix

            y_true = df[label_col].astype(int)
            y_pred = df["_is_anomaly"].astype(int)

            print(f"\n[GROUND TRUTH] Comparing with '{label_col}':")
            print(classification_report(y_true, y_pred,
                                        target_names=["Normal", "Anomaly"],
                                        zero_division=0))

            cm = confusion_matrix(y_true, y_pred)
            print(f"Confusion Matrix:")
            print(f"  [{cm[0][0]:>6} {cm[0][1]:>6}]  (Normal)")
            print(f"  [{cm[1][0]:>6} {cm[1][1]:>6}]  (Anomaly)")
        except Exception as e:
            print(f"[GROUND TRUTH] Could not compare: {e}")

    # ------------------------------------------------------------------
    # Convenience: Save results
    # ------------------------------------------------------------------
    def save_results(self, output_dir: str):
        """Save results, rules, and profile to an output directory."""
        if self._last_result is None:
            raise RuntimeError("No engine run yet.")

        os.makedirs(output_dir, exist_ok=True)

        # Save scored DataFrame
        result_path = os.path.join(output_dir, "engine_results_full.csv")
        self._last_result.result_df.to_csv(result_path, index=False)
        print(f"[SAVE] Full results → {result_path}")

        # Save flagged only
        flagged = self._last_result.result_df[
            self._last_result.result_df["_is_anomaly"] == 1
        ].sort_values("_anomaly_score", ascending=False)
        flagged_path = os.path.join(output_dir, "engine_flagged_anomalies.csv")
        flagged.to_csv(flagged_path, index=False)
        print(f"[SAVE] Flagged anomalies ({len(flagged)}) → {flagged_path}")

        # Save rules
        rules_path = os.path.join(output_dir, "engine_rules.json")
        RuleGenerator.rules_to_json(self._last_result.rules, rules_path)

        # Save summary
        summary_path = os.path.join(output_dir, "engine_summary.json")
        with open(summary_path, "w") as f:
            json.dump(self._last_result.summary(), f, indent=2, default=str)
        print(f"[SAVE] Summary → {summary_path}")


# %% [markdown]
# ## Quick Test

# %%
if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    test_df = pd.DataFrame({
        "txn_id": range(500),
        "user_account": np.random.randint(1, 20, 500),
        "purchase_amount": np.random.exponential(50, 500),
        "event_date": pd.date_range("2024-01-01", periods=500, freq="30min"),
        "store_lat": np.random.uniform(30, 45, 500),
        "store_long": np.random.uniform(-120, -70, 500),
        "product_type": np.random.choice(["electronics", "grocery", "clothing", "travel"], 500),
        "is_fraud": np.random.choice([0, 1], 500, p=[0.95, 0.05]),
    })

    engine = RuleEngine()
    result = engine.run(test_df)

    print(f"\n\nTop 10 anomalous rows:")
    top = result.result_df.nlargest(10, "_anomaly_score")
    for _, row in top.iterrows():
        expls = row["_explanations"]
        reasons = [e["reason"] for e in expls] if expls else ["—"]
        print(f"  Score={row['_anomaly_score']:.3f} | {'; '.join(reasons)}")
