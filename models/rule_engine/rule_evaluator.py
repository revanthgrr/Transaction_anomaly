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
# # Rule Evaluator
#
# Applies generated rules to an enriched DataFrame and produces
# explainable results for every row:
#
# - `anomaly_score`: 0.0–1.0 normalized score
# - `rules_triggered`: List of rule IDs that fired
# - `explanations`: Human-readable explanation per triggered rule
# - `is_anomaly`: Binary flag

# %%
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# %%
try:
    from .data_profiler import DataProfile
    from .rule_generator import Rule
except ImportError:
    from data_profiler import DataProfile
    from rule_generator import Rule

# %% [markdown]
# ## Result Dataclasses

# %%
@dataclass
class TriggeredRule:
    """Details of a single rule that was triggered for a row."""
    rule_id: str
    rule_type: str
    severity: str
    reason: str
    confidence: float
    feature_name: str
    feature_value: float
    threshold: float

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type,
            "severity": self.severity,
            "reason": self.reason,
            "confidence": round(float(self.confidence), 4),
            "feature_name": self.feature_name,
            "feature_value": round(float(self.feature_value), 4) if self.feature_value is not None else None,
            "threshold": round(float(self.threshold), 4),
        }


@dataclass
class EvaluationResult:
    """Full evaluation results across the entire dataset."""
    total_rows: int = 0
    total_anomalies: int = 0
    anomaly_rate: float = 0.0
    score_threshold: float = 0.0
    rules_evaluated: int = 0
    rules_skipped: int = 0
    rule_trigger_counts: dict = field(default_factory=dict)   # rule_id -> count
    per_row_results: list = field(default_factory=list)       # list of dicts

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "total_anomalies": self.total_anomalies,
            "anomaly_rate": round(self.anomaly_rate, 4),
            "score_threshold": round(self.score_threshold, 4),
            "rules_evaluated": self.rules_evaluated,
            "rules_skipped": self.rules_skipped,
            "rule_trigger_counts": self.rule_trigger_counts,
        }

# %% [markdown]
# ## RuleEvaluator Class

# %%
class RuleEvaluator:
    """
    Applies rules to an enriched DataFrame and produces scored,
    explainable results.

    Each rule is evaluated independently. The anomaly score is the
    weighted sum of triggered rules divided by the total possible weight.
    """

    def evaluate(self, df: pd.DataFrame, rules: list,
                 profile: DataProfile) -> tuple:
        """
        Evaluate all rules against the DataFrame.

        Returns:
            (df_with_scores, EvaluationResult)
        """
        result_df = df.copy()
        n = len(df)
        max_weight = sum(r.weight for r in rules if r.enabled)
        max_weight = max(max_weight, 1.0)

        # Refine thresholds dynamically where applicable
        rules = self._refine_thresholds(df, rules, profile)

        # Track per-row triggers
        all_scores = np.zeros(n)
        all_triggers = [[] for _ in range(n)]
        rule_counts = {}
        rules_evaluated = 0
        rules_skipped = 0

        for rule in rules:
            if not rule.enabled:
                rules_skipped += 1
                continue

            feat = rule.derived_feature
            if feat not in df.columns:
                rules_skipped += 1
                continue

            rules_evaluated += 1
            triggered_mask = self._evaluate_single_rule(df, rule, profile)

            if triggered_mask is None:
                rules_skipped += 1
                continue

            triggered_count = int(triggered_mask.sum())
            rule_counts[rule.rule_id] = triggered_count

            if triggered_count > 0:
                # Add weighted score
                all_scores[triggered_mask] += rule.weight

                # Build per-row explanations
                triggered_indices = np.where(triggered_mask)[0]
                for idx in triggered_indices:
                    feat_val = float(df.iloc[idx][feat]) if pd.notna(df.iloc[idx][feat]) else 0
                    # Confidence: how far past the threshold the value is
                    confidence = self._compute_confidence(feat_val, rule)
                    reason = self._build_reason(rule, feat_val, df.iloc[idx], profile)

                    all_triggers[idx].append(TriggeredRule(
                        rule_id=rule.rule_id,
                        rule_type=rule.rule_type,
                        severity=rule.severity,
                        reason=reason,
                        confidence=confidence,
                        feature_name=feat,
                        feature_value=feat_val,
                        threshold=rule.threshold,
                    ))

        # Normalise scores to [0, 1]
        result_df["_anomaly_score"] = np.clip(all_scores / max_weight, 0, 1)

        # Determine anomaly threshold dynamically
        score_threshold = self._compute_score_threshold(result_df["_anomaly_score"])

        result_df["_is_anomaly"] = (result_df["_anomaly_score"] >= score_threshold).astype(int)
        result_df["_rules_triggered_count"] = [len(t) for t in all_triggers]

        # Store serialised explanations
        result_df["_explanations"] = [
            [tr.to_dict() for tr in triggers] for triggers in all_triggers
        ]

        total_anomalies = int(result_df["_is_anomaly"].sum())
        anomaly_rate = total_anomalies / n if n > 0 else 0

        print(f"\n[EVAL] Evaluated {rules_evaluated} rules, skipped {rules_skipped}")
        print(f"[EVAL] Score threshold: {score_threshold:.4f}")
        print(f"[EVAL] Anomalies detected: {total_anomalies} / {n} ({anomaly_rate*100:.2f}%)")

        eval_result = EvaluationResult(
            total_rows=n,
            total_anomalies=total_anomalies,
            anomaly_rate=anomaly_rate,
            score_threshold=score_threshold,
            rules_evaluated=rules_evaluated,
            rules_skipped=rules_skipped,
            rule_trigger_counts=rule_counts,
        )

        return result_df, eval_result

    # ------------------------------------------------------------------
    # Single Rule Evaluation
    # ------------------------------------------------------------------
    def _evaluate_single_rule(self, df: pd.DataFrame, rule: Rule,
                              profile: DataProfile) -> np.ndarray:
        """Evaluate one rule and return a boolean mask of triggered rows."""
        feat = rule.derived_feature
        if feat not in df.columns:
            return None

        values = df[feat].fillna(0)
        op = rule.operator
        thresh = rule.threshold

        if op == "gt":
            return (values > thresh).values
        elif op == "lt":
            mask = (values < thresh) & (values >= 0)
            return mask.values
        elif op == "abs_gt":
            return (values.abs() > thresh).values
        elif op == "eq":
            return (values == thresh).values
        elif op == "custom_temporal":
            return self._eval_custom_temporal(df, rule, profile)
        else:
            return (values > thresh).values

    def _eval_custom_temporal(self, df: pd.DataFrame, rule: Rule,
                              profile: DataProfile) -> np.ndarray:
        """Custom evaluation for rare-hour temporal rules."""
        entity_col = profile.detected_entity_col
        if "_hour" not in df.columns or entity_col is None:
            return np.zeros(len(df), dtype=bool)

        # Calculate per-entity hour distribution
        hour_counts = df.groupby([entity_col, "_hour"]).size().reset_index(name="_hc")
        totals = df.groupby(entity_col).size().reset_index(name="_ht")
        hour_counts = hour_counts.merge(totals, on=entity_col)
        hour_counts["_hour_pct"] = hour_counts["_hc"] / hour_counts["_ht"]

        merged = df.merge(hour_counts[[entity_col, "_hour", "_hour_pct"]],
                          on=[entity_col, "_hour"], how="left")
        merged["_hour_pct"] = merged["_hour_pct"].fillna(0)

        return (merged["_hour_pct"] < rule.threshold).values

    # ------------------------------------------------------------------
    # Threshold Refinement
    # ------------------------------------------------------------------
    def _refine_thresholds(self, df: pd.DataFrame, rules: list,
                           profile: DataProfile) -> list:
        """
        Refine rule thresholds using actual data distributions.
        For example, velocity threshold is set to P95 of the frequency column.
        """
        for rule in rules:
            if not rule.enabled:
                continue

            feat = rule.derived_feature
            if feat not in df.columns:
                continue

            if rule.threshold_method == "p95_entity_frequency":
                # Set velocity threshold to P95 of frequency values
                p95 = df[feat].quantile(0.95)
                rule.threshold = max(float(p95), 3.0)

            elif rule.threshold_method == "p95_entity_distance":
                # Set location deviation to P95 of distance values
                non_zero = df.loc[df[feat] > 0, feat]
                if len(non_zero) > 10:
                    rule.threshold = float(non_zero.quantile(0.95))
                else:
                    rule.threshold = 500.0

        return rules

    # ------------------------------------------------------------------
    # Confidence Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_confidence(value: float, rule: Rule) -> float:
        """
        Compute a 0-1 confidence score based on how far the value
        exceeds the threshold.
        """
        if rule.threshold == 0:
            return 0.5

        if rule.operator in ("gt", "abs_gt"):
            ratio = abs(value) / abs(rule.threshold)
            # Map ratio [1, 5+] -> confidence [0.5, 1.0]
            return min(0.5 + (ratio - 1) * 0.125, 1.0)
        elif rule.operator == "lt":
            if value < 0:
                return 0.3
            ratio = rule.threshold / max(value, 0.001)
            return min(0.5 + (ratio - 1) * 0.1, 1.0)
        elif rule.operator == "eq":
            return 0.6  # Binary rules get moderate confidence
        else:
            return 0.5

    # ------------------------------------------------------------------
    # Reason Builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_reason(rule: Rule, feat_val: float,
                      row: pd.Series, profile: DataProfile) -> str:
        """Build a human-readable explanation for why this rule fired."""
        target = rule.target_column

        if rule.rule_type == "statistical_outlier":
            direction = "above" if feat_val > 0 else "below"
            return (f"{target} is {abs(feat_val):.1f}σ {direction} the entity mean "
                    f"(threshold: ±{rule.threshold:.1f}σ)")

        elif rule.rule_type == "percentile_extreme":
            return (f"{target} is in the top {(1 - rule.threshold)*100:.0f}% "
                    f"for this entity (rank: {feat_val:.2%})")

        elif rule.rule_type == "ratio_anomaly":
            return f"{target} is {feat_val:.1f}× the entity median (threshold: {rule.threshold:.0f}×)"

        elif rule.rule_type == "temporal_anomaly":
            if "night" in rule.rule_id:
                return "Transaction occurred during unusual night hours (00:00–05:00)"
            return f"Transaction at hour with < {rule.threshold*100:.0f}% of entity's historical activity"

        elif rule.rule_type == "velocity_spike":
            return (f"{feat_val:.0f} transactions in 1-hour window "
                    f"(threshold: {rule.threshold:.0f})")

        elif rule.rule_type == "velocity_rapid":
            return f"Only {feat_val:.0f}s since previous transaction (threshold: {rule.threshold:.0f}s)"

        elif rule.rule_type == "location_deviation":
            return (f"Location is {feat_val:.0f} km from entity's median center "
                    f"(threshold: {rule.threshold:.0f} km)")

        elif rule.rule_type == "geo_impossibility":
            return (f"Implied travel speed: {feat_val:.0f} km/h "
                    f"(exceeds {rule.threshold:.0f} km/h physical limit)")

        elif rule.rule_type == "category_anomaly":
            return (f"Category used only {feat_val*100:.1f}% of the time by this entity "
                    f"(threshold: {rule.threshold*100:.0f}%)")

        elif rule.rule_type == "iqr_outlier":
            return f"{target} falls outside the 1.5×IQR fence"

        return f"Rule {rule.rule_id} triggered: {feat_val:.4f} vs threshold {rule.threshold:.4f}"

    # ------------------------------------------------------------------
    # Dynamic Score Threshold
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_score_threshold(scores: pd.Series) -> float:
        """
        Determine the anomaly threshold dynamically.

        Strategy: use the 95th percentile of scores, but clamp to a
        minimum of 0.15 to avoid flagging too many rows in clean datasets.
        """
        if scores.max() == 0:
            return 0.5  # No rules triggered, set high threshold

        p95 = float(scores.quantile(0.95))
        # Ensure we don't flag more than ~10%
        return max(p95, 0.15)


# %% [markdown]
# ## Quick Test

# %%
if __name__ == "__main__":
    from data_profiler import DataProfiler
    from feature_generator import FeatureGenerator
    from rule_generator import RuleGenerator

    np.random.seed(42)
    test_df = pd.DataFrame({
        "txn_id": range(200),
        "customer_id": np.random.randint(1, 10, 200),
        "amount": np.random.exponential(50, 200),
        "event_time": pd.date_range("2024-01-01", periods=200, freq="30min"),
        "lat": np.random.uniform(30, 45, 200),
        "lon": np.random.uniform(-120, -70, 200),
        "category": np.random.choice(["food", "gas", "electronics", "travel"], 200),
        "is_fraud": np.random.choice([0, 1], 200, p=[0.95, 0.05]),
    })

    profiler = DataProfiler()
    profile = profiler.profile(test_df)
    enriched = FeatureGenerator().generate(test_df, profile)
    rules = RuleGenerator().generate(profile)
    evaluator = RuleEvaluator()
    result_df, eval_result = evaluator.evaluate(enriched, rules, profile)

    print(f"\n=== Evaluation Summary ===")
    print(f"Total anomalies: {eval_result.total_anomalies}")
    print(f"Anomaly rate: {eval_result.anomaly_rate*100:.2f}%")
    print(f"Rule triggers: {eval_result.rule_trigger_counts}")
    print(f"\nTop 5 anomalous rows:")
    top = result_df.nlargest(5, "_anomaly_score")
    for _, row in top.iterrows():
        print(f"  Score: {row['_anomaly_score']:.3f} | Triggers: {row['_rules_triggered_count']}")
        for expl in row["_explanations"]:
            print(f"    → {expl['reason']}")
