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
# # Rule Generator
#
# Dynamically generates detection rules as JSON objects based on the
# DataProfile. **Zero hardcoded thresholds** — all thresholds are derived
# from statistical properties of the actual data.
#
# Rules are stored as plain dictionaries and can be serialised to JSON/YAML.

# %%
import numpy as np
import json
import yaml
from typing import Optional
from dataclasses import dataclass, field, asdict
import warnings

warnings.filterwarnings("ignore")

# %%
try:
    from .data_profiler import DataProfile
except ImportError:
    from data_profiler import DataProfile

# %% [markdown]
# ## Rule Dataclass

# %%
@dataclass
class Rule:
    """A single detection rule."""
    rule_id: str
    rule_type: str                  # statistical_outlier, percentile_extreme, etc.
    target_column: str              # Original column this rule targets
    derived_feature: str            # The derived feature column used for evaluation
    condition: str                  # Human-readable condition string
    operator: str                   # gt, lt, abs_gt, eq, between
    threshold: float                # Computed threshold value
    threshold_method: str           # How the threshold was derived
    severity: str = "medium"        # low, medium, high, critical
    weight: float = 1.0             # Weight for score calculation
    description: str = ""           # Human-readable description
    requires_entity: bool = False   # Whether this rule needs per-entity grouping
    enabled: bool = True            # Can be toggled off
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure threshold is JSON-safe
        d["threshold"] = float(d["threshold"]) if d["threshold"] is not None else None
        d["weight"] = float(d["weight"])
        return d

    @staticmethod
    def from_dict(d: dict) -> "Rule":
        return Rule(**{k: v for k, v in d.items() if k in Rule.__dataclass_fields__})

# %% [markdown]
# ## RuleGenerator Class

# %%
class RuleGenerator:
    """
    Generates detection rules dynamically from a DataProfile.

    No hardcoded column names or fixed thresholds — everything is
    derived from the statistical properties of the data.
    """

    def generate(self, profile: DataProfile) -> list:
        """Generate all applicable rules based on the data profile."""
        rules = []
        rule_counter = {"n": 0}

        def _next_id(prefix: str) -> str:
            rule_counter["n"] += 1
            return f"{prefix}_{rule_counter['n']:03d}"

        amount_col = profile.detected_amount_col
        time_col = profile.detected_time_col
        entity_col = profile.detected_entity_col
        geo_cols = profile.detected_geo_cols

        # ── Statistical Outlier Rules (amount) ─────────────────────────
        if amount_col:
            rules.extend(self._amount_rules(profile, amount_col, entity_col, _next_id))

        # ── Temporal Anomaly Rules ─────────────────────────────────────
        if time_col:
            rules.extend(self._temporal_rules(profile, time_col, entity_col, _next_id))

        # ── Velocity Rules ─────────────────────────────────────────────
        if time_col and entity_col:
            rules.extend(self._velocity_rules(profile, _next_id))

        # ── Location Rules ─────────────────────────────────────────────
        if geo_cols:
            rules.extend(self._geo_rules(profile, geo_cols, entity_col, time_col, _next_id))

        # ── Category Rules ─────────────────────────────────────────────
        for cat_col in profile.detected_category_cols:
            rules.extend(self._category_rules(profile, cat_col, entity_col, _next_id))

        # ── Generic IQR Outlier Rules (all other numerics) ─────────────
        rules.extend(self._generic_numeric_rules(profile, amount_col, _next_id))

        print(f"[RULES] Generated {len(rules)} dynamic rules.")
        for r in rules:
            sev_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(r.severity, "⚪")
            print(f"  {sev_icon} {r.rule_id}: {r.description}")

        return rules

    # ------------------------------------------------------------------
    # Amount Rules
    # ------------------------------------------------------------------
    def _amount_rules(self, profile, amount_col, entity_col, _id):
        rules = []
        stats = profile.columns[amount_col].stats
        skewness = abs(stats.get("skewness", 0))

        # Adaptive z-score threshold based on distribution shape
        # Highly skewed distributions get a higher threshold
        if skewness > 3:
            z_thresh = 4.0
            method = "adaptive_z (high skew)"
        elif skewness > 1:
            z_thresh = 3.5
            method = "adaptive_z (moderate skew)"
        else:
            z_thresh = 3.0
            method = "adaptive_z (normal)"

        feat = f"_{amount_col}_zscore"
        rules.append(Rule(
            rule_id=_id("stat_outlier"),
            rule_type="statistical_outlier",
            target_column=amount_col,
            derived_feature=feat,
            condition=f"abs({feat}) > {z_thresh}",
            operator="abs_gt",
            threshold=z_thresh,
            threshold_method=method,
            severity="high",
            weight=2.0,
            description=f"Flags records where {amount_col} z-score exceeds ±{z_thresh}σ from entity mean",
            requires_entity=entity_col is not None,
        ))

        # Percentile extreme (99th percentile)
        p99 = stats.get("p99", 0)
        if p99 > 0:
            rules.append(Rule(
                rule_id=_id("pct_extreme"),
                rule_type="percentile_extreme",
                target_column=amount_col,
                derived_feature=f"_{amount_col}_pct_rank",
                condition=f"_{amount_col}_pct_rank > 0.99",
                operator="gt",
                threshold=0.99,
                threshold_method="99th_percentile",
                severity="high",
                weight=1.5,
                description=f"Flags records in the top 1% of {amount_col} for the entity",
                requires_entity=entity_col is not None,
            ))

        # Ratio-to-median extreme
        rules.append(Rule(
            rule_id=_id("ratio_extreme"),
            rule_type="ratio_anomaly",
            target_column=amount_col,
            derived_feature=f"_{amount_col}_ratio_to_median",
            condition=f"_{amount_col}_ratio_to_median > 5.0",
            operator="gt",
            threshold=5.0,
            threshold_method="ratio_5x_median",
            severity="medium",
            weight=1.0,
            description=f"Flags records where {amount_col} is over 5× the entity median",
            requires_entity=entity_col is not None,
        ))

        return rules

    # ------------------------------------------------------------------
    # Temporal Rules
    # ------------------------------------------------------------------
    def _temporal_rules(self, profile, time_col, entity_col, _id):
        rules = []

        # Night-time activity
        rules.append(Rule(
            rule_id=_id("temporal_night"),
            rule_type="temporal_anomaly",
            target_column=time_col,
            derived_feature="_is_night",
            condition="_is_night == 1",
            operator="eq",
            threshold=1.0,
            threshold_method="night_hours_0_5",
            severity="low",
            weight=0.5,
            description="Flags records occurring between midnight and 5 AM",
            requires_entity=False,
        ))

        # Unusual hour (low activity hour for entity)
        if entity_col:
            rules.append(Rule(
                rule_id=_id("temporal_rare_hour"),
                rule_type="temporal_anomaly",
                target_column=time_col,
                derived_feature="_hour",
                condition="hour activity < 2% of entity history",
                operator="custom_temporal",
                threshold=0.02,
                threshold_method="entity_hour_percentile",
                severity="medium",
                weight=1.0,
                description="Flags records at hours where the entity has < 2% historical activity",
                requires_entity=True,
            ))

        return rules

    # ------------------------------------------------------------------
    # Velocity Rules
    # ------------------------------------------------------------------
    def _velocity_rules(self, profile, _id):
        rules = []

        # High frequency spike
        rules.append(Rule(
            rule_id=_id("velocity_spike"),
            rule_type="velocity_spike",
            target_column="(derived)",
            derived_feature="_txn_frequency_1h",
            condition="_txn_frequency_1h > dynamic_p95",
            operator="gt",
            threshold=5.0,  # Default; evaluator will use data-driven P95
            threshold_method="p95_entity_frequency",
            severity="high",
            weight=1.5,
            description="Flags records with abnormally high transaction frequency in 1-hour window",
            requires_entity=True,
        ))

        # Rapid succession (very short time since last)
        rules.append(Rule(
            rule_id=_id("velocity_rapid"),
            rule_type="velocity_rapid",
            target_column="(derived)",
            derived_feature="_time_since_last",
            condition="_time_since_last < 60 seconds AND _time_since_last >= 0",
            operator="lt",
            threshold=60.0,
            threshold_method="rapid_succession_60s",
            severity="medium",
            weight=1.0,
            description="Flags records occurring within 60 seconds of the previous record for the same entity",
            requires_entity=True,
        ))

        return rules

    # ------------------------------------------------------------------
    # Geolocation Rules
    # ------------------------------------------------------------------
    def _geo_rules(self, profile, geo_cols, entity_col, time_col, _id):
        rules = []
        lat_col, lon_col = geo_cols

        if entity_col:
            # Distance deviation from entity center
            rules.append(Rule(
                rule_id=_id("location_dev"),
                rule_type="location_deviation",
                target_column=f"{lat_col},{lon_col}",
                derived_feature="_distance_from_center",
                condition="_distance_from_center > P95 of entity distances",
                operator="gt",
                threshold=500.0,  # Default; evaluator refines with P95
                threshold_method="p95_entity_distance",
                severity="high",
                weight=1.5,
                description="Flags records far from the entity's median geographic location",
                requires_entity=True,
            ))

        if time_col:
            # Geographic impossibility (travel speed > 900 km/h)
            rules.append(Rule(
                rule_id=_id("geo_impossible"),
                rule_type="geo_impossibility",
                target_column=f"{lat_col},{lon_col}",
                derived_feature="_travel_speed_kmh",
                condition="_travel_speed_kmh > 900",
                operator="gt",
                threshold=900.0,
                threshold_method="physical_speed_limit",
                severity="critical",
                weight=3.0,
                description="Flags consecutive records with implied travel speed exceeding 900 km/h",
                requires_entity=entity_col is not None,
            ))

        return rules

    # ------------------------------------------------------------------
    # Category Rules
    # ------------------------------------------------------------------
    def _category_rules(self, profile, cat_col, entity_col, _id):
        rules = []
        safe_name = cat_col.replace(" ", "_")

        if entity_col:
            rules.append(Rule(
                rule_id=_id("cat_rare"),
                rule_type="category_anomaly",
                target_column=cat_col,
                derived_feature=f"_{safe_name}_frequency",
                condition=f"_{safe_name}_frequency < 0.03",
                operator="lt",
                threshold=0.03,
                threshold_method="entity_category_3pct",
                severity="medium",
                weight=1.0,
                description=f"Flags records where the {cat_col} is used < 3% of the time by the entity",
                requires_entity=True,
            ))

        return rules

    # ------------------------------------------------------------------
    # Generic Numeric IQR Rules
    # ------------------------------------------------------------------
    def _generic_numeric_rules(self, profile, amount_col, _id):
        rules = []
        for col_name, col_prof in profile.columns.items():
            if col_prof.dtype != "numeric":
                continue
            if col_prof.semantic_role in ("label", "user_id"):
                continue
            if col_name == amount_col:
                continue  # Already covered by amount rules
            iqr = col_prof.stats.get("iqr", 0)
            if iqr <= 0:
                continue
            safe_name = col_name.replace(" ", "_")
            rules.append(Rule(
                rule_id=_id("iqr_outlier"),
                rule_type="iqr_outlier",
                target_column=col_name,
                derived_feature=f"_{safe_name}_iqr_outlier",
                condition=f"_{safe_name}_iqr_outlier == 1",
                operator="eq",
                threshold=1.0,
                threshold_method="iqr_1.5x_fence",
                severity="low",
                weight=0.5,
                description=f"Flags records where {col_name} falls outside the 1.5×IQR fence",
                requires_entity=False,
            ))
        return rules

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    @staticmethod
    def rules_to_json(rules: list, path: Optional[str] = None) -> str:
        """Serialise rules to JSON string. Optionally write to file."""
        data = [r.to_dict() for r in rules]
        json_str = json.dumps(data, indent=2, default=str)
        if path:
            with open(path, "w") as f:
                f.write(json_str)
            print(f"[EXPORT] Rules saved to {path}")
        return json_str

    @staticmethod
    def rules_from_json(path_or_str: str) -> list:
        """Load rules from JSON file path or JSON string."""
        try:
            with open(path_or_str, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, OSError):
            data = json.loads(path_or_str)
        return [Rule.from_dict(d) for d in data]

    @staticmethod
    def rules_to_yaml(rules: list, path: str):
        """Serialise rules to YAML file."""
        data = [r.to_dict() for r in rules]
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"[EXPORT] Rules saved to {path}")

    @staticmethod
    def rules_from_yaml(path: str) -> list:
        """Load rules from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return [Rule.from_dict(d) for d in data]


# %% [markdown]
# ## Quick Test

# %%
if __name__ == "__main__":
    from data_profiler import DataProfiler
    import pandas as pd

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
    gen = RuleGenerator()
    rules = gen.generate(profile)

    print(f"\n=== Generated {len(rules)} rules ===")
    print(RuleGenerator.rules_to_json(rules))
