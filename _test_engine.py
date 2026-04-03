"""Quick end-to-end test of the Dynamic Rule Engine."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

import pandas as pd
import numpy as np

from rule_engine.data_profiler import DataProfiler
from rule_engine.feature_generator import FeatureGenerator
from rule_engine.rule_generator import RuleGenerator
from rule_engine.rule_evaluator import RuleEvaluator

np.random.seed(42)
df = pd.DataFrame({
    "txn_id": range(300),
    "customer_id": np.random.randint(1, 15, 300),
    "amount": np.random.exponential(50, 300),
    "event_time": pd.date_range("2024-01-01", periods=300, freq="30min"),
    "lat": np.random.uniform(30, 45, 300),
    "lon": np.random.uniform(-120, -70, 300),
    "category": np.random.choice(["food", "gas", "electronics", "travel"], 300),
    "is_fraud": np.random.choice([0, 1], 300, p=[0.95, 0.05]),
})

print("=== Stage 1: Profiling ===")
profiler = DataProfiler()
profile = profiler.profile(df)
print(f"  entity={profile.detected_entity_col}")
print(f"  amount={profile.detected_amount_col}")
print(f"  time={profile.detected_time_col}")
print(f"  geo={profile.detected_geo_cols}")
print(f"  label={profile.detected_label_col}")
print(f"  categories={profile.detected_category_cols}")

print("\n=== Stage 2: Feature Generation ===")
enriched = FeatureGenerator().generate(df, profile)
derived = [c for c in enriched.columns if c.startswith("_")]
print(f"  Generated {len(derived)} derived features")

print("\n=== Stage 3: Rule Generation ===")
rules = RuleGenerator().generate(profile)
print(f"  Generated {len(rules)} rules")

print("\n=== Stage 4: Evaluation ===")
result_df, eval_result = RuleEvaluator().evaluate(enriched, rules, profile)
print(f"  Anomalies: {eval_result.total_anomalies} / {eval_result.total_rows}")
print(f"  Rate: {eval_result.anomaly_rate*100:.2f}%")

print("\nTop 5 anomalous rows:")
top = result_df.nlargest(5, "_anomaly_score")
for _, row in top.iterrows():
    expls = row["_explanations"]
    reasons = [e["reason"] for e in expls] if expls else ["none"]
    print(f"  Score={row['_anomaly_score']:.3f} | {'; '.join(reasons[:2])}")

print("\n✅ SUCCESS: Core engine works end-to-end!")
