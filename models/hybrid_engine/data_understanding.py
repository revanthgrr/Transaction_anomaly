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
# # Data Understanding Layer
#
# Wraps the existing `DataProfiler` and adds:
# - Missing value analysis
# - Distribution profiling (skew, kurtosis, modality)
# - Feature importance pre-analysis (mutual information with label)

# %%
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import warnings

warnings.filterwarnings("ignore")

# %%
import sys, os
# Handle both package imports and standalone Colab execution
try:
    from rule_engine.data_profiler import DataProfiler, DataProfile
except ImportError:
    try:
        from ..rule_engine.data_profiler import DataProfiler, DataProfile
    except ImportError:
        # Colab fallback: add models/ to path
        _models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        if _models_dir not in sys.path:
            sys.path.insert(0, _models_dir)
        from rule_engine.data_profiler import DataProfiler, DataProfile


# %% [markdown]
# ## Data Understanding Report

# %%
@dataclass
class MissingValueReport:
    """Missing value analysis for the dataset."""
    total_missing: int = 0
    total_cells: int = 0
    missing_rate: float = 0.0
    columns_with_missing: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_missing": self.total_missing,
            "total_cells": self.total_cells,
            "missing_rate": round(self.missing_rate, 4),
            "columns_with_missing": self.columns_with_missing,
        }


@dataclass
class DistributionReport:
    """Distribution profiling for numeric columns."""
    column_distributions: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.column_distributions


@dataclass
class DataUnderstandingReport:
    """Complete data understanding report."""
    profile: DataProfile = None
    missing_report: MissingValueReport = None
    distribution_report: DistributionReport = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile.to_dict() if self.profile else None,
            "missing_values": self.missing_report.to_dict() if self.missing_report else None,
            "distributions": self.distribution_report.to_dict() if self.distribution_report else None,
            "feature_importance_to_label": self.feature_importance,
            "warnings": self.warnings,
        }


# %% [markdown]
# ## DataUnderstanding Class

# %%
class DataUnderstanding:
    """
    Layer 1: Automatic data understanding.

    Profiles the dataset, analyzes missing values, distributions,
    and computes preliminary feature importance if labels exist.
    """

    def __init__(self):
        self.profiler = DataProfiler()

    def analyze(self, df: pd.DataFrame, verbose: bool = True) -> DataUnderstandingReport:
        """Run full data understanding analysis."""
        if verbose:
            print("=" * 60)
            print("  LAYER 1: DATA UNDERSTANDING")
            print("=" * 60)

        # 1. Profile
        if verbose:
            print("\n[1.1] Profiling schema...")
        profile = self.profiler.profile(df)

        if verbose:
            print(f"  → {profile.row_count:,} rows × {len(profile.columns)} columns")
            print(f"  → Entity: {profile.detected_entity_col}")
            print(f"  → Amount: {profile.detected_amount_col}")
            print(f"  → Time:   {profile.detected_time_col}")
            print(f"  → Geo:    {profile.detected_geo_cols}")
            print(f"  → Label:  {profile.detected_label_col}")
            print(f"  → Categories: {profile.detected_category_cols}")

        # 2. Missing value analysis
        if verbose:
            print("\n[1.2] Analyzing missing values...")
        missing_report = self._analyze_missing(df, verbose)

        # 3. Distribution profiling
        if verbose:
            print("\n[1.3] Profiling distributions...")
        dist_report = self._analyze_distributions(df, profile, verbose)

        # 4. Feature importance pre-analysis
        if verbose:
            print("\n[1.4] Computing feature importance...")
        importance = self._compute_feature_importance(df, profile, verbose)

        # 5. Generate warnings
        warnings_list = self._generate_warnings(df, profile, missing_report)

        if verbose and warnings_list:
            print(f"\n[⚠] {len(warnings_list)} warnings:")
            for w in warnings_list:
                print(f"  ⚠ {w}")

        report = DataUnderstandingReport(
            profile=profile,
            missing_report=missing_report,
            distribution_report=dist_report,
            feature_importance=importance,
            warnings=warnings_list,
        )

        if verbose:
            print("\n[✓] Data understanding complete.")

        return report

    # ------------------------------------------------------------------
    # Missing Value Analysis
    # ------------------------------------------------------------------
    def _analyze_missing(self, df: pd.DataFrame,
                         verbose: bool) -> MissingValueReport:
        total_cells = df.shape[0] * df.shape[1]
        total_missing = int(df.isna().sum().sum())
        missing_rate = total_missing / total_cells if total_cells > 0 else 0

        cols_missing = {}
        for col in df.columns:
            n_miss = int(df[col].isna().sum())
            if n_miss > 0:
                cols_missing[col] = {
                    "count": n_miss,
                    "rate": round(n_miss / len(df), 4),
                }

        if verbose:
            print(f"  → Total missing: {total_missing:,} / {total_cells:,} "
                  f"({missing_rate*100:.2f}%)")
            if cols_missing:
                print(f"  → {len(cols_missing)} columns with missing values")
            else:
                print("  → No missing values found ✓")

        return MissingValueReport(
            total_missing=total_missing,
            total_cells=total_cells,
            missing_rate=missing_rate,
            columns_with_missing=cols_missing,
        )

    # ------------------------------------------------------------------
    # Distribution Profiling
    # ------------------------------------------------------------------
    def _analyze_distributions(self, df: pd.DataFrame, profile: DataProfile,
                                verbose: bool) -> DistributionReport:
        distributions = {}

        for col_name, col_prof in profile.columns.items():
            if col_prof.dtype != "numeric":
                continue
            if col_prof.semantic_role in ("label", "user_id"):
                continue

            series = df[col_name].dropna()
            if len(series) < 10:
                continue

            skew = float(series.skew())
            kurt = float(series.kurtosis())

            # Determine distribution shape
            if abs(skew) < 0.5:
                shape = "symmetric"
            elif skew > 0:
                shape = "right-skewed"
            else:
                shape = "left-skewed"

            if kurt > 3:
                tail = "heavy-tailed (leptokurtic)"
            elif kurt < -1:
                tail = "light-tailed (platykurtic)"
            else:
                tail = "normal-tailed (mesokurtic)"

            distributions[col_name] = {
                "skewness": round(skew, 4),
                "kurtosis": round(kurt, 4),
                "shape": shape,
                "tail_behavior": tail,
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "range": round(float(series.max() - series.min()), 4),
            }

        if verbose:
            print(f"  → Profiled {len(distributions)} numeric distributions")
            # Show highly skewed ones
            skewed = {k: v for k, v in distributions.items()
                      if abs(v["skewness"]) > 2}
            if skewed:
                print(f"  → {len(skewed)} highly skewed columns: "
                      f"{list(skewed.keys())[:5]}")

        return DistributionReport(column_distributions=distributions)

    # ------------------------------------------------------------------
    # Feature Importance Pre-Analysis
    # ------------------------------------------------------------------
    def _compute_feature_importance(self, df: pd.DataFrame,
                                     profile: DataProfile,
                                     verbose: bool) -> Dict[str, float]:
        label_col = profile.detected_label_col
        if not label_col or label_col not in df.columns:
            if verbose:
                print("  → No label column found, skipping importance analysis")
            return {}

        try:
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.impute import SimpleImputer
        except ImportError:
            if verbose:
                print("  → sklearn not available, skipping")
            return {}

        # Select numeric features only
        numeric_cols = [
            col for col, cp in profile.columns.items()
            if cp.dtype == "numeric" and cp.semantic_role not in ("label", "user_id")
        ]

        if len(numeric_cols) < 2:
            if verbose:
                print("  → Not enough numeric features for importance analysis")
            return {}

        X = df[numeric_cols].copy()
        y = df[label_col].astype(int)

        # Impute missing
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)

        # Subsample for speed
        if len(X_imp) > 10000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_imp), 10000, replace=False)
            X_imp = X_imp[idx]
            y = y.iloc[idx]

        try:
            mi_scores = mutual_info_classif(X_imp, y, random_state=42)
            importance = {
                col: round(float(score), 4)
                for col, score in zip(numeric_cols, mi_scores)
            }
            importance = dict(sorted(importance.items(),
                                     key=lambda x: x[1], reverse=True))
        except Exception as e:
            if verbose:
                print(f"  → Importance computation failed: {e}")
            return {}

        if verbose:
            top3 = list(importance.items())[:3]
            print(f"  → Top features by mutual information: "
                  f"{', '.join(f'{k}={v:.3f}' for k, v in top3)}")

        return importance

    # ------------------------------------------------------------------
    # Warning Generation
    # ------------------------------------------------------------------
    def _generate_warnings(self, df: pd.DataFrame, profile: DataProfile,
                           missing_report: MissingValueReport) -> List[str]:
        warnings_list = []

        # High missing rate
        if missing_report.missing_rate > 0.1:
            warnings_list.append(
                f"High overall missing rate: {missing_report.missing_rate*100:.1f}%"
            )

        # Columns with > 50% missing
        for col, info in missing_report.columns_with_missing.items():
            if info["rate"] > 0.5:
                warnings_list.append(
                    f"Column '{col}' has {info['rate']*100:.0f}% missing values"
                )

        # No entity column
        if not profile.detected_entity_col:
            warnings_list.append(
                "No entity/user column detected — per-entity features disabled"
            )

        # No timestamp
        if not profile.detected_time_col:
            warnings_list.append(
                "No timestamp column detected — temporal features disabled"
            )

        # Severe class imbalance
        label_col = profile.detected_label_col
        if label_col and label_col in df.columns:
            fraud_rate = df[label_col].astype(int).mean()
            if fraud_rate < 0.01:
                warnings_list.append(
                    f"Severe class imbalance: fraud rate = {fraud_rate*100:.2f}%"
                )

        return warnings_list


# %%
if __name__ == "__main__":
    np.random.seed(42)
    test_df = pd.DataFrame({
        "txn_id": range(500),
        "user_id": np.random.randint(1, 20, 500),
        "amount": np.random.exponential(50, 500),
        "timestamp": pd.date_range("2024-01-01", periods=500, freq="30min"),
        "lat": np.random.uniform(30, 45, 500),
        "lon": np.random.uniform(-120, -70, 500),
        "category": np.random.choice(["food", "gas", "electronics"], 500),
        "is_fraud": np.random.choice([0, 1], 500, p=[0.95, 0.05]),
    })

    du = DataUnderstanding()
    report = du.analyze(test_df)
    print(f"\nReport keys: {list(report.to_dict().keys())}")
