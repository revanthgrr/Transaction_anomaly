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
# # ML Integrations
#
# Optional machine learning model hooks that can be plugged into the
# rule engine pipeline. Each integration implements the `MLIntegration`
# interface and produces anomaly predictions that are added as
# additional "rules" with scores.
#
# These are **opt-in** and do not affect the core rule engine.

# %%
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Base Interface

# %%
class MLIntegration(ABC):
    """Base class for ML model integrations."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the integration."""
        ...

    @abstractmethod
    def fit_predict(self, df: pd.DataFrame, numeric_cols: list) -> pd.Series:
        """
        Fit and predict anomaly scores for each row.

        Args:
            df: The enriched DataFrame
            numeric_cols: List of numeric feature columns to use

        Returns:
            Series of anomaly scores (0.0 = normal, 1.0 = anomaly)
        """
        ...

# %% [markdown]
# ## Isolation Forest Integration

# %%
class IsolationForestIntegration(MLIntegration):
    """Wraps sklearn's IsolationForest as an optional anomaly scorer."""

    def __init__(self, contamination="auto", n_estimators=100, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

    def name(self) -> str:
        return "Isolation Forest"

    def fit_predict(self, df: pd.DataFrame, numeric_cols: list) -> pd.Series:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler
        from sklearn.impute import SimpleImputer

        X = df[numeric_cols].copy()
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_imp)

        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )

        # decision_function returns negative scores for anomalies
        raw_scores = model.decision_function(X_scaled)
        predictions = model.predict(X_scaled)

        # Normalise: convert decision_function to [0, 1] anomaly score
        # More negative = more anomalous
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            normalised = 1 - (raw_scores - min_s) / (max_s - min_s)
        else:
            normalised = np.zeros(len(raw_scores))

        # Only keep scores for predicted anomalies, zero for normal
        normalised[predictions == 1] = 0.0

        return pd.Series(normalised, index=df.index, name="_ml_iforest_score")

# %% [markdown]
# ## XGBoost Integration

# %%
class XGBoostIntegration(MLIntegration):
    """
    Wraps XGBoost as an optional anomaly scorer.

    If ground-truth labels exist, trains supervised. Otherwise, uses
    the top 5% of the rule engine's anomaly scores as synthetic labels.
    """

    def __init__(self, n_estimators=200, max_depth=6, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def name(self) -> str:
        return "XGBoost"

    def fit_predict(self, df: pd.DataFrame, numeric_cols: list,
                    label_col: Optional[str] = None,
                    anomaly_score_col: str = "_anomaly_score") -> pd.Series:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("[ML] XGBoost not installed. Skipping.")
            return pd.Series(np.zeros(len(df)), index=df.index, name="_ml_xgb_score")

        from sklearn.model_selection import train_test_split
        from sklearn.impute import SimpleImputer

        X = df[numeric_cols].copy()
        imputer = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

        # Determine labels
        if label_col and label_col in df.columns:
            y = df[label_col].astype(int)
        elif anomaly_score_col in df.columns:
            threshold = df[anomaly_score_col].quantile(0.95)
            y = (df[anomaly_score_col] >= threshold).astype(int)
        else:
            threshold = X_imp.iloc[:, 0].quantile(0.95)
            y = (X_imp.iloc[:, 0] >= threshold).astype(int)

        if y.sum() < 2 or (y == 0).sum() < 2:
            print("[ML] Not enough samples in both classes for XGBoost.")
            return pd.Series(np.zeros(len(df)), index=df.index, name="_ml_xgb_score")

        n_neg = (y == 0).sum()
        n_pos = max((y == 1).sum(), 1)

        model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            scale_pos_weight=n_neg / n_pos,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=self.random_state,
            n_jobs=-1,
        )

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_imp, y, test_size=0.25, random_state=42, stratify=y
            )
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            probs = model.predict_proba(X_imp)[:, 1]
        except Exception as e:
            print(f"[ML] XGBoost training failed: {e}")
            return pd.Series(np.zeros(len(df)), index=df.index, name="_ml_xgb_score")

        return pd.Series(probs, index=df.index, name="_ml_xgb_score")

# %% [markdown]
# ## Integration Runner

# %%
def run_ml_integrations(df: pd.DataFrame, profile,
                        enable_iforest: bool = True,
                        enable_xgboost: bool = False) -> pd.DataFrame:
    """
    Run selected ML integrations and add their scores to the DataFrame.

    Args:
        df: Enriched DataFrame (after feature generation + rule evaluation)
        profile: DataProfile from the profiler
        enable_iforest: Whether to run Isolation Forest
        enable_xgboost: Whether to run XGBoost

    Returns:
        DataFrame with additional ML score columns
    """
    result = df.copy()

    # Collect numeric feature columns (both original and derived)
    numeric_cols = [
        c for c in result.columns
        if pd.api.types.is_numeric_dtype(result[c])
        and c not in ("_is_anomaly", "_anomaly_score", "_rules_triggered_count")
        and not c.startswith("_ml_")
    ]

    # Filter out label columns
    if profile.detected_label_col:
        numeric_cols = [c for c in numeric_cols if c != profile.detected_label_col]

    if len(numeric_cols) < 2:
        print("[ML] Not enough numeric features for ML models.")
        return result

    label_col = profile.detected_label_col

    if enable_iforest:
        print("\n[ML] Running Isolation Forest integration...")
        ifo = IsolationForestIntegration()
        result["_ml_iforest_score"] = ifo.fit_predict(result, numeric_cols)
        anomalies = (result["_ml_iforest_score"] > 0).sum()
        print(f"  → Isolation Forest flagged {anomalies} anomalies")

    if enable_xgboost:
        print("\n[ML] Running XGBoost integration...")
        xgb = XGBoostIntegration()
        result["_ml_xgb_score"] = xgb.fit_predict(result, numeric_cols, label_col)
        anomalies = (result["_ml_xgb_score"] > 0.5).sum()
        print(f"  → XGBoost flagged {anomalies} with >50% probability")

    return result


# %% [markdown]
# ## Quick Test

# %%
if __name__ == "__main__":
    np.random.seed(42)
    test_df = pd.DataFrame({
        "a": np.random.normal(100, 10, 200),
        "b": np.random.normal(50, 5, 200),
        "c": np.random.exponential(20, 200),
    })

    ifo = IsolationForestIntegration()
    scores = ifo.fit_predict(test_df, ["a", "b", "c"])
    print(f"IForest anomaly scores > 0: {(scores > 0).sum()}")
