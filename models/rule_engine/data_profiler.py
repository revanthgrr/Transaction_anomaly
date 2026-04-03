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
# # Data Profiler
#
# Automatically analyzes any uploaded dataset and produces a schema profile.
# Detects column types (numeric, categorical, datetime, geo, text, identifier)
# and infers semantic roles (amount, timestamp, user_id, lat, lon, category,
# label) using keyword matching + statistical heuristics.
#
# **Zero hardcoded column names.** The profiler adapts to any schema.

# %%
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Column & Data Profile Dataclasses

# %%
@dataclass
class ColumnProfile:
    """Profile for a single column in the dataset."""
    name: str
    dtype: str  # numeric, categorical, datetime, geo, text, identifier
    semantic_role: Optional[str] = None  # amount, timestamp, user_id, lat, lon, category, label
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "semantic_role": self.semantic_role,
            "stats": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                      for k, v in self.stats.items()},
        }


@dataclass
class DataProfile:
    """Complete profile of a dataset."""
    columns: dict  # str -> ColumnProfile
    row_count: int = 0
    detected_entity_col: Optional[str] = None      # user_id equivalent
    detected_amount_col: Optional[str] = None       # amount equivalent
    detected_time_col: Optional[str] = None         # timestamp equivalent
    detected_geo_cols: Optional[tuple] = None       # (lat_col, lon_col)
    detected_category_cols: list = field(default_factory=list)
    detected_label_col: Optional[str] = None        # ground-truth label

    def to_dict(self) -> dict:
        return {
            "row_count": self.row_count,
            "detected_entity_col": self.detected_entity_col,
            "detected_amount_col": self.detected_amount_col,
            "detected_time_col": self.detected_time_col,
            "detected_geo_cols": list(self.detected_geo_cols) if self.detected_geo_cols else None,
            "detected_category_cols": self.detected_category_cols,
            "detected_label_col": self.detected_label_col,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
        }

# %% [markdown]
# ## Semantic Keyword Dictionaries
#
# These are used for fuzzy matching against column names. They are
# intentionally broad so the system works across domains (banking,
# e-commerce, healthcare, logistics, etc.).

# %%
_AMOUNT_KEYWORDS = ["amt", "amount", "price", "value", "total", "cost",
                    "revenue", "salary", "payment", "fee", "charge", "balance",
                    "spend", "income", "sales"]

_TIME_KEYWORDS = ["time", "date", "timestamp", "datetime", "created",
                  "updated", "occurred", "posted", "trans_date", "event_time",
                  "order_date", "purchase_date"]

_ENTITY_KEYWORDS = ["user", "id", "account", "customer", "client", "member",
                    "card", "cc_num", "employee", "patient", "device",
                    "session", "merchant_id", "vendor"]

_LAT_KEYWORDS = ["lat", "latitude"]
_LON_KEYWORDS = ["lon", "long", "longitude", "lng"]

_CATEGORY_KEYWORDS = ["cat", "category", "type", "class", "group", "segment",
                      "department", "genre", "status", "level", "tier", "kind"]

_LABEL_KEYWORDS = ["is_fraud", "fraud", "label", "target", "is_anomaly",
                   "anomaly", "suspicious", "flagged", "outcome", "result"]

# %% [markdown]
# ## DataProfiler Class

# %%
class DataProfiler:
    """
    Schema-agnostic data profiler.

    Analyzes any DataFrame and produces a DataProfile containing:
    - Column type classification (numeric, categorical, datetime, geo, text, id)
    - Semantic role inference (amount, timestamp, user_id, lat, lon, category, label)
    - Statistical summaries per column
    """

    def profile(self, df: pd.DataFrame) -> DataProfile:
        """Run full profiling on a DataFrame."""
        columns = {}
        for col in df.columns:
            columns[col] = self._profile_column(df, col)

        profile = DataProfile(columns=columns, row_count=len(df))

        # --- Semantic role assignment (priority-ordered, first-match) ---
        self._assign_semantic_roles(df, profile)

        return profile

    # ------------------------------------------------------------------
    # Column-Level Profiling
    # ------------------------------------------------------------------
    def _profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        """Classify a single column and compute statistics."""
        series = df[col]
        dtype = self._classify_dtype(df, col)
        stats = self._compute_stats(series, dtype)
        return ColumnProfile(name=col, dtype=dtype, stats=stats)

    def _classify_dtype(self, df: pd.DataFrame, col: str) -> str:
        """Determine the high-level type of a column."""
        series = df[col]
        col_lower = col.lower()

        # 1. Already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # 2. Try parsing as datetime
        if series.dtype == object:
            sample = series.dropna().head(50)
            if len(sample) > 0:
                try:
                    parsed = pd.to_datetime(sample, infer_datetime_format=True)
                    if parsed.notna().mean() > 0.8:
                        return "datetime"
                except (ValueError, TypeError, OverflowError):
                    pass

        # 3. Numeric
        if pd.api.types.is_numeric_dtype(series):
            # Check if it's geolocation
            if self._looks_like_geo(series, col_lower):
                return "geo"
            # Check if it's an identifier (integer with high cardinality ratio)
            nunique = series.nunique()
            if nunique > 0.5 * len(series) and nunique > 50:
                # High-cardinality numeric -> likely an identifier
                if any(kw in col_lower for kw in _ENTITY_KEYWORDS):
                    return "identifier"
            return "numeric"

        # 4. Categorical vs Text
        if series.dtype == object or pd.api.types.is_categorical_dtype(series):
            nunique = series.nunique()
            avg_len = series.dropna().astype(str).str.len().mean()
            # Low cardinality or short strings -> categorical
            if nunique < 50 or (nunique < 0.1 * len(series) and avg_len < 30):
                return "categorical"
            # High cardinality + long strings -> text
            if avg_len > 50:
                return "text"
            return "identifier"

        return "text"

    def _looks_like_geo(self, series: pd.Series, col_lower: str) -> bool:
        """Check if a numeric column contains geographic coordinates."""
        # Binary / low-cardinality columns are never geo (e.g. is_fraud 0/1)
        if series.nunique() <= 5:
            return False
        if any(kw in col_lower for kw in _LAT_KEYWORDS + _LON_KEYWORDS):
            return True
        # Statistical check: lat in [-90, 90], lon in [-180, 180]
        if series.notna().sum() < 10:
            return False
        q_min, q_max = series.quantile(0.01), series.quantile(0.99)
        if -90 <= q_min and q_max <= 90 and series.std() < 50:
            return True  # Likely latitude
        if -180 <= q_min and q_max <= 180 and series.std() < 100:
            return True  # Likely longitude
        return False

    def _compute_stats(self, series: pd.Series, dtype: str) -> dict:
        """Compute relevant statistics based on column type."""
        stats = {
            "null_count": int(series.isna().sum()),
            "null_rate": round(float(series.isna().mean()), 4),
            "unique_count": int(series.nunique()),
        }

        if dtype == "numeric" or dtype == "geo":
            num = series.dropna()
            if len(num) > 0:
                stats.update({
                    "mean": round(float(num.mean()), 4),
                    "std": round(float(num.std()), 4),
                    "min": round(float(num.min()), 4),
                    "max": round(float(num.max()), 4),
                    "p1": round(float(num.quantile(0.01)), 4),
                    "p5": round(float(num.quantile(0.05)), 4),
                    "p25": round(float(num.quantile(0.25)), 4),
                    "p50": round(float(num.quantile(0.50)), 4),
                    "p75": round(float(num.quantile(0.75)), 4),
                    "p95": round(float(num.quantile(0.95)), 4),
                    "p99": round(float(num.quantile(0.99)), 4),
                    "skewness": round(float(num.skew()), 4),
                    "kurtosis": round(float(num.kurtosis()), 4),
                    "iqr": round(float(num.quantile(0.75) - num.quantile(0.25)), 4),
                })

        elif dtype == "categorical" or dtype == "identifier":
            top_values = series.value_counts().head(10).to_dict()
            stats["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        elif dtype == "datetime":
            try:
                parsed = pd.to_datetime(series, errors="coerce")
                valid = parsed.dropna()
                if len(valid) > 0:
                    stats["min_date"] = str(valid.min())
                    stats["max_date"] = str(valid.max())
                    stats["date_range_days"] = int((valid.max() - valid.min()).days)
            except Exception:
                pass

        return stats

    # ------------------------------------------------------------------
    # Semantic Role Assignment
    # ------------------------------------------------------------------
    def _assign_semantic_roles(self, df: pd.DataFrame, profile: DataProfile):
        """
        Assign semantic roles to columns using keyword matching + heuristics.
        Each role is assigned to at most one column (first-match wins).
        """
        assigned_roles = set()

        # Pass 1: Keyword-based matching (high confidence)
        for col_name, col_prof in profile.columns.items():
            cl = col_name.lower()

            # Label / ground truth
            if "label" not in assigned_roles:
                if any(kw == cl or kw in cl for kw in _LABEL_KEYWORDS):
                    if col_prof.dtype in ("numeric", "categorical"):
                        unique_vals = df[col_name].nunique()
                        if unique_vals <= 5:
                            col_prof.semantic_role = "label"
                            profile.detected_label_col = col_name
                            assigned_roles.add("label")
                            continue

            # Timestamp
            if "timestamp" not in assigned_roles:
                if col_prof.dtype == "datetime":
                    col_prof.semantic_role = "timestamp"
                    profile.detected_time_col = col_name
                    assigned_roles.add("timestamp")
                    continue
                if any(kw in cl for kw in _TIME_KEYWORDS):
                    col_prof.semantic_role = "timestamp"
                    profile.detected_time_col = col_name
                    assigned_roles.add("timestamp")
                    continue

            # Latitude
            if "lat" not in assigned_roles:
                if col_prof.dtype == "geo" and any(kw in cl for kw in _LAT_KEYWORDS):
                    col_prof.semantic_role = "lat"
                    assigned_roles.add("lat")
                    continue

            # Longitude
            if "lon" not in assigned_roles:
                if col_prof.dtype == "geo" and any(kw in cl for kw in _LON_KEYWORDS):
                    col_prof.semantic_role = "lon"
                    assigned_roles.add("lon")
                    continue

            # Amount
            if "amount" not in assigned_roles:
                if col_prof.dtype == "numeric" and any(kw in cl for kw in _AMOUNT_KEYWORDS):
                    col_prof.semantic_role = "amount"
                    profile.detected_amount_col = col_name
                    assigned_roles.add("amount")
                    continue

            # Entity / User ID
            if "user_id" not in assigned_roles:
                if col_prof.dtype in ("identifier", "numeric", "categorical"):
                    if any(kw in cl for kw in _ENTITY_KEYWORDS):
                        col_prof.semantic_role = "user_id"
                        profile.detected_entity_col = col_name
                        assigned_roles.add("user_id")
                        continue

            # Category
            if any(kw in cl for kw in _CATEGORY_KEYWORDS):
                if col_prof.dtype == "categorical":
                    col_prof.semantic_role = "category"
                    profile.detected_category_cols.append(col_name)
                    continue

        # Pass 2: Fallback heuristics for missing roles
        # Amount fallback: pick the numeric column with highest variance
        if "amount" not in assigned_roles:
            best_col, best_var = None, -1
            for col_name, col_prof in profile.columns.items():
                if col_prof.dtype == "numeric" and col_prof.semantic_role is None:
                    var = col_prof.stats.get("std", 0)
                    if var > best_var:
                        best_var = var
                        best_col = col_name
            if best_col:
                profile.columns[best_col].semantic_role = "amount"
                profile.detected_amount_col = best_col
                assigned_roles.add("amount")

        # Entity fallback: pick first identifier column
        if "user_id" not in assigned_roles:
            for col_name, col_prof in profile.columns.items():
                if col_prof.dtype == "identifier" and col_prof.semantic_role is None:
                    col_prof.semantic_role = "user_id"
                    profile.detected_entity_col = col_name
                    assigned_roles.add("user_id")
                    break

        # Geo fallback: find pairs of numeric columns in lat/lon ranges
        if "lat" not in assigned_roles or "lon" not in assigned_roles:
            lat_candidate, lon_candidate = None, None
            for col_name, col_prof in profile.columns.items():
                if col_prof.dtype == "geo" and col_prof.semantic_role is None:
                    q_min = col_prof.stats.get("min", 0)
                    q_max = col_prof.stats.get("max", 0)
                    if -90 <= q_min and q_max <= 90 and lat_candidate is None:
                        lat_candidate = col_name
                    elif -180 <= q_min and q_max <= 180 and lon_candidate is None:
                        lon_candidate = col_name
            if lat_candidate and lon_candidate:
                profile.columns[lat_candidate].semantic_role = "lat"
                profile.columns[lon_candidate].semantic_role = "lon"
                assigned_roles.update(["lat", "lon"])

        # Set geo_cols tuple
        lat_col = next((c for c, p in profile.columns.items() if p.semantic_role == "lat"), None)
        lon_col = next((c for c, p in profile.columns.items() if p.semantic_role == "lon"), None)
        if lat_col and lon_col:
            profile.detected_geo_cols = (lat_col, lon_col)

        # Category fallback: any remaining low-cardinality categorical columns
        for col_name, col_prof in profile.columns.items():
            if col_prof.dtype == "categorical" and col_prof.semantic_role is None:
                unique = col_prof.stats.get("unique_count", 999)
                if unique <= 30:
                    col_prof.semantic_role = "category"
                    if col_name not in profile.detected_category_cols:
                        profile.detected_category_cols.append(col_name)


# %% [markdown]
# ## Quick Test
#
# Run this file directly to test profiling on a sample DataFrame.

# %%
if __name__ == "__main__":
    # Synthetic test data
    np.random.seed(42)
    test_df = pd.DataFrame({
        "txn_id": range(100),
        "user_account": np.random.randint(1, 10, 100),
        "purchase_amount": np.random.exponential(50, 100),
        "event_date": pd.date_range("2024-01-01", periods=100, freq="h"),
        "store_lat": np.random.uniform(30, 45, 100),
        "store_long": np.random.uniform(-120, -70, 100),
        "product_type": np.random.choice(["electronics", "grocery", "clothing"], 100),
        "is_fraud": np.random.choice([0, 1], 100, p=[0.95, 0.05]),
    })

    profiler = DataProfiler()
    result = profiler.profile(test_df)

    print(f"Rows: {result.row_count}")
    print(f"Entity: {result.detected_entity_col}")
    print(f"Amount: {result.detected_amount_col}")
    print(f"Time:   {result.detected_time_col}")
    print(f"Geo:    {result.detected_geo_cols}")
    print(f"Label:  {result.detected_label_col}")
    print(f"Cats:   {result.detected_category_cols}")
    print("\nColumn Profiles:")
    for name, cp in result.columns.items():
        print(f"  {name:20s} | type={cp.dtype:12s} | role={cp.semantic_role}")
