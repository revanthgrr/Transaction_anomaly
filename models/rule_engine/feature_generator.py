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
# # Feature Generator
#
# Automatically creates derived features based on the DataProfile.
# All features are generated **conditionally** — only if the required
# source columns exist. No hardcoded column names.
#
# Feature names are prefixed with `_` to avoid collisions.

# %%
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# %%
# Import the DataProfile type—handle both package and standalone usage.
try:
    from .data_profiler import DataProfile
except ImportError:
    from data_profiler import DataProfile

# %% [markdown]
# ## Haversine Utility

# %%
def _haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * asin(sqrt(a))


def _haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorised haversine for pandas Series."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

# %% [markdown]
# ## FeatureGenerator Class

# %%
class FeatureGenerator:
    """
    Creates derived features dynamically based on what columns are present
    in the DataProfile from the profiler stage.

    All generated features are prefixed with `_` to distinguish from originals.
    """

    def generate(self, df: pd.DataFrame, profile: DataProfile) -> pd.DataFrame:
        """Generate all applicable derived features and return enriched DataFrame."""
        enriched = df.copy()

        # Ensure timestamp is parsed
        time_col = profile.detected_time_col
        if time_col and time_col in enriched.columns:
            if not pd.api.types.is_datetime64_any_dtype(enriched[time_col]):
                try:
                    enriched[time_col] = pd.to_datetime(enriched[time_col], errors="coerce")
                except Exception:
                    time_col = None  # Give up on time features

        entity_col = profile.detected_entity_col
        amount_col = profile.detected_amount_col
        geo_cols = profile.detected_geo_cols  # (lat, lon) or None

        # ── Amount-based features ──────────────────────────────────────
        if amount_col and amount_col in enriched.columns:
            enriched = self._amount_features(enriched, amount_col, entity_col)

        # ── Temporal features ──────────────────────────────────────────
        if time_col and time_col in enriched.columns:
            enriched = self._temporal_features(enriched, time_col, entity_col)

        # ── Geolocation features ───────────────────────────────────────
        if geo_cols:
            lat_col, lon_col = geo_cols
            if lat_col in enriched.columns and lon_col in enriched.columns:
                enriched = self._geo_features(enriched, lat_col, lon_col,
                                              entity_col, time_col)

        # ── Category features ─────────────────────────────────────────
        for cat_col in profile.detected_category_cols:
            if cat_col in enriched.columns:
                enriched = self._category_features(enriched, cat_col, entity_col)

        # ── Generic numeric outlier features ──────────────────────────
        enriched = self._numeric_outlier_features(enriched, profile)

        # Sort by entity + time if possible
        sort_cols = []
        if entity_col and entity_col in enriched.columns:
            sort_cols.append(entity_col)
        if time_col and time_col in enriched.columns:
            sort_cols.append(time_col)
        if sort_cols:
            enriched = enriched.sort_values(sort_cols).reset_index(drop=True)

        generated = [c for c in enriched.columns if c.startswith("_")]
        print(f"[FEATURES] Generated {len(generated)} derived features: {generated}")
        return enriched

    # ------------------------------------------------------------------
    # Amount Features
    # ------------------------------------------------------------------
    def _amount_features(self, df: pd.DataFrame, amount_col: str,
                         entity_col: Optional[str]) -> pd.DataFrame:
        """z-score, percentile rank, rolling mean, ratio to median."""
        col = amount_col

        if entity_col and entity_col in df.columns:
            grp = df.groupby(entity_col)[col]

            # Per-entity z-score
            stats = df.groupby(entity_col)[col].agg(["mean", "std"]).reset_index()
            stats.columns = [entity_col, "_tmp_mean", "_tmp_std"]
            stats["_tmp_std"] = stats["_tmp_std"].fillna(1).replace(0, 1)
            df = df.merge(stats, on=entity_col, how="left")
            df[f"_{col}_zscore"] = ((df[col] - df["_tmp_mean"]) / df["_tmp_std"]).fillna(0)
            df.drop(columns=["_tmp_mean", "_tmp_std"], inplace=True)

            # Ratio to entity median
            medians = df.groupby(entity_col)[col].median().reset_index()
            medians.columns = [entity_col, "_tmp_median"]
            medians["_tmp_median"] = medians["_tmp_median"].replace(0, 1)
            df = df.merge(medians, on=entity_col, how="left")
            df[f"_{col}_ratio_to_median"] = (df[col] / df["_tmp_median"]).fillna(1)
            df.drop(columns=["_tmp_median"], inplace=True)

            # Percentile rank within entity
            df[f"_{col}_pct_rank"] = grp.rank(pct=True).fillna(0.5)

            # Rolling mean (5-window) per entity
            df[f"_{col}_rolling_mean_5"] = (
                grp.transform(lambda x: x.rolling(5, min_periods=1).mean())
            ).fillna(df[col])
        else:
            # Global z-score
            mean_val = df[col].mean()
            std_val = df[col].std()
            std_val = std_val if std_val > 0 else 1
            df[f"_{col}_zscore"] = ((df[col] - mean_val) / std_val).fillna(0)
            df[f"_{col}_pct_rank"] = df[col].rank(pct=True).fillna(0.5)
            median_val = df[col].median()
            median_val = median_val if median_val != 0 else 1
            df[f"_{col}_ratio_to_median"] = (df[col] / median_val).fillna(1)
            df[f"_{col}_rolling_mean_5"] = (
                df[col].rolling(5, min_periods=1).mean()
            ).fillna(df[col])

        return df

    # ------------------------------------------------------------------
    # Temporal Features
    # ------------------------------------------------------------------
    def _temporal_features(self, df: pd.DataFrame, time_col: str,
                           entity_col: Optional[str]) -> pd.DataFrame:
        """Hour, day_of_week, is_weekend, is_night, time_since_last."""
        ts = df[time_col]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            return df

        df["_hour"] = ts.dt.hour
        df["_day_of_week"] = ts.dt.dayofweek
        df["_is_weekend"] = df["_day_of_week"].isin([5, 6]).astype(int)
        df["_is_night"] = df["_hour"].between(0, 5).astype(int)

        if entity_col and entity_col in df.columns:
            df = df.sort_values([entity_col, time_col])
            df["_time_since_last"] = (
                df.groupby(entity_col)[time_col]
                .diff()
                .dt.total_seconds()
                .fillna(-1)
            )

            # Transaction frequency in last 1 hour per entity
            df["_txn_frequency_1h"] = (
                df.groupby(entity_col)[time_col]
                .transform(self._count_in_window)
            )
        else:
            df = df.sort_values(time_col)
            df["_time_since_last"] = ts.diff().dt.total_seconds().fillna(-1)
            df["_txn_frequency_1h"] = 1

        return df

    @staticmethod
    def _count_in_window(ts_series, window_seconds=3600):
        """Count records in the preceding window."""
        ts = ts_series.values.astype("datetime64[s]").astype(np.int64)
        counts = np.ones(len(ts), dtype=int)
        for i in range(1, len(ts)):
            c = 1
            for j in range(i - 1, max(-1, i - 30), -1):
                if (ts[i] - ts[j]) <= window_seconds:
                    c += 1
                else:
                    break
            counts[i] = c
        return pd.Series(counts, index=ts_series.index)

    # ------------------------------------------------------------------
    # Geolocation Features
    # ------------------------------------------------------------------
    def _geo_features(self, df: pd.DataFrame, lat_col: str, lon_col: str,
                      entity_col: Optional[str],
                      time_col: Optional[str]) -> pd.DataFrame:
        """Distance from entity center, travel speed."""
        if entity_col and entity_col in df.columns:
            centers = (
                df.groupby(entity_col)[[lat_col, lon_col]]
                .median()
                .reset_index()
                .rename(columns={lat_col: "_c_lat", lon_col: "_c_lon"})
            )
            df = df.merge(centers, on=entity_col, how="left")
            mask = df[lat_col].notna() & df["_c_lat"].notna()
            df["_distance_from_center"] = 0.0
            if mask.any():
                df.loc[mask, "_distance_from_center"] = _haversine_vec(
                    df.loc[mask, lat_col], df.loc[mask, lon_col],
                    df.loc[mask, "_c_lat"], df.loc[mask, "_c_lon"]
                )
            df.drop(columns=["_c_lat", "_c_lon"], inplace=True)
        else:
            med_lat = df[lat_col].median()
            med_lon = df[lon_col].median()
            mask = df[lat_col].notna()
            df["_distance_from_center"] = 0.0
            if mask.any():
                df.loc[mask, "_distance_from_center"] = _haversine_vec(
                    df.loc[mask, lat_col], df.loc[mask, lon_col],
                    med_lat, med_lon
                )

        # Travel speed (requires time + entity + sorted data)
        if time_col and time_col in df.columns:
            group_cols = [entity_col] if (entity_col and entity_col in df.columns) else []
            df = df.sort_values(group_cols + [time_col]) if group_cols else df.sort_values(time_col)

            df["_prev_lat"] = df.groupby(entity_col)[lat_col].shift(1) if entity_col else df[lat_col].shift(1)
            df["_prev_lon"] = df.groupby(entity_col)[lon_col].shift(1) if entity_col else df[lon_col].shift(1)
            df["_prev_time"] = df.groupby(entity_col)[time_col].shift(1) if entity_col else df[time_col].shift(1)

            mask = df["_prev_lat"].notna() & df[lat_col].notna() & df["_prev_time"].notna()
            df["_travel_speed_kmh"] = 0.0
            if mask.any():
                dist = _haversine_vec(
                    df.loc[mask, "_prev_lat"], df.loc[mask, "_prev_lon"],
                    df.loc[mask, lat_col], df.loc[mask, lon_col]
                )
                hours = (df.loc[mask, time_col] - df.loc[mask, "_prev_time"]).dt.total_seconds() / 3600
                hours = hours.replace(0, np.nan)
                df.loc[mask, "_travel_speed_kmh"] = (dist / hours).fillna(0)

            df.drop(columns=["_prev_lat", "_prev_lon", "_prev_time"], inplace=True, errors="ignore")

        return df

    # ------------------------------------------------------------------
    # Category Features
    # ------------------------------------------------------------------
    def _category_features(self, df: pd.DataFrame, cat_col: str,
                           entity_col: Optional[str]) -> pd.DataFrame:
        """Category frequency per entity."""
        safe_name = cat_col.replace(" ", "_")
        if entity_col and entity_col in df.columns:
            cat_counts = df.groupby([entity_col, cat_col]).size().reset_index(name="_cnt")
            totals = df.groupby(entity_col).size().reset_index(name="_total")
            cat_counts = cat_counts.merge(totals, on=entity_col)
            cat_counts[f"_{safe_name}_frequency"] = cat_counts["_cnt"] / cat_counts["_total"]
            df = df.merge(
                cat_counts[[entity_col, cat_col, f"_{safe_name}_frequency"]],
                on=[entity_col, cat_col],
                how="left",
            )
            df[f"_{safe_name}_frequency"] = df[f"_{safe_name}_frequency"].fillna(0)
        else:
            total = len(df)
            freqs = df[cat_col].value_counts() / total
            df[f"_{safe_name}_frequency"] = df[cat_col].map(freqs).fillna(0)

        return df

    # ------------------------------------------------------------------
    # Generic Numeric Outlier Features (IQR-based)
    # ------------------------------------------------------------------
    def _numeric_outlier_features(self, df: pd.DataFrame,
                                   profile: DataProfile) -> pd.DataFrame:
        """IQR outlier flag for each numeric column not already covered."""
        for col_name, col_prof in profile.columns.items():
            if col_prof.dtype != "numeric":
                continue
            if col_prof.semantic_role in ("label", "user_id"):
                continue
            # Skip if already has derived features
            if f"_{col_name}_zscore" in df.columns:
                continue
            iqr = col_prof.stats.get("iqr", 0)
            if iqr <= 0:
                continue
            p25 = col_prof.stats.get("p25", 0)
            p75 = col_prof.stats.get("p75", 0)
            lower = p25 - 1.5 * iqr
            upper = p75 + 1.5 * iqr
            safe_name = col_name.replace(" ", "_")
            df[f"_{safe_name}_iqr_outlier"] = (
                (df[col_name] < lower) | (df[col_name] > upper)
            ).astype(int)

        return df


# %% [markdown]
# ## Quick Test

# %%
if __name__ == "__main__":
    from data_profiler import DataProfiler

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
    gen = FeatureGenerator()
    enriched = gen.generate(test_df, profile)
    print(f"\nOriginal columns:  {len(test_df.columns)}")
    print(f"Enriched columns:  {len(enriched.columns)}")
    print(f"New features:      {[c for c in enriched.columns if c.startswith('_')]}")
