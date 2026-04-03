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
# # Unsupervised Anomaly Detection Module
#
# Adaptive unsupervised anomaly detection using multiple models:
# - **Isolation Forest** (always runs, any dataset size)
# - **HDBSCAN** (density-based, auto-tuning, subsampled for large data)
# - **Local Outlier Factor** (local density, subsampled for large data)
#
# Each model outputs a **continuous normalized score** (0.0 = normal, 1.0 = anomaly).
# Final score is a weighted average across all models.

# %%
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore")

# %%
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


# %% [markdown]
# ## Result Dataclass

# %%
@dataclass
class UnsupervisedResult:
    """Results from the unsupervised anomaly detection module."""
    iforest_scores: np.ndarray = None
    hdbscan_scores: np.ndarray = None
    lof_scores: np.ndarray = None
    combined_score: np.ndarray = None
    contamination_rate: float = 0.0
    models_used: List[str] = field(default_factory=list)
    model_weights: dict = field(default_factory=dict)


# %% [markdown]
# ## UnsupervisedModule Class

# %%
class UnsupervisedModule:
    """
    Layer 4: Adaptive unsupervised anomaly detection.

    Runs multiple unsupervised models, produces continuous anomaly scores,
    and combines them into a single normalized score.

    Automatically handles large datasets by subsampling for
    distance-based models (HDBSCAN, LOF).
    """

    SAMPLE_LIMIT = 50000  # Max rows for distance-based models

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._scaler = None
        self._imputer = None

    def detect(self, df: pd.DataFrame, feature_cols: List[str],
               verbose: bool = True) -> Tuple[np.ndarray, UnsupervisedResult]:
        """
        Run unsupervised anomaly detection on the given features.

        Args:
            df: Input DataFrame
            feature_cols: List of numeric column names to use
            verbose: Print progress

        Returns:
            (combined_scores_array, UnsupervisedResult)
        """
        if verbose:
            print("\n" + "=" * 60)
            print("  LAYER 4: UNSUPERVISED ANOMALY DETECTION")
            print("=" * 60)

        # Prepare data
        X = self._prepare_data(df, feature_cols, verbose)
        n_samples = X.shape[0]

        result = UnsupervisedResult()

        # --- Model 1: Isolation Forest (always runs on full data) ---
        if verbose:
            print(f"\n[4.1] Isolation Forest ({n_samples:,} samples)...")
        iforest_scores, contamination = self._run_isolation_forest(X)
        result.iforest_scores = iforest_scores
        result.contamination_rate = contamination
        result.models_used.append("isolation_forest")
        if verbose:
            n_anom = (iforest_scores > 0.5).sum()
            print(f"  → Anomalies (score > 0.5): {n_anom:,} "
                  f"({n_anom/n_samples*100:.2f}%)")
            print(f"  → Auto-tuned contamination: {contamination:.4f}")

        # --- PCA for distance-based methods ---
        n_components = min(10, X.shape[1])
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        if verbose:
            explained = sum(pca.explained_variance_ratio_) * 100
            print(f"\n[PCA] Reduced to {n_components} components "
                  f"(explains {explained:.1f}% variance)")

        # --- Model 2: HDBSCAN ---
        if verbose:
            print(f"\n[4.2] HDBSCAN...")
        hdbscan_scores = self._run_hdbscan(X_pca, n_samples, verbose)
        if hdbscan_scores is not None:
            result.hdbscan_scores = hdbscan_scores
            result.models_used.append("hdbscan")
            if verbose:
                n_anom = (hdbscan_scores > 0.5).sum()
                print(f"  → Anomalies (score > 0.5): {n_anom:,}")

        # --- Model 3: Local Outlier Factor ---
        if verbose:
            print(f"\n[4.3] Local Outlier Factor...")
        lof_scores = self._run_lof(X_pca, n_samples, verbose)
        if lof_scores is not None:
            result.lof_scores = lof_scores
            result.models_used.append("lof")
            if verbose:
                n_anom = (lof_scores > 0.5).sum()
                print(f"  → Anomalies (score > 0.5): {n_anom:,}")

        # --- Combine scores ---
        combined, weights = self._combine_scores(result)
        result.combined_score = combined
        result.model_weights = weights

        if verbose:
            print(f"\n[4.✓] Unsupervised module complete.")
            print(f"  → Models used: {result.models_used}")
            print(f"  → Weights: {weights}")
            n_anom = (combined > 0.5).sum()
            print(f"  → Combined anomalies (score > 0.5): {n_anom:,} "
                  f"({n_anom/n_samples*100:.2f}%)")

        return combined, result

    # ------------------------------------------------------------------
    # Data Preparation
    # ------------------------------------------------------------------
    def _prepare_data(self, df: pd.DataFrame, feature_cols: List[str],
                      verbose: bool) -> np.ndarray:
        """Impute, scale, and return numeric matrix."""
        X = df[feature_cols].copy()

        self._imputer = SimpleImputer(strategy="median")
        X_imp = self._imputer.fit_transform(X)

        self._scaler = RobustScaler()
        X_scaled = self._scaler.fit_transform(X_imp)

        if verbose:
            print(f"  → Prepared {X_scaled.shape[0]:,} × {X_scaled.shape[1]} "
                  f"feature matrix")

        return X_scaled

    # ------------------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------------------
    def _run_isolation_forest(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run Isolation Forest and return normalized scores + auto contamination."""
        model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Fit the model first
        model.fit(X)

        # decision_function: more negative = more anomalous
        raw_scores = model.decision_function(X)

        # Auto-tune contamination using IQR of decision scores
        q25 = np.percentile(raw_scores, 25)
        q75 = np.percentile(raw_scores, 75)
        iqr = q75 - q25
        lower_fence = q25 - 1.5 * iqr
        contamination = max(float(np.mean(raw_scores < lower_fence)), 0.001)
        contamination = min(contamination, 0.15)  # Cap at 15%

        # Normalize to [0, 1]: more negative → higher score
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            normalized = 1.0 - (raw_scores - min_s) / (max_s - min_s)
        else:
            normalized = np.zeros(len(raw_scores))

        return normalized, contamination

    # ------------------------------------------------------------------
    # HDBSCAN
    # ------------------------------------------------------------------
    def _run_hdbscan(self, X_pca: np.ndarray, n_samples: int,
                     verbose: bool) -> Optional[np.ndarray]:
        """Run HDBSCAN with auto-tuned min_cluster_size."""
        try:
            import hdbscan
        except ImportError:
            if verbose:
                print("  → HDBSCAN not installed, skipping")
            return None

        # Subsample if too large
        if n_samples > self.SAMPLE_LIMIT:
            if verbose:
                print(f"  → Subsampling {self.SAMPLE_LIMIT:,} / {n_samples:,} "
                      f"for HDBSCAN")
            rng = np.random.RandomState(self.random_state)
            sample_idx = rng.choice(n_samples, self.SAMPLE_LIMIT, replace=False)
            X_sample = X_pca[sample_idx]
        else:
            X_sample = X_pca
            sample_idx = None

        # Auto-tune min_cluster_size
        min_cluster = max(5, int(0.01 * len(X_sample)))
        min_cluster = min(min_cluster, 100)  # Cap

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster,
            min_samples=5,
            core_dist_n_jobs=-1,
        )
        clusterer.fit(X_sample)

        # Outlier scores from HDBSCAN (0 = inlier, 1 = outlier)
        sample_scores = clusterer.outlier_scores_

        # Normalize to [0, 1]
        min_s, max_s = sample_scores.min(), sample_scores.max()
        if max_s - min_s > 0:
            sample_scores_norm = (sample_scores - min_s) / (max_s - min_s)
        else:
            sample_scores_norm = np.zeros(len(sample_scores))

        # If subsampled, map scores back to full dataset via nearest neighbors
        if sample_idx is not None:
            if verbose:
                print("  → Mapping scores back to full dataset via k-NN...")
            full_scores = np.zeros(n_samples)
            full_scores[sample_idx] = sample_scores_norm

            # For un-sampled rows, find nearest sampled neighbor
            unsampled_mask = np.ones(n_samples, dtype=bool)
            unsampled_mask[sample_idx] = False
            unsampled_idx = np.where(unsampled_mask)[0]

            if len(unsampled_idx) > 0:
                nn = NearestNeighbors(n_neighbors=3, n_jobs=-1)
                nn.fit(X_pca[sample_idx])
                distances, indices = nn.kneighbors(X_pca[unsampled_idx])
                # Weighted average of nearest sampled neighbors' scores
                weights = 1.0 / (distances + 1e-8)
                weights = weights / weights.sum(axis=1, keepdims=True)
                neighbor_scores = sample_scores_norm[indices]
                full_scores[unsampled_idx] = (weights * neighbor_scores).sum(axis=1)

            return full_scores
        else:
            return sample_scores_norm

    # ------------------------------------------------------------------
    # Local Outlier Factor
    # ------------------------------------------------------------------
    def _run_lof(self, X_pca: np.ndarray, n_samples: int,
                 verbose: bool) -> Optional[np.ndarray]:
        """Run LOF with subsampling for large datasets."""
        # Subsample if too large
        if n_samples > self.SAMPLE_LIMIT:
            if verbose:
                print(f"  → Subsampling {self.SAMPLE_LIMIT:,} / {n_samples:,} "
                      f"for LOF")
            rng = np.random.RandomState(self.random_state)
            sample_idx = rng.choice(n_samples, self.SAMPLE_LIMIT, replace=False)
            X_sample = X_pca[sample_idx]
        else:
            X_sample = X_pca
            sample_idx = None

        n_neighbors = min(20, len(X_sample) - 1)
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=False,
            n_jobs=-1,
        )
        lof_labels = lof.fit_predict(X_sample)

        # negative_outlier_factor_: more negative = more anomalous
        raw_scores = -lof.negative_outlier_factor_  # Flip sign: higher = more anomalous

        # Normalize to [0, 1]
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s > 0:
            sample_scores_norm = (raw_scores - min_s) / (max_s - min_s)
        else:
            sample_scores_norm = np.zeros(len(raw_scores))

        # Map back if subsampled
        if sample_idx is not None:
            if verbose:
                print("  → Mapping LOF scores back via k-NN...")
            full_scores = np.zeros(n_samples)
            full_scores[sample_idx] = sample_scores_norm

            unsampled_mask = np.ones(n_samples, dtype=bool)
            unsampled_mask[sample_idx] = False
            unsampled_idx = np.where(unsampled_mask)[0]

            if len(unsampled_idx) > 0:
                nn = NearestNeighbors(n_neighbors=3, n_jobs=-1)
                nn.fit(X_pca[sample_idx])
                distances, indices = nn.kneighbors(X_pca[unsampled_idx])
                weights = 1.0 / (distances + 1e-8)
                weights = weights / weights.sum(axis=1, keepdims=True)
                neighbor_scores = sample_scores_norm[indices]
                full_scores[unsampled_idx] = (weights * neighbor_scores).sum(axis=1)

            return full_scores
        else:
            return sample_scores_norm

    # ------------------------------------------------------------------
    # Score Combination
    # ------------------------------------------------------------------
    def _combine_scores(self, result: UnsupervisedResult
                        ) -> Tuple[np.ndarray, dict]:
        """Combine scores from all models into a single score."""
        scores_list = []
        names = []

        if result.iforest_scores is not None:
            scores_list.append(result.iforest_scores)
            names.append("isolation_forest")

        if result.hdbscan_scores is not None:
            scores_list.append(result.hdbscan_scores)
            names.append("hdbscan")

        if result.lof_scores is not None:
            scores_list.append(result.lof_scores)
            names.append("lof")

        if len(scores_list) == 0:
            return np.zeros(0), {}

        # Equal weight average (simple, robust)
        weight = 1.0 / len(scores_list)
        weights = {name: round(weight, 4) for name in names}
        combined = np.mean(scores_list, axis=0)

        return combined, weights


# %%
if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    test_df = pd.DataFrame({
        "feat1": np.concatenate([np.random.normal(0, 1, 950),
                                  np.random.normal(5, 0.5, 50)]),
        "feat2": np.concatenate([np.random.normal(0, 1, 950),
                                  np.random.normal(-4, 0.3, 50)]),
        "feat3": np.random.exponential(2, n),
    })

    module = UnsupervisedModule()
    scores, result = module.detect(test_df, ["feat1", "feat2", "feat3"])
    print(f"\nScores shape: {scores.shape}")
    print(f"Top 10 scores: {sorted(scores, reverse=True)[:10]}")
