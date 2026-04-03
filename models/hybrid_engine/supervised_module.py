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
# # Supervised Learning Module (GPU-Accelerated)
#
# Optimized supervised fraud detection using:
# - **XGBoost** (gpu_hist on GPU, hist on CPU)
# - **LightGBM** (gpu on GPU, cpu fallback)
# - **CatBoost** (GPU task_type on GPU, CPU fallback)
#
# Features:
# - Automatic GPU detection (Colab, local CUDA)
# - Class imbalance handling (SMOTE / class weights)
# - Feature selection via importance thresholding
# - Hyperparameter tuning via **Optuna**
# - Auto-selects the best model by ROC-AUC

# %%
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import warnings
import time

warnings.filterwarnings("ignore")

# %%
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc


# %% [markdown]
# ## GPU Detection

# %%
def detect_gpu() -> bool:
    """Check if GPU is available for gradient boosting."""
    # Check NVIDIA GPU via subprocess
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check via torch if available
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass

    return False


# %% [markdown]
# ## Result Dataclass

# %%
@dataclass
class SupervisedResult:
    """Results from the supervised learning module."""
    best_model_name: str = ""
    best_model: Any = None
    best_roc_auc: float = 0.0
    best_pr_auc: float = 0.0
    best_f1: float = 0.0
    all_model_scores: Dict[str, dict] = field(default_factory=dict)
    predictions: np.ndarray = None
    feature_names: List[str] = field(default_factory=list)
    feature_importances: Dict[str, float] = field(default_factory=dict)
    optuna_best_params: Dict[str, dict] = field(default_factory=dict)
    gpu_used: bool = False


# %% [markdown]
# ## SupervisedModule Class

# %%
class SupervisedModule:
    """
    Layer 5: GPU-accelerated supervised fraud detection.

    Trains XGBoost, LightGBM, and CatBoost with Optuna hyperparameter
    tuning. Automatically handles class imbalance via SMOTE or class
    weights. Auto-selects the best performing model.
    """

    def __init__(self, n_optuna_trials: int = 20, random_state: int = 42):
        self.n_optuna_trials = n_optuna_trials
        self.random_state = random_state
        self.gpu_available = detect_gpu()

    def train_and_predict(self, df: pd.DataFrame, feature_cols: List[str],
                          label_col: str, verbose: bool = True
                          ) -> Tuple[np.ndarray, SupervisedResult]:
        """
        Train all models, tune hyperparameters, select best, and predict.

        Args:
            df: Input DataFrame
            feature_cols: Numeric feature columns
            label_col: Ground truth label column name
            verbose: Print progress

        Returns:
            (probability_array, SupervisedResult)
        """
        if verbose:
            print("\n" + "=" * 60)
            print("  LAYER 5: SUPERVISED LEARNING MODULE")
            print("=" * 60)
            print(f"  GPU Available: {'✓ YES' if self.gpu_available else '✗ NO (CPU mode)'}")

        # Prepare data
        X, y, selected_features = self._prepare_data(
            df, feature_cols, label_col, verbose
        )

        if X is None:
            if verbose:
                print("  → Not enough data to train supervised models")
            return np.zeros(len(df)), SupervisedResult()

        result = SupervisedResult(
            feature_names=selected_features,
            gpu_used=self.gpu_available,
        )

        # Handle imbalance
        X_train, y_train = self._handle_imbalance(X, y, verbose)

        # Train each model with Optuna
        models = {}
        model_scores = {}

        # --- XGBoost ---
        if verbose:
            print(f"\n[5.1] Training XGBoost (Optuna {self.n_optuna_trials} trials)...")
        t0 = time.time()
        xgb_model, xgb_params, xgb_score = self._train_xgboost(
            X, y, X_train, y_train, verbose
        )
        if xgb_model is not None:
            models["xgboost"] = xgb_model
            model_scores["xgboost"] = xgb_score
            result.optuna_best_params["xgboost"] = xgb_params
            if verbose:
                print(f"  → XGBoost ROC-AUC: {xgb_score['roc_auc']:.4f} | "
                      f"F1: {xgb_score['f1']:.4f} | "
                      f"Time: {time.time()-t0:.1f}s")

        # --- LightGBM ---
        if verbose:
            print(f"\n[5.2] Training LightGBM (Optuna {self.n_optuna_trials} trials)...")
        t0 = time.time()
        lgb_model, lgb_params, lgb_score = self._train_lightgbm(
            X, y, X_train, y_train, verbose
        )
        if lgb_model is not None:
            models["lightgbm"] = lgb_model
            model_scores["lightgbm"] = lgb_score
            result.optuna_best_params["lightgbm"] = lgb_params
            if verbose:
                print(f"  → LightGBM ROC-AUC: {lgb_score['roc_auc']:.4f} | "
                      f"F1: {lgb_score['f1']:.4f} | "
                      f"Time: {time.time()-t0:.1f}s")

        # --- CatBoost ---
        if verbose:
            print(f"\n[5.3] Training CatBoost (Optuna {self.n_optuna_trials} trials)...")
        t0 = time.time()
        cb_model, cb_params, cb_score = self._train_catboost(
            X, y, X_train, y_train, verbose
        )
        if cb_model is not None:
            models["catboost"] = cb_model
            model_scores["catboost"] = cb_score
            result.optuna_best_params["catboost"] = cb_params
            if verbose:
                print(f"  → CatBoost ROC-AUC: {cb_score['roc_auc']:.4f} | "
                      f"F1: {cb_score['f1']:.4f} | "
                      f"Time: {time.time()-t0:.1f}s")

        if not models:
            if verbose:
                print("\n  ⚠ No supervised models were successfully trained")
            return np.zeros(len(df)), result

        # Select best model
        best_name = max(model_scores, key=lambda k: model_scores[k]["roc_auc"])
        best_model = models[best_name]

        result.best_model_name = best_name
        result.best_model = best_model
        result.best_roc_auc = model_scores[best_name]["roc_auc"]
        result.best_pr_auc = model_scores[best_name]["pr_auc"]
        result.best_f1 = model_scores[best_name]["f1"]
        result.all_model_scores = model_scores

        # Get feature importances from best model
        result.feature_importances = self._get_feature_importances(
            best_model, best_name, selected_features
        )

        # Final prediction on full data
        probs = best_model.predict_proba(X)[:, 1]
        result.predictions = probs

        if verbose:
            print(f"\n[5.✓] Best model: {best_name}")
            print(f"  → ROC-AUC: {result.best_roc_auc:.4f}")
            print(f"  → PR-AUC:  {result.best_pr_auc:.4f}")
            print(f"  → F1:      {result.best_f1:.4f}")
            top_feats = sorted(result.feature_importances.items(),
                               key=lambda x: x[1], reverse=True)[:5]
            print(f"  → Top features: {[f'{k}={v:.3f}' for k, v in top_feats]}")

        return probs, result

    # ------------------------------------------------------------------
    # Data Preparation
    # ------------------------------------------------------------------
    def _prepare_data(self, df: pd.DataFrame, feature_cols: List[str],
                      label_col: str, verbose: bool
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Prepare features matrix with imputation and feature selection."""
        if label_col not in df.columns:
            if verbose:
                print(f"  ⚠ Label column '{label_col}' not found")
            return None, None, []

        y = df[label_col].astype(int).values

        # Need at least 10 per class
        if y.sum() < 10 or (y == 0).sum() < 10:
            if verbose:
                print(f"  ⚠ Not enough samples: {y.sum()} fraud, "
                      f"{(y==0).sum()} normal")
            return None, None, []

        # Filter valid features
        valid_cols = [c for c in feature_cols if c in df.columns
                      and pd.api.types.is_numeric_dtype(df[c])]

        X = df[valid_cols].copy()
        imputer = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(
            imputer.fit_transform(X), columns=valid_cols, index=X.index
        )

        # Feature selection: quick XGBoost fit, drop < 0.01 importance
        if len(valid_cols) > 5:
            selected = self._feature_selection(X_imp, y, valid_cols, verbose)
        else:
            selected = valid_cols

        if verbose:
            print(f"  → Features: {len(valid_cols)} → {len(selected)} "
                  f"(after selection)")
            print(f"  → Samples: {len(y):,} (fraud: {y.sum():,} = "
                  f"{y.mean()*100:.2f}%)")

        return X_imp[selected].values, y, selected

    def _feature_selection(self, X: pd.DataFrame, y: np.ndarray,
                           cols: List[str], verbose: bool) -> List[str]:
        """Quick feature importance threshold filter."""
        try:
            from xgboost import XGBClassifier
            quick = XGBClassifier(
                n_estimators=50, max_depth=4, random_state=self.random_state,
                eval_metric="logloss", verbosity=0, n_jobs=-1,
            )
            quick.fit(X, y)
            importances = quick.feature_importances_
            mask = importances >= 0.01
            selected = [c for c, m in zip(cols, mask) if m]
            if len(selected) < 3:
                # Keep at least top 3
                top_idx = np.argsort(importances)[-3:]
                selected = [cols[i] for i in top_idx]
            if verbose and len(selected) < len(cols):
                dropped = len(cols) - len(selected)
                print(f"  → Dropped {dropped} low-importance features")
            return selected
        except Exception:
            return cols

    # ------------------------------------------------------------------
    # Class Imbalance Handling
    # ------------------------------------------------------------------
    def _handle_imbalance(self, X: np.ndarray, y: np.ndarray,
                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE for small datasets, rely on class weights for large."""
        fraud_rate = y.mean()
        if fraud_rate > 0.1:
            if verbose:
                print("  → Class ratio acceptable, no resampling needed")
            return X, y

        if len(X) < 50000:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=self.random_state)
                X_res, y_res = smote.fit_resample(X, y)
                if verbose:
                    print(f"  → SMOTE: {len(X):,} → {len(X_res):,} samples")
                return X_res, y_res
            except ImportError:
                if verbose:
                    print("  → imbalanced-learn not installed, using class weights")
        else:
            if verbose:
                print("  → Large dataset: using class weights instead of SMOTE")

        return X, y

    # ------------------------------------------------------------------
    # Cross-validation metric helper
    # ------------------------------------------------------------------
    def _cv_score(self, model, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute cross-validated ROC-AUC, PR-AUC, and F1."""
        cv = StratifiedKFold(n_splits=3, shuffle=True,
                             random_state=self.random_state)
        try:
            probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            preds = (probs > 0.5).astype(int)
            roc = roc_auc_score(y, probs)
            precision, recall, _ = precision_recall_curve(y, probs)
            pr = auc(recall, precision)
            f1 = f1_score(y, preds, zero_division=0)
            return {"roc_auc": roc, "pr_auc": pr, "f1": f1}
        except Exception:
            return {"roc_auc": 0.0, "pr_auc": 0.0, "f1": 0.0}

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    def _train_xgboost(self, X_full, y_full, X_train, y_train, verbose
                       ) -> Tuple[Any, dict, dict]:
        try:
            from xgboost import XGBClassifier
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as e:
            if verbose:
                print(f"  → XGBoost/Optuna not available: {e}")
            return None, {}, {}

        n_neg = (y_train == 0).sum()
        n_pos = max((y_train == 1).sum(), 1)
        scale_pw = n_neg / n_pos

        tree_method = "gpu_hist" if self.gpu_available else "hist"

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            model = XGBClassifier(
                **params,
                scale_pos_weight=scale_pw,
                tree_method=tree_method,
                eval_metric="logloss",
                verbosity=0,
                random_state=self.random_state,
                n_jobs=-1,
            )
            scores = self._cv_score(model, X_full, y_full)
            return scores["roc_auc"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=False)

        best_params = study.best_params
        best_model = XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pw,
            tree_method=tree_method,
            eval_metric="logloss",
            verbosity=0,
            random_state=self.random_state,
            n_jobs=-1,
        )
        best_model.fit(X_train, y_train)

        scores = self._cv_score(best_model, X_full, y_full)
        return best_model, best_params, scores

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    def _train_lightgbm(self, X_full, y_full, X_train, y_train, verbose
                        ) -> Tuple[Any, dict, dict]:
        try:
            from lightgbm import LGBMClassifier
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as e:
            if verbose:
                print(f"  → LightGBM/Optuna not available: {e}")
            return None, {}, {}

        n_neg = (y_train == 0).sum()
        n_pos = max((y_train == 1).sum(), 1)
        scale_pw = n_neg / n_pos

        device = "gpu" if self.gpu_available else "cpu"

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }
            model = LGBMClassifier(
                **params,
                scale_pos_weight=scale_pw,
                device=device,
                verbose=-1,
                random_state=self.random_state,
                n_jobs=-1,
            )
            scores = self._cv_score(model, X_full, y_full)
            return scores["roc_auc"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=False)

        best_params = study.best_params
        best_model = LGBMClassifier(
            **best_params,
            scale_pos_weight=scale_pw,
            device=device,
            verbose=-1,
            random_state=self.random_state,
            n_jobs=-1,
        )
        best_model.fit(X_train, y_train)

        scores = self._cv_score(best_model, X_full, y_full)
        return best_model, best_params, scores

    # ------------------------------------------------------------------
    # CatBoost
    # ------------------------------------------------------------------
    def _train_catboost(self, X_full, y_full, X_train, y_train, verbose
                        ) -> Tuple[Any, dict, dict]:
        try:
            from catboost import CatBoostClassifier
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as e:
            if verbose:
                print(f"  → CatBoost/Optuna not available: {e}")
            return None, {}, {}

        n_neg = (y_train == 0).sum()
        n_pos = max((y_train == 1).sum(), 1)
        scale_pw = n_neg / n_pos

        task_type = "GPU" if self.gpu_available else "CPU"

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 500),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            }
            model = CatBoostClassifier(
                **params,
                scale_pos_weight=scale_pw,
                task_type=task_type,
                verbose=0,
                random_state=self.random_state,
            )
            scores = self._cv_score(model, X_full, y_full)
            return scores["roc_auc"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=False)

        best_params = study.best_params
        best_model = CatBoostClassifier(
            **best_params,
            scale_pos_weight=scale_pw,
            task_type=task_type,
            verbose=0,
            random_state=self.random_state,
        )
        best_model.fit(X_train, y_train)

        scores = self._cv_score(best_model, X_full, y_full)
        return best_model, best_params, scores

    # ------------------------------------------------------------------
    # Feature Importances
    # ------------------------------------------------------------------
    def _get_feature_importances(self, model, model_name: str,
                                  feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importances from the best model."""
        try:
            importances = model.feature_importances_
            importance_dict = {
                name: round(float(imp), 4)
                for name, imp in zip(feature_names, importances)
            }
            return dict(sorted(importance_dict.items(),
                               key=lambda x: x[1], reverse=True))
        except Exception:
            return {}


# %%
if __name__ == "__main__":
    np.random.seed(42)
    n = 2000
    test_df = pd.DataFrame({
        "f1": np.concatenate([np.random.normal(0, 1, 1900),
                               np.random.normal(3, 0.5, 100)]),
        "f2": np.concatenate([np.random.normal(0, 1, 1900),
                               np.random.normal(-3, 0.5, 100)]),
        "f3": np.random.exponential(2, n),
        "is_fraud": np.array([0]*1900 + [1]*100),
    })

    module = SupervisedModule(n_optuna_trials=5)
    probs, result = module.train_and_predict(
        test_df, ["f1", "f2", "f3"], "is_fraud"
    )
    print(f"\nBest model: {result.best_model_name}")
    print(f"ROC-AUC: {result.best_roc_auc:.4f}")
