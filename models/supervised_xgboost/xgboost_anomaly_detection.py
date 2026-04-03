# -*- coding: utf-8 -*-
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
# # XGBoost-Based Transaction Anomaly Detection
#
# This notebook loads a CSV file of transactions, engineers features from the
# raw data, trains an XGBoost model, and identifies anomalous/fraudulent
# transactions. It complements the rule-based algorithms in `fi.py` by using
# a machine-learning approach for higher accuracy.
#
# **Usage:**
# 1. Set `FILE_PATH` in the Configuration cell to your CSV file path.
# 2. Run all cells in order.
#
# **Expected CSV columns (auto-mapped if present):**
# | Column | Maps to |
# |---|---|
# | `trans_date_trans_time` / `timestamp` | transaction datetime |
# | `amt` / `amount` | transaction amount |
# | `cc_num` / `user_id` / `customer_id` | user identifier |
# | `lat`, `long` / `lon` | transaction latitude/longitude |
# | `merch_lat`, `merch_long` | merchant latitude/longitude |
# | `category` | merchant category |
# | `is_fraud` *(optional)* | ground-truth label (1=fraud) |

# %% [markdown]
# ## 0. Imports & Setup

# %%
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Configuration

# %%
def get_project_root():
    dir_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    current_dir = dir_path
    while True:
        if os.path.isdir(os.path.join(current_dir, 'data')):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir
        
    # Fallback to current working directory (e.g. /content in Colab)
    return dir_path

PROJECT_ROOT = get_project_root()
# Set FILE_PATH to analyze an arbitrary CSV. 
# If empty, the script will attempt to find a local CSV for demonstration.
FILE_PATH = os.environ.get("DATA_FILE", "")

# %% [markdown]
# ## 2. Load Data

# %%
def load_data(file_path: str) -> pd.DataFrame:
    """Load and return transaction data from a CSV file or Kaggle."""
    if file_path and os.path.isfile(file_path):
        print(f'[LOAD] Loading data from: {file_path}')
        return pd.read_csv(file_path)
    
    # Search the current directory (Colab root) and data directory directly
    search_dirs = [
        os.getcwd(),
        PROJECT_ROOT,
        os.path.join(PROJECT_ROOT, 'data')
    ]
    
    # Force add Colab default temporary paths if running in Colab
    if os.path.exists('/content'):
        search_dirs.extend(['/content', '/content/data'])
        
    search_dirs = list(set(search_dirs))
    
    csv_files = []
    for d in search_dirs:
        if os.path.isdir(d):
            # Ignore Colab's default sample_data
            csv_files.extend([os.path.join(d, f) for f in os.listdir(d) 
                              if f.lower().endswith('.csv') and 'sample_data' not in d])
    
    csv_files = list(set(csv_files))
    
    if len(csv_files) > 0:
        # Default to the largest one for XGBoost (usually the full dataset)
        chosen_csv = sorted(csv_files, key=os.path.getsize, reverse=True)[0]
        print(f'[LOAD] FILE_PATH empty. Falling back to CSV: {chosen_csv}')
        return pd.read_csv(chosen_csv)
        
    print('[LOAD] No local CSV found -- downloading Kaggle dataset ...')
    import kagglehub
    path = kagglehub.dataset_download('kartik2112/fraud-detection')
    csv_files = [f for f in os.listdir(path) if f.lower().endswith('.csv')]
    print('   Files found:', csv_files)
    df = pd.read_csv(os.path.join(path, csv_files[0]))

    print(f'   Shape : {df.shape}')
    return df

# %%
df_raw = load_data(FILE_PATH)
df_raw.head()

# %% [markdown]
# ## 3. Standardise Column Names

# %%
def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common column variants to a consistent set of names."""
    col_map = {
        'trans_date_trans_time': 'timestamp',
        'amt': 'amount',
        'long': 'lon',
        'merch_long': 'merch_lon',
        'cc_num': 'user_id',
    }
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'trans_date_trans_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['trans_date_trans_time'])

    # Ensure user_id exists
    if 'user_id' not in df.columns:
        for c in ['customer_id', 'card_id']:
            if c in df.columns:
                df['user_id'] = df[c]
                break

    # Sort chronologically per user
    if 'user_id' in df.columns and 'timestamp' in df.columns:
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    return df

# %%
df = standardise_columns(df_raw)
print(f'[OK] Data loaded: {len(df)} transactions, {df["user_id"].nunique()} users')
df.head()

# %% [markdown]
# ## 4. Feature Engineering
#
# Features created:
# | Feature | Description |
# |---|---|
# | `amount_z_score` | Standard deviations from user mean |
# | `hour` | Hour of transaction (0–23) |
# | `day_of_week` | Day of the week (0=Mon, 6=Sun) |
# | `is_weekend` | 1 if Saturday/Sunday |
# | `is_night` | 1 if hour in [0, 5] |
# | `location_distance_km` | km from user's median location |
# | `merch_distance_km` | km between transaction loc and merchant loc |
# | `time_since_last_txn` | Seconds since previous transaction by same user |
# | `txn_count_1h` | Rolling count of transactions in last 60 min |
# | `amount_to_median` | Ratio of amount to user's median amount |
# | `category_encoded` | Label-encoded merchant category |

# %%
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * asin(sqrt(a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from raw transaction data."""
    feat = df.copy()

    # --- Temporal features ---
    if 'timestamp' in feat.columns:
        feat['hour'] = feat['timestamp'].dt.hour
        feat['day_of_week'] = feat['timestamp'].dt.dayofweek
        feat['is_weekend'] = feat['day_of_week'].isin([5, 6]).astype(int)
        feat['is_night'] = feat['hour'].between(0, 5).astype(int)

        # Time since last transaction (per user)
        if 'user_id' in feat.columns:
            feat['time_since_last_txn'] = (
                feat.groupby('user_id')['timestamp']
                    .diff()
                    .dt.total_seconds()
                    .fillna(-1)
            )
    else:
        feat['hour'] = 0
        feat['day_of_week'] = 0
        feat['is_weekend'] = 0
        feat['is_night'] = 0
        feat['time_since_last_txn'] = -1

    # --- Amount features ---
    if 'amount' in feat.columns and 'user_id' in feat.columns:
        user_stats = (
            feat.groupby('user_id')['amount']
                .agg(['mean', 'std', 'median'])
                .reset_index()
        )
        user_stats.columns = ['user_id', 'user_mean', 'user_std', 'user_median']
        user_stats['user_std'] = user_stats['user_std'].fillna(1).replace(0, 1)
        feat = feat.merge(user_stats, on='user_id', how='left')
        feat['amount_z_score'] = (feat['amount'] - feat['user_mean']) / feat['user_std']
        feat['amount_to_median'] = feat['amount'] / feat['user_median'].replace(0, 1)
        feat.drop(columns=['user_mean', 'user_std', 'user_median'], inplace=True)
    else:
        feat['amount_z_score'] = 0
        feat['amount_to_median'] = 1

    # --- Location features ---
    if 'lat' in feat.columns and 'lon' in feat.columns and 'user_id' in feat.columns:
        user_centers = (
            feat.groupby('user_id')[['lat', 'lon']]
                .median()
                .reset_index()
                .rename(columns={'lat': 'center_lat', 'lon': 'center_lon'})
        )
        feat = feat.merge(user_centers, on='user_id', how='left')
        feat['location_distance_km'] = feat.apply(
            lambda r: haversine(r['lat'], r['lon'], r['center_lat'], r['center_lon']),
            axis=1,
        )
        feat.drop(columns=['center_lat', 'center_lon'], inplace=True)
    else:
        feat['location_distance_km'] = 0

    if all(c in feat.columns for c in ['lat', 'lon', 'merch_lat', 'merch_lon']):
        feat['merch_distance_km'] = feat.apply(
            lambda r: haversine(r['lat'], r['lon'], r['merch_lat'], r['merch_lon']),
            axis=1,
        )
    else:
        feat['merch_distance_km'] = 0

    # --- Velocity (count of transactions in last 60 min per user) ---
    if 'timestamp' in feat.columns and 'user_id' in feat.columns:
        def _count_in_window(ts_series, window_seconds=3600):
            """Count how many prior transactions fall within window_seconds."""
            ts = ts_series.values.astype('datetime64[s]').astype(np.int64)
            counts = np.ones(len(ts), dtype=int)
            for i in range(1, len(ts)):
                c = 1
                for j in range(i - 1, max(-1, i - 50), -1):
                    if (ts[i] - ts[j]) <= window_seconds:
                        c += 1
                    else:
                        break
                counts[i] = c
            return pd.Series(counts, index=ts_series.index)

        feat['txn_count_1h'] = (
            feat.groupby('user_id')['timestamp']
                .transform(_count_in_window)
        )
    else:
        feat['txn_count_1h'] = 1

    # --- Category encoding ---
    if 'category' in feat.columns:
        feat['category_encoded'] = feat['category'].astype('category').cat.codes
    else:
        feat['category_encoded'] = 0

    return feat

# %%
print('[FEAT] Engineering features ...')
df = engineer_features(df)
feature_cols_available = [c for c in [
    'amount', 'amount_z_score', 'amount_to_median',
    'hour', 'day_of_week', 'is_weekend', 'is_night',
    'location_distance_km', 'merch_distance_km',
    'time_since_last_txn', 'txn_count_1h', 'category_encoded',
] if c in df.columns]
print(f'   Features available: {feature_cols_available}')
df[feature_cols_available].describe()

# %% [markdown]
# ## 5. Train XGBoost Model
#
# - **Supervised** (if `is_fraud` column exists): uses ground-truth labels with a 75/25 train/test split.
# - **Semi-supervised** (fallback): derives labels from rule-based flags (`fraud_score ≥ 2`).
# - **Unsupervised** (last resort): labels top 5% of `amount_z_score` as anomalies.

# %%
FEATURE_COLS = [
    'amount', 'amount_z_score', 'amount_to_median',
    'hour', 'day_of_week', 'is_weekend', 'is_night',
    'location_distance_km', 'merch_distance_km',
    'time_since_last_txn', 'txn_count_1h',
    'category_encoded',
]


def train_xgboost(df: pd.DataFrame):
    """
    Train an XGBoost classifier on the dataset.

    If `is_fraud` column exists → supervised learning with train/test split.
    Otherwise → semi-supervised: trains on synthetic labels derived from
    the rule-based flags (fraud_score ≥ 2).
    """
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score,
    )

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].fillna(0)

    # Determine labels
    if 'is_fraud' in df.columns:
        y = df['is_fraud'].astype(int)
        label_source = 'ground-truth (is_fraud)'
    else:
        flag_cols = [
            'flag_amount_anomaly', 'flag_location_deviation',
            'flag_temporal_anomaly', 'flag_velocity',
            'flag_category_deviation', 'flag_geo_impossible',
        ]
        existing_flags = [c for c in flag_cols if c in df.columns]
        if existing_flags:
            y = (df[existing_flags].sum(axis=1) >= 2).astype(int)
            label_source = 'rule-based composite (fraud_score ≥ 2)'
        else:
            threshold = df['amount_z_score'].quantile(0.95)
            y = (df['amount_z_score'] >= threshold).astype(int)
            label_source = 'unsupervised (top-5% amount z-score)'

    print(f'\n[LABEL] Label source: {label_source}')
    print(f'   Fraud/Anomaly count: {y.sum()} / {len(y)} ({y.mean()*100:.2f}%)')

    # Handle class imbalance via scale_pos_weight
    n_neg = (y == 0).sum()
    n_pos = max((y == 1).sum(), 1)
    scale_pos_weight = n_neg / n_pos

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print('\n=== XGBoost Classification Report (Test Set) ===')
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud/Anomaly']))

    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f'ROC-AUC: {auc:.4f}')
    except ValueError:
        auc = None
        print('ROC-AUC: could not compute (single class in test set)')

    # Full-dataset predictions
    df['xgb_fraud_prob'] = model.predict_proba(X)[:, 1]
    df['xgb_predicted_fraud'] = model.predict(X)

    return model, df, available_features, {
        'y_test': y_test, 'y_pred': y_pred, 'y_prob': y_prob, 'auc': auc,
    }

# %%
print('[TRAIN] Training XGBoost model ...')
model, df, feature_names, metrics = train_xgboost(df)

# %% [markdown]
# ## 6. Visualisations
#
# Plots generated:
# 1. **Confusion Matrix** — actual vs predicted labels
# 2. **ROC Curve** — model discrimination ability
# 3. **Feature Importance** — which features drive predictions
# 4. **Fraud Probability Distribution** — score spread for legit vs fraud
# 5. **Anomaly Rate by Hour** — time-of-day fraud patterns
# 6. **Top 20 Most Suspicious Transactions**

# %%
def plot_results(model, df, feature_names, metrics):
    """Generate evaluation plots."""
    from sklearn.metrics import confusion_matrix, roc_curve

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('XGBoost Anomaly Detection Results', fontsize=18, fontweight='bold')

    # --- 1. Confusion Matrix ---
    ax1 = fig.add_subplot(2, 3, 1)
    cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'], ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix')

    # --- 2. ROC Curve ---
    ax2 = fig.add_subplot(2, 3, 2)
    if metrics['auc'] is not None:
        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_prob'])
        ax2.plot(fpr, tpr, color='#e74c3c', lw=2,
                 label=f'AUC = {metrics["auc"]:.4f}')
        ax2.plot([0, 1], [0, 1], 'k--', lw=1)
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend(loc='lower right')
    else:
        ax2.text(0.5, 0.5, 'ROC not available', ha='center', va='center')
        ax2.set_title('ROC Curve')

    # --- 3. Feature Importance ---
    ax3 = fig.add_subplot(2, 3, 3)
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    ax3.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx],
             color='#3498db')
    ax3.set_title('Feature Importance')
    ax3.set_xlabel('Importance')

    # --- 4. Fraud Probability Distribution ---
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(df.loc[df['xgb_predicted_fraud'] == 0, 'xgb_fraud_prob'],
             bins=50, alpha=0.7, label='Legit', color='#2ecc71')
    ax4.hist(df.loc[df['xgb_predicted_fraud'] == 1, 'xgb_fraud_prob'],
             bins=50, alpha=0.7, label='Fraud', color='#e74c3c')
    ax4.set_xlabel('Fraud Probability')
    ax4.set_ylabel('Count')
    ax4.set_title('Probability Distribution')
    ax4.legend()

    # --- 5. Anomalies by Hour ---
    ax5 = fig.add_subplot(2, 3, 5)
    if 'hour' in df.columns:
        hourly = df.groupby('hour')['xgb_predicted_fraud'].mean() * 100
        ax5.bar(hourly.index, hourly.values, color='#9b59b6')
        ax5.set_xlabel('Hour of Day')
        ax5.set_ylabel('Anomaly Rate (%)')
        ax5.set_title('Anomaly Rate by Hour')

    # --- 6. Top Anomalous Transactions ---
    ax6 = fig.add_subplot(2, 3, 6)
    top = df.nlargest(20, 'xgb_fraud_prob')
    ax6.barh(range(len(top)), top['xgb_fraud_prob'].values, color='#e74c3c')
    ax6.set_yticks(range(len(top)))
    ax6.set_yticklabels([f'Txn {i}' for i in range(len(top))], fontsize=7)
    ax6.set_xlabel('Fraud Probability')
    ax6.set_title('Top 20 Most Suspicious')
    ax6.invert_yaxis()

    plt.tight_layout()
    _out_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(_out_dir, exist_ok=True)
    plt.savefig(os.path.join(_out_dir, 'xgboost_results.png'), dpi=150)
    plt.show()
    print('\n[PLOT] Results plot saved as xgboost_results.png')

# %%
plot_results(model, df, feature_names, metrics)

# %% [markdown]
# ## 7. Save Results

# %%
def save_results(df: pd.DataFrame):
    """Save flagged transactions to CSV."""
    out_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)

    # Full results
    out_full = os.path.join(out_dir, 'xgboost_full_results.csv')
    df.to_csv(out_full, index=False)
    print(f'[SAVE] Full results saved to: {out_full}')

    # Flagged-only
    flagged = df[df['xgb_predicted_fraud'] == 1].sort_values(
        'xgb_fraud_prob', ascending=False,
    )
    out_flagged = os.path.join(out_dir, 'xgboost_flagged_transactions.csv')
    flagged.to_csv(out_flagged, index=False)
    print(f'[FLAG] Flagged transactions ({len(flagged)}) saved to: {out_flagged}')

    # Summary
    print(f'\n=== Summary ===')
    print(f'Total transactions analysed : {len(df)}')
    print(f'Flagged as anomalous        : {len(flagged)} ({len(flagged)/len(df)*100:.2f}%)')
    print(f'Average fraud probability   : {df["xgb_fraud_prob"].mean():.4f}')
    print(f'Max fraud probability       : {df["xgb_fraud_prob"].max():.4f}')

# %%
save_results(df)

# %% [markdown]
# ## 8. Top Suspicious Transactions

# %%
display_cols = ['user_id', 'timestamp', 'amount', 'category',
                'xgb_fraud_prob', 'xgb_predicted_fraud']
display_cols = [c for c in display_cols if c in df.columns]
top15 = df.nlargest(15, 'xgb_fraud_prob')[display_cols]
print('[ALERT] Top 15 Most Suspicious Transactions:')
top15

# %% [markdown]
# ---
# *Run as a script:* `python xgboost_anomaly_detection.py`

# %%
if __name__ == '__main__':
    print('[DONE] Complete!')
