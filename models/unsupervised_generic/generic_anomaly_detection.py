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
# # Generic Unsupervised Anomaly Detection
#
# This script is designed to process **any** CSV file containing numeric data
# and perform highly automated unsupervised anomaly detection. It requires
# zero manual feature engineering. It creates an ensemble of three
# distinct machine learning algorithms:
#
# 1. **Isolation Forest**: Isolates anomalies in a high-dimensional feature space.
# 2. **DBSCAN**: Density-based clustering, treating low-density points as anomalies.
# 3. **Local Outlier Factor (LOF)**: Detects local density deviations relative to neighbors.
#
# Finally, it aggregates their predictions via a voting mechanism, creating a 
# robust, highly-generalizable anomaly detection pipeline.

# %% [markdown]
# ## 1. Initialization and Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# Set FILE_PATH to analyze an arbitrary CSV. 
# If empty, the script will attempt to find a local CSV for demonstration.

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
# Empty by default so it auto-detects dynamically added Colab CSVs
FILE_PATH = os.environ.get("DATA_FILE", "")

# %% [markdown]
# ## 2. Data Loading

# %%
def load_generic_data(file_path: str) -> pd.DataFrame:
    """Load arbitrary CSV data."""
    if file_path and os.path.isfile(file_path):
        print(f'[LOAD] Loading from {file_path}')
        return pd.read_csv(file_path)
    
    # Search the current directory (Colab root) and data directory directly
    search_dirs = [
        os.getcwd(),
        PROJECT_ROOT,
        os.path.join(PROJECT_ROOT, 'data')
    ]
    
    # Force add Colab default temporary paths if running in Colab (e.g., from mounted GDrive)
    if os.path.exists('/content'):
        search_dirs.extend(['/content', '/content/data'])
        
    search_dirs = list(set(search_dirs))
    
    csv_files = []
    for d in search_dirs:
        if os.path.isdir(d):
            # Ignore Colab's default sample_data directory
            csv_files.extend([os.path.join(d, f) for f in os.listdir(d) 
                              if f.lower().endswith('.csv') and 'sample_data' not in d])
                              
    # Remove duplicate paths just in case
    csv_files = list(set(csv_files))
            
    if len(csv_files) == 0:
        found_files = []
        for d in search_dirs:
            if os.path.exists(d):
                found_files.extend([f"{d}/{x}" for x in os.listdir(d) if 'sample_data' not in d])
                
        raise FileNotFoundError(
            f"No CSV file found!\n"
            f"Searched exact folders: {search_dirs}\n"
            f"All physical files currently visible to Python in those folders: {found_files}\n"
            "If your file is uploading, wait for it to finish. If your file is a .zip, please extract the .csv first."
        )
    
    # Default to the smallest one if many for quick testing
    chosen_csv = sorted(csv_files, key=os.path.getsize)[0]
    print(f'[LOAD] FILE_PATH empty. Falling back to generic CSV: {chosen_csv}')
    return pd.read_csv(chosen_csv)

# %%
try:
    df_raw = load_generic_data(FILE_PATH)
    print(f'[INFO] Original dataset shape: {df_raw.shape}')
except Exception as e:
    print(f"[ERROR] Could not load data: {e}")
    df_raw = pd.DataFrame() # Fallback safely

if not df_raw.empty:
    print(df_raw.head())

# %% [markdown]
# ## 3. Automatic Generic Preprocessing
#
# Finds only numeric columns, applies Missing Value imputation, and standardizes
# the scales using RobustScaler (which is resistant to outliers).

# %%
def preprocess_generic(df: pd.DataFrame):
    """
    Automatically isolates numeric features and standardises them.
    Drops rows/columns strictly containing all nulls.
    """
    # 1. Filter numerics natively
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Drop completely empty columns
    df_numeric = df_numeric.dropna(axis=1, how='all')
    print(f'[PREP] Identified {df_numeric.shape[1]} numeric features out of {df.shape[1]} original features.')
    
    if df_numeric.shape[1] == 0:
         raise ValueError("No numeric data found to process.")
    
    # 2. Imputation (Replace NaNs with median of the column)
    imputer = SimpleImputer(strategy='median')
    data_imputed = imputer.fit_transform(df_numeric)
    
    # 3. Scaling (Robust scaling resists extreme anomalies, maintaining standard shape)
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data_imputed)
    
    return data_scaled, df_numeric.columns

# %%
if not df_raw.empty:
    X_scaled, numeric_cols = preprocess_generic(df_raw)
    print(f"[OK] Preprocessing complete. Scaled Matrix Shape: {X_scaled.shape}")

# %% [markdown]
# ## 4. Machine Learning Ensemble Preparation
# To improve distance-based calculations (DBSCAN/LOF) in datasets that might have
# highly-dimensional or correlated features, we use Principal Component Analysis (PCA)
# as an optional feature transformation prior to those specific algorithms.

# %%
def calculate_dynamic_eps(X_pca, k=5, percentile=95):
    """
    DBSCAN depends heavily on `eps` (the proximity radius). Instead of hardcoding,
    we dynamically calculate an optimal distance by viewing the k-th nearest neighbor 
    distances of all points. We choose the 95th percentile point's distance as a generic 
    cutoff for "density" so the sparsest 5% fall outside clusters.
    """
    if X_pca.shape[0] < k: return 0.5 # Safety fallback
    
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(X_pca)
    distances, _ = nbrs.kneighbors(X_pca)
    
    # Look at distance to the k-th neighbor
    k_distances = distances[:, k-1]
    
    # The 'knee' typically lies around the 90-95th percentile
    dynamic_eps = np.percentile(k_distances, percentile)
    return max(dynamic_eps, 0.01) # Avoid 0

# %% [markdown]
# ## 5. Training Unsupervised Models
#
# Training the 3 distinct anomaly models:
# - **Model 1:** `IsolationForest`
# - **Model 2:** `DBSCAN`
# - **Model 3:** `LocalOutlierFactor`

# %%
def train_ensemble(X_scaled):
    """
    Train 3 Unsupervised Anomaly Detection Algorithms.
    Returns array 1s (Anomaly) and 0s (Normal) for each.
    """
    n_samples = X_scaled.shape[0]
    
    # Determine subsampling to prevent OOM errors on massive CSVs (LOF/DBSCAN scale poorly >100k)
    # DBSCAN & LOF calculate distance matrices.
    sample_limit = 50000 
    
    # --- MODEL 1: Isolation Forest ---
    print("[TRAIN][1/3] Training Isolation Forest...")
    ifo = IsolationForest(
        n_estimators=100, 
        contamination='auto', 
        random_state=42, 
        n_jobs=-1
    )
    # Returns 1 (normal) and -1 (anomaly), map to 1 (anomaly), 0 (normal)
    ifo_preds = ifo.fit_predict(X_scaled)
    ifo_anomalies = np.where(ifo_preds == -1, 1, 0)
    
    # --- PCA Dimensionality Reduction for Distance Methods ---
    pca_components = min(10, X_scaled.shape[1]) 
    print(f"[TRAIN] Dimensionality Reduction: reducing distance space to {pca_components} components.")
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # We will limit rows evaluated for distance methods if huge dataset, but for simplicity
    # of assigning back to original DataFrame, we process all if under limit.
    if n_samples > sample_limit:
        print(f"[WARNING] Large dataset ({n_samples} rows). Skipping DBSCAN & LOF (distance algorithms).")
        # Fill with zeros so I-Forest dictates if row is too large
        dbscan_anomalies = np.zeros(n_samples)
        lof_anomalies = np.zeros(n_samples)
    else:
        # --- MODEL 2: DBSCAN ---
        print("[TRAIN][2/3] Training DBSCAN...")
        eps_opt = calculate_dynamic_eps(X_pca, k=5, percentile=95)
        print(f"   Calculated optimal epsilon parameter = {eps_opt:.4f}")
        
        dbscan = DBSCAN(eps=eps_opt, min_samples=5, n_jobs=-1)
        # Returns cluster label (>= 0) or -1 (anomaly)
        dbscan_preds = dbscan.fit_predict(X_pca)
        dbscan_anomalies = np.where(dbscan_preds == -1, 1, 0)

        # --- MODEL 3: Local Outlier Factor ---
        print("[TRAIN][3/3] Training Local Outlier Factor...")
        lof = LocalOutlierFactor(n_neighbors=20, novelty=False, n_jobs=-1)
        # Returns 1 (normal) and -1 (anomaly)
        lof_preds = lof.fit_predict(X_pca)
        lof_anomalies = np.where(lof_preds == -1, 1, 0)

    print("[DONE] Ensemble models finished processing.")
    return ifo_anomalies, dbscan_anomalies, lof_anomalies

# %%
if not df_raw.empty:
    ifo_anomalies, dbscan_anomalies, lof_anomalies = train_ensemble(X_scaled)

# %% [markdown]
# ## 6. Aggregating the Ensemble (Voting System)
#
# We tally the results: if at least 2 out of the 3 algorithms agree a point is an 
# anomaly, we classify it as a **High Confidence Anomaly**.

# %%
def build_ensemble_results(df: pd.DataFrame, ifo, dbscan, lof):
    """Aggregate predictions into the main DataFrame."""
    res_df = df.copy()
    
    # Assign predictions
    res_df['anomaly_model_iforest'] = ifo
    res_df['anomaly_model_dbscan'] = dbscan
    res_df['anomaly_model_lof'] = lof
    
    # Tally votes (0 to 3)
    res_df['anomaly_votes'] = ifo + dbscan + lof
    
    # High Confidence Flag (>= 2 votes)
    # Note: If dataset > 50k, only I-Forest ran, so I-Forest's 1 vote = anomaly
    vote_threshold = 2 if len(df) <= 50000 else 1 
    
    res_df['is_anomaly_ensemble'] = (res_df['anomaly_votes'] >= vote_threshold).astype(int)
    
    return res_df

# %%
if not df_raw.empty:
    df_results = build_ensemble_results(df_raw, ifo_anomalies, dbscan_anomalies, lof_anomalies)

    anomaly_cnt = df_results['is_anomaly_ensemble'].sum()
    print(f"\n[RESULTS] Identified {anomaly_cnt} TOTAL outliers out of {len(df_results)} rows ({anomaly_cnt/len(df_results)*100:.2f}%)")
    
    print("\nVote Distribution:")
    print(df_results['anomaly_votes'].value_counts().sort_index(ascending=False))

# %% [markdown]
# ## 7. 2D PCA Visualisation
#
# No matter how many numeric features existed originally, we can project the data
# down to 2 dimensions for visual inspection of the anomalies.

# %%
def plot_anomalies_pca(X_scaled, y_ensemble):
    """Projects high-d data to 2D for generic visualization of the ensemble's choice."""
    if len(np.unique(y_ensemble)) == 1:
        print("[PLOT] No mixed classes found. Skipping plot.")
        return
        
    # Reduce purely to 2 components for the visual plot
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    
    # 0 = Normal, 1 = Anomaly
    normals = X_2d[y_ensemble == 0]
    anomalies = X_2d[y_ensemble == 1]
    
    plt.scatter(normals[:,0], normals[:,1], c='#3498db', alpha=0.5, label='Normal (0)', edgecolors='w', s=30)
    plt.scatter(anomalies[:,0], anomalies[:,1], c='#e74c3c', alpha=0.9, label='Anomaly (1)', edgecolors='k', s=50)
    
    plt.title('2D PCA Projection of Generic High-Dimensional Anomalies')
    plt.xlabel(f'Principal Component 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Principal Component 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend()
    
    _out_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(_out_dir, exist_ok=True)
    plt.savefig(os.path.join(_out_dir, 'generic_anomaly_plot.png'), dpi=150)
    plt.close()
    print("[PLOT] Saved generic_anomaly_plot.png")

# %%
if not df_raw.empty:
    plot_anomalies_pca(X_scaled, df_results['is_anomaly_ensemble'].values)

# %% [markdown]
# ## 8. Save Final CSV Files

# %%
def save_generic_results(df: pd.DataFrame):
    """Outputs the results to CSVs matching the input format."""
    out_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    
    full_path = os.path.join(out_dir, "generic_anomalies_full.csv")
    flagged_path = os.path.join(out_dir, "generic_anomalies_flagged.csv")
    
    df.to_csv(full_path, index=False)
    
    flagged = df[df['is_anomaly_ensemble'] == 1].sort_values(by="anomaly_votes", ascending=False)
    flagged.to_csv(flagged_path, index=False)
    
    print(f"[SAVE] Exported full data back to -> {full_path}")
    print(f"[SAVE] Exported the {len(flagged)} flagged anomalies to -> {flagged_path}")

# %%
if not df_raw.empty:
    save_generic_results(df_results)

# %% [markdown]
# ## 9. Evaluation (If Ground Truth Exists)
# Check if the original dataset contained a target label (e.g., 'is_fraud', 'Class', 'label') 
# and compare it against our unsupervised ensemble predictions.

# %%
def evaluate_performance(df: pd.DataFrame):
    """Calculates accuracy and displays classification report if labels exist."""
    potential_labels = ['is_fraud', 'class', 'label', 'fraud', 'is_anomaly', 'target']
    
    # Find the actual label column (case-insensitive search)
    label_col = None
    for col in df.columns:
        if col.lower() in potential_labels:
            label_col = col
            break
            
    if label_col is None:
        print("\n[EVAL] No ground-truth label found in original target. Skipping accuracy check.")
        return
        
    print(f"\n[EVAL] Found ground-truth label column: '{label_col}'. Evaluating unsupervised accuracy...")
    
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    y_true = df[label_col].values
    y_pred = df['is_anomaly_ensemble'].values
    
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"--- Ensemble Model Accuracy: {acc*100:.2f}% ---\n")
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal (0)', 'Anomaly (1)'], zero_division=0))
    
    # Display confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"[{cm[0][0]:>6} {cm[0][1]:>6}]")
    print(f"[{cm[1][0]:>6} {cm[1][1]:>6}]")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal (0)', 'Anomaly (1)'], 
                yticklabels=['Normal (0)', 'Anomaly (1)'])
    plt.title(f'Confusion Matrix (Accuracy: {acc*100:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    _out_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(_out_dir, exist_ok=True)
    plt.savefig(os.path.join(_out_dir, 'confusion_matrix.png'), dpi=150)
    print("[PLOT] Saved confusion_matrix.png")
    plt.show()

# %%
if not df_raw.empty:
    evaluate_performance(df_results)

# %%
if __name__ == '__main__':
    print('\n[DONE] Generic Anomaly Detection Pipeline completed!')
