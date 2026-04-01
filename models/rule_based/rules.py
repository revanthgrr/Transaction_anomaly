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

# %%
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# %%
warnings.filterwarnings('ignore')

# %%
print('All imports successful!')

# %%
# =============== CHOOSE YOUR DATA SOURCE ===============
DATA_SOURCE = 'custom'  # Options: 'kaggle', 'synthetic', 'custom', 'colab_upload'
CUSTOM_CSV_PATH = '../../data/fraudTest.csv'     # Path to your local fraudTest.csv
# =======================================================


# %%
def generate_synthetic_data(n_users=50, n_transactions=5000):
    """Generate synthetic transaction data with realistic patterns."""
    np.random.seed(42)
    cities = {
        'New York': (40.71, -74.01), 'Boston': (42.36, -71.06),
        'Chicago': (41.88, -87.63), 'Los Angeles': (34.05, -118.24),
        'Miami': (25.76, -80.19), 'Tokyo': (35.68, 139.69),
        'London': (51.51, -0.13), 'Paris': (48.86, 2.35)
    }
    categories = ['grocery', 'gas_transport', 'food_dining', 'shopping_net',
                  'entertainment', 'health_fitness', 'electronics', 'travel']
    us_cities = ['New York', 'Boston', 'Chicago', 'Los Angeles', 'Miami']

    user_profiles = {}
    for uid in range(n_users):
        home = np.random.choice(us_cities)
        user_profiles[uid] = {
            'home_city': home, 'home_lat': cities[home][0], 'home_lon': cities[home][1],
            'avg_amount': np.random.uniform(20, 200),
            'std_amount': np.random.uniform(10, 50),
            'usual_categories': list(np.random.choice(categories[:5], size=3, replace=False)),
            'active_start': np.random.randint(7, 10), 'active_end': np.random.randint(18, 22)
        }

    records = []
    base_time = datetime(2024, 1, 1)
    for i in range(n_transactions):
        uid = np.random.randint(0, n_users)
        p = user_profiles[uid]
        is_fraud = np.random.random() < 0.05

        if is_fraud:
            fraud_type = np.random.choice(['amount', 'location', 'time', 'velocity', 'category', 'geo_impossible'])
            amount = p['avg_amount'] + p['std_amount'] * np.random.uniform(4, 20) if fraud_type == 'amount' else np.random.normal(p['avg_amount'], p['std_amount'])
            if fraud_type in ['location', 'geo_impossible']:
                city = np.random.choice(['Tokyo', 'London', 'Paris'])
                lat, lon = cities[city][0] + np.random.normal(0, 0.1), cities[city][1] + np.random.normal(0, 0.1)
            else:
                lat, lon = p['home_lat'] + np.random.normal(0, 0.5), p['home_lon'] + np.random.normal(0, 0.5)
            hour = np.random.randint(1, 5) if fraud_type == 'time' else np.random.randint(p['active_start'], p['active_end'])
            category = np.random.choice(['electronics', 'travel', 'entertainment']) if fraud_type == 'category' else np.random.choice(p['usual_categories'])
            time_offset = timedelta(minutes=np.random.randint(0, 30)) if fraud_type == 'velocity' else timedelta(hours=np.random.randint(0, 48))
        else:
            amount = max(1, np.random.normal(p['avg_amount'], p['std_amount']))
            lat, lon = p['home_lat'] + np.random.normal(0, 0.5), p['home_lon'] + np.random.normal(0, 0.5)
            hour = np.random.randint(p['active_start'], p['active_end'])
            category = np.random.choice(p['usual_categories'])
            time_offset = timedelta(hours=np.random.randint(1, 72))

        trans_time = base_time + timedelta(days=i // 20) + timedelta(hours=hour, minutes=np.random.randint(0, 60))
        records.append({
            'transaction_id': f'TXN_{i:06d}', 'user_id': uid,
            'trans_date_trans_time': trans_time, 'amt': round(max(1, amount), 2),
            'lat': round(lat, 4), 'long': round(lon, 4),
            'merch_lat': round(lat + np.random.normal(0, 0.05), 4),
            'merch_long': round(lon + np.random.normal(0, 0.05), 4),
            'category': category, 'is_fraud': int(is_fraud)
        })

    return pd.DataFrame(records)


# %%
# ---------- Load data based on chosen source ----------
if DATA_SOURCE == 'kaggle':
    import kagglehub
    print("📥 Downloading fraud-detection dataset from Kaggle...")
    path = kagglehub.dataset_download('kartik2112/fraud-detection')
    
    # Specifically look for 'fraudTest.csv' in the downloaded files
    csv_files = [f for f in os.listdir(path) if 'fraudtest' in f.lower()]
    if not csv_files:
        # Fallback to first available CSV if fraudTest is unexpectedly named
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    selected_file = csv_files[0]
    print(f"✅ Loading: {selected_file}")
    df = pd.read_csv(os.path.join(path, selected_file))

elif DATA_SOURCE == 'synthetic':
    df = generate_synthetic_data()

elif DATA_SOURCE == 'colab_upload':
    from google.colab import files
    print("☁️ Detected Google Colab. Please upload your CSV file now:")
    uploaded = files.upload()
    uploaded_file_name = list(uploaded.keys())[0]
    df = pd.read_csv(uploaded_file_name)
    print(f"✅ Successfully loaded '{uploaded_file_name}'!")

elif DATA_SOURCE == 'custom':
    try:
        df = pd.read_csv(CUSTOM_CSV_PATH)
        print(f"✅ Successfully loaded {CUSTOM_CSV_PATH}")
    except FileNotFoundError:
        print(f"⚠️ Could not find file at '{CUSTOM_CSV_PATH}'.")
        print("💡 If you are in Google Colab, please change DATA_SOURCE to 'colab_upload'")
        raise

# %%
print(f'Dataset shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(df.head())

# %%
# --- Dynamic Column Mapping ---
def map_generic_columns(df):
    """Automatically maps generic columns to standard names for rule engine."""
    mapped_df = df.copy()
    col_mapping = {}
    
    # Identify obvious targets based on contents:
    for col in mapped_df.columns:
        c_low = str(col).lower()
        # Time
        if 'timestamp' not in col_mapping.values():
            if 'time' in c_low or 'date' in c_low:
                col_mapping[col] = 'timestamp'
                continue
            if pd.api.types.is_datetime64_any_dtype(mapped_df[col]):
                col_mapping[col] = 'timestamp'
                continue
                
        # Amount
        if 'amount' not in col_mapping.values():
            if any(k in c_low for k in ['amt', 'amount', 'val', 'price']) and pd.api.types.is_numeric_dtype(mapped_df[col]):
                col_mapping[col] = 'amount'
                continue
                
        # Users
        if 'user_id' not in col_mapping.values():
            if any(k in c_low for k in ['id', 'user', 'num', 'account', 'cc']):
                col_mapping[col] = 'user_id'
                continue
                
        # Loc
        if 'lat' not in col_mapping.values() and 'lat' in c_low:
            col_mapping[col] = 'lat'
        elif 'lon' not in col_mapping.values() and ('lon' in c_low or 'lng' in c_low):
            col_mapping[col] = 'lon'
            
        # Category
        if 'category' not in col_mapping.values() and ('cat' in c_low or 'type' in c_low):
            col_mapping[col] = 'category'
            
    # Fallbacks for numerical amounts
    if 'amount' not in col_mapping.values():
        numeric_cols = mapped_df.select_dtypes(include=[np.number]).columns
        avail = [c for c in numeric_cols if c not in col_mapping]
        if avail: col_mapping[avail[0]] = 'amount'

    # Fallbacks for categorical users
    if 'user_id' not in col_mapping.values():
        print("⚠️ [WARNING] No user_id found. Group-based rules (velocity, temporal) will treat everything as one user.")
        mapped_df['user_id'] = 'GlobalUser'
        
    mapped_df = mapped_df.rename(columns=col_mapping)
    
    if 'timestamp' in mapped_df.columns:
        try:
            mapped_df['timestamp'] = pd.to_datetime(mapped_df['timestamp'])
            mapped_df = mapped_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        except Exception:
            pass
            
    # Ground Truth mapping
    for c in mapped_df.columns:
        if str(c).lower() in ['is_fraud', 'class', 'label', 'fraud', 'is_anomaly', 'target']:
            mapped_df = mapped_df.rename(columns={c: 'is_fraud'})
            break

    print("\n[MAPPER] Dynamically mapped the following columns to standard names:")
    for k,v in col_mapping.items():
        print(f"  {k} -> {v}")
    return mapped_df

# %%
df = map_generic_columns(df)
print(f'\nStandardized mapped columns: {list(df.columns)}')
print(df.head())

# %%
# --- Rules ---
def detect_amount_anomaly(df, z_threshold=3.0):
    if 'amount' not in df.columns or 'user_id' not in df.columns:
        print('⚠️ [SKIP] Amount Anomaly: Required column missing.')
        df_ret = pd.DataFrame(index=df.index); df_ret['flag_amount_anomaly'] = 0
        return df_ret, False
        
    user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std']).reset_index()
    user_stats.columns = ['user_id', 'user_mean', 'user_std']
    # If a user only has one tx, std is NaN or 0 -> force 1 to avoid div-zero
    user_stats['user_std'] = user_stats['user_std'].fillna(1).replace(0, 1)

    result = df.merge(user_stats, on='user_id', how='left')
    result['amount_z_score'] = (result['amount'] - result['user_mean']) / result['user_std']
    result['amount_z_score'] = result['amount_z_score'].fillna(0)
    result['flag_amount_anomaly'] = (result['amount_z_score'].abs() > z_threshold).astype(int)

    flagged = result['flag_amount_anomaly'].sum()
    print(f'🔴 Amount Anomaly: {flagged} flagged ({flagged/len(df)*100:.2f}%)')
    return result[['amount_z_score', 'flag_amount_anomaly']], True

# %%
amount_results, ok_amount = detect_amount_anomaly(df)
for c in amount_results.columns: df[c] = amount_results[c]

# %%
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

# %%
def detect_location_deviation(df, distance_threshold_km=500):
    if 'lat' not in df.columns or 'lon' not in df.columns or 'user_id' not in df.columns:
        print('⚠️ [SKIP] Location Deviation: "lat" or "lon" missing.')
        df_ret = pd.DataFrame(index=df.index); df_ret['flag_location_deviation'] = 0; df_ret['location_distance_km'] = 0
        return df_ret, False

    user_centers = df.groupby('user_id')[['lat', 'lon']].median().reset_index()
    user_centers.columns = ['user_id', 'center_lat', 'center_lon']
    result = df.merge(user_centers, on='user_id', how='left')

    result['location_distance_km'] = result.apply(
        lambda r: haversine(r['lat'], r['lon'], r['center_lat'], r['center_lon']) if pd.notnull(r['lat']) and pd.notnull(r['center_lat']) else 0, axis=1)
    result['flag_location_deviation'] = (result['location_distance_km'] > distance_threshold_km).astype(int)

    flagged = result['flag_location_deviation'].sum()
    print(f'🔴 Location Deviation: {flagged} flagged ({flagged/len(df)*100:.2f}%)')
    return result[['location_distance_km', 'flag_location_deviation']], True

# %%
location_results, ok_loc = detect_location_deviation(df)
for c in location_results.columns: df[c] = location_results[c]

# %%
def detect_temporal_anomaly(df, unusual_pct_threshold=0.05):
    if 'timestamp' not in df.columns or 'user_id' not in df.columns:
        print('⚠️ [SKIP] Temporal Anomaly: "timestamp" missing.')
        df_ret = pd.DataFrame(index=df.index); df_ret['flag_temporal_anomaly'] = 0; df_ret['hour_pct'] = 0; df_ret['hour'] = 0
        return df_ret, False
        
    df_temp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_temp['timestamp']):
         print('⚠️ [SKIP] Temporal Anomaly: "timestamp" cannot be parsed as datetime.')
         df_ret = pd.DataFrame(index=df.index); df_ret['flag_temporal_anomaly'] = 0; df_ret['hour_pct'] = 0; df_ret['hour'] = 0
         return df_ret, False
         
    df_temp['hour'] = df_temp['timestamp'].dt.hour
    user_hour_counts = df_temp.groupby(['user_id', 'hour']).size().reset_index(name='count')
    user_totals = df_temp.groupby('user_id').size().reset_index(name='total')
    user_hour_counts = user_hour_counts.merge(user_totals, on='user_id')
    user_hour_counts['hour_pct'] = user_hour_counts['count'] / user_hour_counts['total']

    result = df_temp.merge(user_hour_counts[['user_id', 'hour', 'hour_pct']], on=['user_id', 'hour'], how='left')
    result['hour_pct'] = result['hour_pct'].fillna(0)
    result['flag_temporal_anomaly'] = (result['hour_pct'] < unusual_pct_threshold).astype(int)

    flagged = result['flag_temporal_anomaly'].sum()
    print(f'🔴 Temporal Anomaly: {flagged} flagged ({flagged/len(df)*100:.2f}%)')
    return result[['hour', 'hour_pct', 'flag_temporal_anomaly']], True

# %%
temporal_results, ok_temp = detect_temporal_anomaly(df)
for c in temporal_results.columns: df[c] = temporal_results[c]

# %%
def detect_velocity(df, window_minutes=60, max_transactions=5):
    if 'timestamp' not in df.columns or 'user_id' not in df.columns:
         print('⚠️ [SKIP] Velocity Check: "timestamp" missing.')
         df_ret = pd.DataFrame(index=df.index); df_ret['flag_velocity'] = 0; df_ret['txn_count_in_window']=0
         return df_ret, False
         
    df_sorted = df.sort_values(['user_id', 'timestamp']).copy()
    if not pd.api.types.is_datetime64_any_dtype(df_sorted['timestamp']):
         print('⚠️ [SKIP] Velocity Check: "timestamp" not valid datetime.')
         df_ret = pd.DataFrame(index=df.index); df_ret['flag_velocity'] = 0; df_ret['txn_count_in_window']=0
         return df_ret, False
         
    flags = np.zeros(len(df_sorted), dtype=int)
    counts = np.zeros(len(df_sorted), dtype=int)
    window = pd.Timedelta(minutes=window_minutes)

    for uid, group in df_sorted.groupby('user_id'):
        idxs = group.index.tolist()
        times = group['timestamp'].values
        for i, idx in enumerate(idxs):
            count = sum(1 for j in range(max(0, i-20), i+1) if (times[i] - times[j]) <= window)
            counts[idx] = count
            if count > max_transactions: flags[idx] = 1

    df_sorted['txn_count_in_window'] = counts
    df_sorted['flag_velocity'] = flags
    flagged = flags.sum()
    print(f'🔴 Velocity Check: {flagged} flagged ({flagged/len(df)*100:.2f}%)')
    return df_sorted[['txn_count_in_window', 'flag_velocity']].loc[df.index], True

# %%
velocity_results, ok_vel = detect_velocity(df)
for c in velocity_results.columns: df[c] = velocity_results[c]

# %%
def detect_category_deviation(df, rare_threshold=0.05):
    if 'category' not in df.columns or 'user_id' not in df.columns:
        print('⚠️ [SKIP] Category Deviation: "category" missing.')
        df_ret = pd.DataFrame(index=df.index); df_ret['flag_category_deviation'] = 0; df_ret['cat_pct'] = 0
        return df_ret, False

    user_cat = df.groupby(['user_id', 'category']).size().reset_index(name='cat_count')
    user_total = df.groupby('user_id').size().reset_index(name='total_count')
    user_cat = user_cat.merge(user_total, on='user_id')
    user_cat['cat_pct'] = user_cat['cat_count'] / user_cat['total_count']

    result = df.merge(user_cat[['user_id', 'category', 'cat_pct']], on=['user_id', 'category'], how='left')
    result['cat_pct'] = result['cat_pct'].fillna(0)
    result['flag_category_deviation'] = (result['cat_pct'] < rare_threshold).astype(int)

    flagged = result['flag_category_deviation'].sum()
    print(f'🔴 Category Deviation: {flagged} flagged ({flagged/len(df)*100:.2f}%)')
    return result[['cat_pct', 'flag_category_deviation']], True

# %%
category_results, ok_cat = detect_category_deviation(df)
for c in category_results.columns: df[c] = category_results[c]

# %%
def detect_geographic_impossibility(df, max_speed_kmh=900):
    if 'lat' not in df.columns or 'lon' not in df.columns or 'timestamp' not in df.columns:
        print('⚠️ [SKIP] Geographic Impossibility: "lat", "lon" or "timestamp" missing.')
        df_ret = pd.DataFrame(index=df.index); df_ret['flag_geo_impossible'] = 0; df_ret['travel_speed_kmh'] = 0
        return df_ret, False

    df_sorted = df.sort_values(['user_id', 'timestamp']).copy()
    if not pd.api.types.is_datetime64_any_dtype(df_sorted['timestamp']):
        print('⚠️ [SKIP] Geographic Impossibility: "timestamp" not valid datetime.')
        df_ret = pd.DataFrame(index=df.index); df_ret['flag_geo_impossible'] = 0; df_ret['travel_speed_kmh'] = 0
        return df_ret, False
        
    speeds = np.zeros(len(df_sorted))
    flags = np.zeros(len(df_sorted), dtype=int)

    for uid, group in df_sorted.groupby('user_id'):
        idxs = group.index.tolist()
        for i in range(1, len(idxs)):
            idx_curr, idx_prev = idxs[i], idxs[i-1]
            try:
                dist = haversine(group.loc[idx_prev, 'lat'], group.loc[idx_prev, 'lon'],
                               group.loc[idx_curr, 'lat'], group.loc[idx_curr, 'lon'])
                time_diff = (group.loc[idx_curr, 'timestamp'] - group.loc[idx_prev, 'timestamp']).total_seconds() / 3600
                if time_diff > 0:
                    speed = dist / time_diff
                    speeds[idx_curr] = speed
                    if speed > max_speed_kmh: flags[idx_curr] = 1
            except:
                pass

    df_sorted['travel_speed_kmh'] = speeds
    df_sorted['flag_geo_impossible'] = flags
    flagged = flags.sum()
    print(f'🔴 Geographic Impossibility: {flagged} flagged ({flagged/len(df)*100:.2f}%)')
    return df_sorted[['travel_speed_kmh', 'flag_geo_impossible']].loc[df.index], True

# %%
geo_results, ok_geo = detect_geographic_impossibility(df)
for c in geo_results.columns: df[c] = geo_results[c]

# %%
# --- Dynamic Evaluation ---
rules_run = sum([ok_amount, ok_loc, ok_temp, ok_vel, ok_cat, ok_geo])
print(f"\n[EVAL] {rules_run}/6 rule checks successfully completed on this dataset.")

# %%
flag_cols = ['flag_amount_anomaly', 'flag_location_deviation', 'flag_temporal_anomaly',
             'flag_velocity', 'flag_category_deviation', 'flag_geo_impossible']

# %%
existing_flags = [c for c in flag_cols if c in df.columns]
df['fraud_score'] = df[existing_flags].sum(axis=1)

# %%
# Dynamic Target Threshold: If < 3 rules ran, require 1 vote. If 3+ rules ran, require 2+ votes.
dynamic_threshold = 2 if rules_run >= 3 else 1
if rules_run == 0:
    print("❌ No rules were able to run. Cannot predict anomalies.")
    df['predicted_fraud'] = 0
else:
    df['predicted_fraud'] = (df['fraud_score'] >= dynamic_threshold).astype(int)

    print(f'\n=== Fraud Score Distribution (Threshold: >={dynamic_threshold}) ===')
    print(df['fraud_score'].value_counts().sort_index())
    print(f'\nTotal predicted fraud: {df["predicted_fraud"].sum()} / {len(df)} ({df["predicted_fraud"].mean()*100:.2f}%)')

    if 'is_fraud' in df.columns:
        from sklearn.metrics import classification_report, confusion_matrix

        print('\n=== Classification Report ===')
        try:
            print(classification_report(df['is_fraud'], df['predicted_fraud'], target_names=['Legit', 'Fraud']))
        except:
             print(classification_report(df['is_fraud'], df['predicted_fraud']))

        cm = confusion_matrix(df['is_fraud'], df['predicted_fraud'])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Legit', 'Fraud'],
                    yticklabels=['Legit', 'Fraud'], ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix (Rules Executed: {rules_run}/6)')
        plt.tight_layout()
        plt.show()
    else:
        print('No ground-truth labels found. Showing flag distribution only.')
