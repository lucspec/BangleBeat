import glob
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# =============================================================================
# 1. LOAD & PREPARE
# =============================================================================
DB_SEARCH_PATH = "./data/Gadgetbridge*.db"
CSV_SEARCH_PATH = "./data/loglog*.csv"

def to_datetime_auto(series):
    if series.empty: return series
    if series.median() > 1e11: return pd.to_datetime(series, unit='ms')
    return pd.to_datetime(series, unit='s')

print("📂 Loading Data...")
# Load Ref
ref_dfs = []
for db in glob.glob(DB_SEARCH_PATH):
    try:
        conn = sqlite3.connect(db)
        devs = pd.read_sql("SELECT _id, TYPE_NAME FROM DEVICE WHERE TYPE_NAME LIKE '%GARMIN%' OR TYPE_NAME LIKE '%POLAR%'", conn)
        for _, d in devs.iterrows():
            tbl = 'GARMIN_ACTIVITY_SAMPLE' if 'GARMIN' in d['TYPE_NAME'] else 'POLAR_H10_ACTIVITY_SAMPLE'
            if conn.execute(f"SELECT name FROM sqlite_master WHERE name='{tbl}'").fetchone():
                df = pd.read_sql(f"SELECT TIMESTAMP, HEART_RATE FROM {tbl} WHERE DEVICE_ID={d['_id']} AND HEART_RATE>0", conn)
                if not df.empty:
                    df['datetime'] = to_datetime_auto(df['TIMESTAMP'])
                    ref_dfs.append(df)
        conn.close()
    except: pass

# Load Bangle
bangle_dfs = []
for f in glob.glob(CSV_SEARCH_PATH):
    try:
        df = pd.read_csv(f, encoding='latin-1')
        df.columns = df.columns.str.strip()
        df = df.rename(columns={'Epoch (ms)': 'EPOCH', 'BPM': 'HEART_RATE'})
        if 'HEART_RATE' in df.columns:
            df = df[df['HEART_RATE'] > 0]
            if not df.empty:
                df['datetime'] = to_datetime_auto(df['EPOCH'])
                bangle_dfs.append(df[['datetime', 'HEART_RATE']])
    except: pass

if not ref_dfs or not bangle_dfs: raise ValueError("No data found!")

ref_df = pd.concat(ref_dfs, ignore_index=True).drop_duplicates('datetime').sort_values('datetime')
bangle_df = pd.concat(bangle_dfs, ignore_index=True).drop_duplicates('datetime').sort_values('datetime')
print(f"✅ Loaded: {len(ref_df):,} Ref / {len(bangle_df):,} Bangle rows")

# =============================================================================
# 2. LAG CORRECTION & CHUNKING
# =============================================================================
print("🔗 Calculating Lag & Chunking...")

# 1. Global Lag Calc
initial_align = pd.merge_asof(
    bangle_df, ref_df, on='datetime', direction='nearest', 
    tolerance=pd.Timedelta('5s'), suffixes=('_bangle', '_ref')
).dropna()

best_lag = 0
best_corr = 0
for lag in range(-20, 21):
    c = initial_align['HEART_RATE_bangle'].corr(initial_align['HEART_RATE_ref'].shift(lag))
    if c > best_corr:
        best_corr = c
        best_lag = lag

print(f"   ⚡ Lag: {best_lag}s")
ref_df['datetime'] -= pd.Timedelta(seconds=best_lag)

# 2. Sessionizing
aligned = pd.merge_asof(
    bangle_df, ref_df, on='datetime', direction='nearest', 
    tolerance=pd.Timedelta('5s'), suffixes=('_bangle', '_ref')
).dropna(subset=['HEART_RATE_ref'])

aligned['session_id'] = (aligned['datetime'].diff() > pd.Timedelta('30s')).cumsum()
valid_sessions = aligned['session_id'].value_counts()[lambda x: x > 60].index
aligned = aligned[aligned['session_id'].isin(valid_sessions)].copy()

# =============================================================================
# 3. FEATURE ENGINEERING (The Upgrade)
# =============================================================================
print("🔧 Engineering Features...")

def process_chunk(chunk):
    chunk = chunk.sort_values('datetime').set_index('datetime')
    
    # Base Inputs
    # We use a 5s MA as the "Base Hypothesis" to filter raw sensor jitter
    chunk['base_hr'] = chunk['HEART_RATE_bangle'].rolling('5s').mean()
    chunk['slow_ma'] = chunk['HEART_RATE_bangle'].rolling('30s').mean()
    
    # 1. Intensity Ratio (Sprint vs Recovery)
    # > 1.0 means HR is rising (Sprint), < 1.0 means falling (Recovery)
    chunk['intensity_ratio'] = chunk['base_hr'] / (chunk['slow_ma'] + 1e-6)
    
    # 2. Volatility (Confidence)
    chunk['volatility'] = chunk['HEART_RATE_bangle'].rolling('15s').std().fillna(0)
    
    # 3. Acceleration (Derivative)
    chunk['hr_accel'] = chunk['base_hr'].diff().fillna(0)
    
    # TARGET: RESIDUAL (Error)
    # We want to predict: "How far off is the Base HR from the Real HR?"
    chunk['target_residual'] = chunk['HEART_RATE_ref'] - chunk['base_hr']
    
    return chunk.dropna()

processed_chunks = []
for _, group in aligned.groupby('session_id'):
    processed_chunks.append(process_chunk(group))

if not processed_chunks: raise ValueError("No data left.")
final_df = pd.concat(processed_chunks)

# =============================================================================
# 4. TRAIN (Residual Learning)
# =============================================================================
print("🧠 Training on Residuals...")

features = ['base_hr', 'slow_ma', 'intensity_ratio', 'volatility', 'hr_accel']
X = final_df[features]
y = final_df['target_residual'] # <--- Predicting the ERROR, not the HR

unique_sessions = final_df['session_id'].unique()
train_ids, test_ids = train_test_split(unique_sessions, test_size=0.2, shuffle=False)

train_mask = final_df['session_id'].isin(train_ids)
test_mask = final_df['session_id'].isin(test_ids)
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Deeper trees to capture sharp peaks
model = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05)
model.fit(X_train, y_train)

# INFERENCE
# Final HR = Base Hypothesis + Predicted Error
pred_residual = model.predict(X_test)
final_prediction = X_test['base_hr'] + pred_residual
true_values = final_df.loc[test_mask, 'HEART_RATE_ref']

mae = mean_absolute_error(true_values, final_prediction)
print(f"🚀 FINAL MAE: {mae:.2f} BPM")

# =============================================================================
# 5. PLOT
# =============================================================================
plt.figure(figsize=(12, 6))

if len(test_ids) > 0:
    sid = test_ids[0]
    sample = final_df[final_df['session_id'] == sid]
    
    # Re-run prediction for this specific session slice
    s_feats = sample[features]
    s_base = sample['base_hr']
    s_res = model.predict(s_feats)
    s_final = s_base + s_res
    
    plt.plot(sample.index, sample['HEART_RATE_ref'], 'k-', lw=2, label='Reference')
    plt.plot(sample.index, sample['HEART_RATE_bangle'], 'r--', alpha=0.15, label='Bangle Raw')
    
    # Plot the Base (what we started with) vs The Corrected
    plt.plot(sample.index, s_base, 'orange', lw=1, alpha=0.8, label='Base Input (5s MA)')
    plt.plot(sample.index, s_final, 'g-', lw=2, label='Residual Corrected')
    
    plt.title(f"Residual Learning | Session #{sid} | MAE: {mae:.2f}")
    plt.legend()
else:
    plt.title("No Test Data")

plt.tight_layout()
plt.savefig("residual_result.png")
print("📸 Saved 'residual_result.png'")
