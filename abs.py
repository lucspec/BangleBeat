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
# 1. LOAD DATA
# =============================================================================
DB_SEARCH_PATH = "./data/Gadgetbridge*.db"
CSV_SEARCH_PATH = "./data/loglog*.csv"

def to_datetime_auto(series):
    if series.empty: return series
    if series.median() > 1e11: return pd.to_datetime(series, unit='ms')
    return pd.to_datetime(series, unit='s')

print("📂 Loading Data...")

# Load Reference
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
# 2. GLOBAL LAG CORRECTION (The Missing Piece)
# =============================================================================
print("🔗 Calculating Global Sensor Lag...")

# Initial alignment to find the lag
initial_align = pd.merge_asof(
    bangle_df, ref_df, on='datetime', direction='nearest', 
    tolerance=pd.Timedelta('5s'), suffixes=('_bangle', '_ref')
).dropna()

if len(initial_align) < 100: raise ValueError("Not enough overlap to calc lag.")

# Find lag
best_lag = 0
best_corr = 0
# Check +/- 20 seconds
for lag in range(-20, 21):
    c = initial_align['HEART_RATE_bangle'].corr(initial_align['HEART_RATE_ref'].shift(lag))
    if c > best_corr:
        best_corr = c
        best_lag = lag

print(f"   ⚡ Optimal Lag Detected: {best_lag} seconds (Correlation: {best_corr:.2f})")

# APPLY SHIFT TO REFERENCE DATASET
# (Negative lag means Bangle is ahead, Positive means Ref is ahead)
# We shift Ref to match Bangle
ref_df['datetime'] = ref_df['datetime'] - pd.Timedelta(seconds=best_lag)

# =============================================================================
# 3. SESSION CHUNKING
# =============================================================================
print("✂️  Sessionizing Data...")

# Re-Align with Corrected Time
aligned = pd.merge_asof(
    bangle_df, ref_df, on='datetime', direction='nearest', 
    tolerance=pd.Timedelta('5s'), suffixes=('_bangle', '_ref')
).dropna(subset=['HEART_RATE_ref'])

# Detect Breaks
aligned['time_diff'] = aligned['datetime'].diff()
aligned['session_id'] = (aligned['time_diff'] > pd.Timedelta('30s')).cumsum()

# Filter short chunks
session_counts = aligned['session_id'].value_counts()
valid_sessions = session_counts[session_counts > 60].index
aligned = aligned[aligned['session_id'].isin(valid_sessions)].copy()

print(f"   Processing {len(valid_sessions)} valid sessions...")

# =============================================================================
# 4. SIGNAL DECOMPOSITION (Smoothing Features)
# =============================================================================
print("🔧 Engineering Features (Fast vs Slow MA)...")

def process_chunk(chunk):
    chunk = chunk.sort_values('datetime').set_index('datetime')
    
    # 1. Fast Moving Average (The Trend) - 5s
    # Captures true HR changes, ignores single-point spikes
    chunk['ma_fast'] = chunk['HEART_RATE_bangle'].rolling('5s').mean()
    
    # 2. Slow Moving Average (The Baseline) - 30s
    # Captures the general intensity level
    chunk['ma_slow'] = chunk['HEART_RATE_bangle'].rolling('30s').mean()
    
    # 3. Volatility (Noise Level)
    # If this is high, the model should trust 'ma_slow' more.
    chunk['volatility'] = chunk['HEART_RATE_bangle'].rolling('10s').std().fillna(0)
    
    # 4. Divergence (How far is current point from baseline?)
    chunk['divergence'] = chunk['HEART_RATE_bangle'] - chunk['ma_slow']
    
    # Clean NaNs created by rolling
    return chunk.dropna()

processed_chunks = []
for _, group in aligned.groupby('session_id'):
    processed_chunks.append(process_chunk(group))

if not processed_chunks: raise ValueError("No valid data after processing.")
final_df = pd.concat(processed_chunks)

# =============================================================================
# 5. TRAIN
# =============================================================================
print("🧠 Training...")

# We do NOT use the raw 'HEART_RATE_bangle' as a primary feature anymore.
# We
