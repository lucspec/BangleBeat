#!/usr/bin/env python3
"""
Heart Rate Comparison Tool for Gadgetbridge Devices
Compare heart rate data from:
- Garmin Instinct 2X Solar (GARMIN_ACTIVITY_SAMPLE)
- Bangle.js (BANGLE_JSACTIVITY_SAMPLE)
- Polar H10 (POLAR_H10_ACTIVITY_SAMPLE)
"""

import sys
import sqlite3
from datetime import datetime, timedelta
import os

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import seaborn as sns
except ImportError as e:
    print(f"Error: {e}")
    print("Run: poetry install --with analysis")
    sys.exit(1)

DB_PATH = "/home/luc/Sync/phone-general/BioData/Gadgetbridge.db"

def get_garmin_hr_data(conn, device_id=1, start_date=None, end_date=None):
    """Get heart rate data from Garmin Instinct 2X Solar"""
    query = """
    SELECT
        'Garmin Instinct 2X Solar' as device_name,
        TIMESTAMP,
        HEART_RATE
    FROM GARMIN_ACTIVITY_SAMPLE
    WHERE DEVICE_ID = ? AND HEART_RATE IS NOT NULL AND HEART_RATE > 0
    """

    params = [device_id]

    if start_date:
        timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        query += " AND TIMESTAMP >= ?"
        params.append(timestamp)
    if end_date:
        timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        query += " AND TIMESTAMP <= ?"
        params.append(timestamp)

    query += " ORDER BY TIMESTAMP"

    df = pd.read_sql_query(query, conn, params=params)
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
    return df

def get_banglejs_hr_data(conn, device_id=3, start_date=None, end_date=None):
    """Get heart rate data from Bangle.js"""
    query = """
    SELECT
        'Bangle.js' as device_name,
        TIMESTAMP,
        HEART_RATE
    FROM BANGLE_JSACTIVITY_SAMPLE
    WHERE DEVICE_ID = ? AND HEART_RATE IS NOT NULL AND HEART_RATE > 0
    """

    params = [device_id]

    if start_date:
        timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        query += " AND TIMESTAMP >= ?"
        params.append(timestamp)
    if end_date:
        timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        query += " AND TIMESTAMP <= ?"
        params.append(timestamp)

    query += " ORDER BY TIMESTAMP"

    df = pd.read_sql_query(query, conn, params=params)
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
    return df

def get_polar_hr_data(conn, device_id=4, start_date=None, end_date=None):
    """Get heart rate data from Polar H10 (chest strap - reference device)"""
    query = """
    SELECT
        'Polar H10' as device_name,
        TIMESTAMP,
        HEART_RATE
    FROM POLAR_H10_ACTIVITY_SAMPLE
    WHERE DEVICE_ID = ? AND HEART_RATE IS NOT NULL AND HEART_RATE > 0
    """

    params = [device_id]

    if start_date:
        timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        query += " AND TIMESTAMP >= ?"
        params.append(timestamp)
    if end_date:
        timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        query += " AND TIMESTAMP <= ?"
        params.append(timestamp)

    query += " ORDER BY TIMESTAMP"

    df = pd.read_sql_query(query, conn, params=params)
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
    return df

def get_all_hr_data(conn, start_date=None, end_date=None):
    """Get heart rate data from all devices"""
    garmin = get_garmin_hr_data(conn, start_date=start_date, end_date=end_date)
    bangle = get_banglejs_hr_data(conn, start_date=start_date, end_date=end_date)
    polar = get_polar_hr_data(conn, start_date=start_date, end_date=end_date)

    all_data = pd.concat([garmin, bangle, polar], ignore_index=True)
    return all_data

def calculate_device_statistics(df):
    """Calculate statistical metrics for each device"""
    stats_list = []

    for device in df['device_name'].unique():
        device_data = df[df['device_name'] == device]['HEART_RATE']

        stats_dict = {
            'Device': device,
            'Count': len(device_data),
            #'Mean HR': device_data.mean(),
            #'Median HR': device_data.median(),
            'Std Dev': device_data.std(),
            'Min HR': device_data.min(),
            'Max HR': device_data.max(),
            'Q1 (25%)': device_data.quantile(0.25),
            'Q3 (75%)': device_data.quantile(0.75),
        }
        stats_list.append(stats_dict)

    return pd.DataFrame(stats_list)

def compare_with_polar(df, tolerance_seconds=60):
    """
    Compare devices against Polar H10 (gold standard chest strap)

    Parameters:
    - tolerance_seconds: Maximum time difference to consider measurements as simultaneous
    """
    polar_data = df[df['device_name'] == 'Polar H10'][['datetime', 'HEART_RATE']].copy()

    if polar_data.empty:
        print("No Polar H10 data available for comparison")
        return None

    polar_data.columns = ['datetime', 'polar_hr']
    comparisons = []

    for device in df['device_name'].unique():
        if device == 'Polar H10':
            continue

        device_data = df[df['device_name'] == device][['datetime', 'HEART_RATE']].copy()
        device_data.columns = ['datetime', 'device_hr']

        # Merge on nearest timestamps
        merged = pd.merge_asof(
            device_data.sort_values('datetime'),
            polar_data.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(f'{tolerance_seconds}s')
        ).dropna()

        if not merged.empty:
            diff = merged['device_hr'] - merged['polar_hr']
            mae = np.abs(diff).mean()
            rmse = np.sqrt((diff ** 2).mean())
            correlation = merged['device_hr'].corr(merged['polar_hr'])

            comparisons.append({
                'Device': device,
                'Paired Samples': len(merged),
                'MAE (bpm)': mae,
                'RMSE (bpm)': rmse,
                'Correlation': correlation,
                'Mean Diff (bpm)': diff.mean(),
                'Std Diff (bpm)': diff.std(),
                'Max Error (bpm)': np.abs(diff).max()
            })

    return pd.DataFrame(comparisons) if comparisons else None

def plot_hr_distributions(df):
    """Plot heart rate distribution for each device"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    devices = sorted(df['device_name'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Box plot
    data_for_box = [df[df['device_name'] == d]['HEART_RATE'].values for d in devices]
    bp = axes[0].boxplot(data_for_box, labels=devices, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[0].set_title('Heart Rate Distribution by Device', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Heart Rate (bpm)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=15)

    # Histogram overlay
    for device, color in zip(devices, colors):
        device_data = df[df['device_name'] == device]['HEART_RATE']
        axes[1].hist(device_data, bins=30, alpha=0.5, label=device, color=color, edgecolor='black')

    axes[1].set_title('Heart Rate Histogram Overlay', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Heart Rate (bpm)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_hr_normal_distributions(df):
    """Plot heart rate distribution for each device (normalized for different sample sizes)"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    devices = sorted(df['device_name'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Box plot
    data_for_box = [df[df['device_name'] == d]['HEART_RATE'].values for d in devices]
    bp = axes[0].boxplot(data_for_box, labels=devices, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    axes[0].set_title('Heart Rate Distribution by Device', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Heart Rate (bpm)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=15)
    
    # Normalized histogram overlay (density=True makes it comparable across sample sizes)
    for device, color in zip(devices, colors):
        device_data = df[df['device_name'] == device]['HEART_RATE']
        n_samples = len(device_data)
        axes[1].hist(device_data, bins=30, alpha=0.5, 
                    label=f'{device} (n={n_samples:,})', 
                    color=color, edgecolor='black',
                    density=True)  # This normalizes to probability density
    
    axes[1].set_title('Heart Rate Distribution (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Heart Rate (bpm)', fontsize=12)
    axes[1].set_ylabel('Probability Density', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_hr_timeline(df, max_hours=48, hide_bpm=True):
    """Plot heart rate over time for each device"""
    recent_date = df['datetime'].max() - timedelta(hours=max_hours)
    df_recent = df[df['datetime'] >= recent_date].copy()

    if df_recent.empty:
        print(f"No data in the last {max_hours} hours")
        return

    fig, ax = plt.subplots(figsize=(16, 6))

    colors = {'Garmin Instinct 2X Solar': '#1f77b4',
              'Bangle.js': '#ff7f0e',
              'Polar H10': '#2ca02c'}

    for device in sorted(df_recent['device_name'].unique()):
        device_data = df_recent[df_recent['device_name'] == device]
        if hide_bpm:
            ax.scatter(
                device_data['datetime'], 
                device_data['HEART_RATE']/max(device_data['HEART_RATE']),
                label=device, alpha=0.6, s=15, color=colors.get(device, 'gray'))
        else:
            ax.scatter(
                device_data['datetime'], 
                device_data['HEART_RATE'],
                label=device, alpha=0.6, s=15, color=colors.get(device, 'gray'))

    ax.set_title(f'Heart Rate Timeline (Last {max_hours} Hours)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date/Time', fontsize=12)
    ax.set_ylabel('Heart Rate (bpm)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_bland_altman(df, tolerance_seconds=60):
    """Bland-Altman plot comparing devices to Polar H10"""
    polar_data = df[df['device_name'] == 'Polar H10'][['datetime', 'HEART_RATE']].copy()

    if polar_data.empty:
        print("No Polar H10 data available for Bland-Altman analysis")
        return

    polar_data.columns = ['datetime', 'polar_hr']

    other_devices = [d for d in df['device_name'].unique() if d != 'Polar H10']

    if not other_devices:
        print("Need at least one other device for comparison")
        return

    fig, axes = plt.subplots(1, len(other_devices), figsize=(8*len(other_devices), 6))

    if len(other_devices) == 1:
        axes = [axes]

    for idx, device in enumerate(other_devices):
        device_data = df[df['device_name'] == device][['datetime', 'HEART_RATE']].copy()
        device_data.columns = ['datetime', 'device_hr']

        merged = pd.merge_asof(
            device_data.sort_values('datetime'),
            polar_data.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(f'{tolerance_seconds}s')
        ).dropna()

        if not merged.empty:
            mean_hr = (merged['device_hr'] + merged['polar_hr']) / 2
            diff_hr = merged['device_hr'] - merged['polar_hr']

            mean_diff = diff_hr.mean()
            std_diff = diff_hr.std()

            axes[idx].scatter(mean_hr, diff_hr, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
            axes[idx].axhline(mean_diff, color='red', linestyle='--', linewidth=2,
                             label=f'Mean: {mean_diff:.2f} bpm')
            axes[idx].axhline(mean_diff + 1.96 * std_diff, color='red', linestyle=':', linewidth=2,
                             label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.2f}')
            axes[idx].axhline(mean_diff - 1.96 * std_diff, color='red', linestyle=':', linewidth=2,
                             label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.2f}')
            axes[idx].axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

            axes[idx].set_xlabel('Mean HR (bpm)', fontsize=12)
            axes[idx].set_ylabel('Difference (Device - Polar H10) [bpm]', fontsize=12)
            axes[idx].set_title(f'Bland-Altman: {device} vs Polar H10', fontsize=14, fontweight='bold')
            axes[idx].legend(loc='best')
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_scatter_comparison(df, tolerance_seconds=60):
    """Scatter plot showing device HR vs Polar H10 HR"""
    polar_data = df[df['device_name'] == 'Polar H10'][['datetime', 'HEART_RATE']].copy()

    if polar_data.empty:
        print("No Polar H10 data available")
        return

    polar_data.columns = ['datetime', 'polar_hr']
    other_devices = [d for d in df['device_name'].unique() if d != 'Polar H10']

    if not other_devices:
        return

    fig, axes = plt.subplots(1, len(other_devices), figsize=(8*len(other_devices), 6))

    if len(other_devices) == 1:
        axes = [axes]

    for idx, device in enumerate(other_devices):
        device_data = df[df['device_name'] == device][['datetime', 'HEART_RATE']].copy()
        device_data.columns = ['datetime', 'device_hr']

        merged = pd.merge_asof(
            device_data.sort_values('datetime'),
            polar_data.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(f'{tolerance_seconds}s')
        ).dropna()

        if not merged.empty:
            axes[idx].scatter(merged['polar_hr'], merged['device_hr'], alpha=0.5, s=30,
                             edgecolors='black', linewidth=0.5)

            # Plot perfect agreement line
            min_val = min(merged['polar_hr'].min(), merged['device_hr'].min())
            max_val = max(merged['polar_hr'].max(), merged['device_hr'].max())
            axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                          label='Perfect Agreement')

            # Calculate and display R²
            correlation = merged['polar_hr'].corr(merged['device_hr'])
            axes[idx].text(0.05, 0.95, f'R = {correlation:.3f}\nR² = {correlation**2:.3f}',
                          transform=axes[idx].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            axes[idx].set_xlabel('Polar H10 HR (bpm)', fontsize=12)
            axes[idx].set_ylabel(f'{device} HR (bpm)', fontsize=12)
            axes[idx].set_title(f'{device} vs Polar H10', fontsize=14, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

def main():
    try:
        print(f"Connecting to database: {DB_PATH}")
        conn = sqlite3.connect(DB_PATH)

        # Get all HR data
        print("\nFetching heart rate data from all devices...")
        hr_data = get_all_hr_data(conn)

        if hr_data.empty:
            print("No heart rate data found!")
            return

        print(f"\nTotal HR measurements: {len(hr_data)}")
        print(f"Date range: {hr_data['datetime'].min()} to {hr_data['datetime'].max()}")
        print(f"\nDevices found:")
        for device in sorted(hr_data['device_name'].unique()):
            count = len(hr_data[hr_data['device_name'] == device])
            date_range = hr_data[hr_data['device_name'] == device]['datetime']
            print(f"  - {device}: {count:,} measurements ({date_range.min()} to {date_range.max()})")

        # Calculate statistics
        print("\n" + "="*80)
        print("DEVICE STATISTICS")
        print("="*80)
        stats_df = calculate_device_statistics(hr_data)
        print(stats_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

        # Compare with Polar H10
        print("\n" + "="*80)
        print("ACCURACY COMPARISON (vs Polar H10 Chest Strap)")
        print("="*80)
        polar_comparison = compare_with_polar(hr_data, tolerance_seconds=60)
        if polar_comparison is not None and not polar_comparison.empty:
            print(polar_comparison.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
            print("\nInterpretation:")
            print("  - MAE/RMSE: Lower is better (how many bpm off on average)")
            print("  - Correlation: Closer to 1.0 is better (how well it tracks changes)")
            print("  - Mean Diff: Positive = reads higher than Polar, Negative = reads lower")
        else:
            print("Not enough simultaneous measurements to compare devices")
            print("(Devices need to be worn at the same time within 60 seconds)")

        # Generate visualizations
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        print("\n1. Heart rate distributions...")
        plot_hr_distributions(hr_data)

        print("2. Heart rate timeline...")
        plot_hr_timeline(hr_data, max_hours=48)

        if 'Polar H10' in hr_data['device_name'].values:
            print("3. Bland-Altman comparison (medical-grade accuracy plot)...")
            plot_bland_altman(hr_data, tolerance_seconds=60)

            print("4. Scatter plot comparison...")
            plot_scatter_comparison(hr_data, tolerance_seconds=60)

        conn.close()
        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)

    except FileNotFoundError:
        print(f"Error: Database not found at {DB_PATH}")
        print("Please update DB_PATH in the script.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
