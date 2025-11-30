#!/usr/bin/env python3
"""
Heart Rate Data Library and Comparison Tool
Modular functions for loading HR data from various sources
"""

import sys
import sqlite3
from datetime import datetime, timedelta
import os
from typing import Optional, List
from pathlib import Path

from getGadgetbridgeData import getGadgetbridgeData
from getLoglogData import getLoglogData

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

def combineHrData(*dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multiple heart rate dataframes
    
    Parameters:
    -----------
    *dataframes : pd.DataFrame
        Variable number of dataframes to combine
    
    Returns:
    --------
    pd.DataFrame
        Combined dataframe with standard columns
    """
    # Filter out empty dataframes
    valid_dfs = [df for df in dataframes if not df.empty]
    
    if not valid_dfs:
        return pd.DataFrame()
    
    # Ensure all dataframes have the required columns
    required_cols = ['device_name', 'TIMESTAMP', 'HEART_RATE', 'datetime']
    
    standardized_dfs = []
    for df in valid_dfs:
        if all(col in df.columns for col in required_cols):
            # Keep only required columns plus any extras
            standardized_dfs.append(df)
        else:
            print(f"Warning: Dataframe missing required columns, skipping")
    
    if standardized_dfs:
        return pd.concat(standardized_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def normalizeHrData(df: pd.DataFrame, 
                    min_hr: float = 0, 
                    max_hr: float = 200,
                    inplace: bool = False) -> pd.DataFrame:
    """
    Normalize heart rate data to 0-1 scale based on physiological range
    
    Parameters:
    -----------
    df : pd.DataFrame
        Heart rate dataframe with HEART_RATE column
    min_hr : float
        Minimum heart rate for normalization (default: 0)
    max_hr : float
        Maximum heart rate for normalization (default: 200)
    inplace : bool
        If True, modifies the original dataframe. If False, returns a copy.
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added HR_NORMALIZED column (values 0-1)
    """
    if df.empty or 'HEART_RATE' not in df.columns:
        return df
    
    result = df if inplace else df.copy()
    
    # Normalize to 0-1 scale
    result['HR_NORMALIZED'] = (result['HEART_RATE'] - min_hr) / (max_hr - min_hr)
    
    # Clip values to 0-1 range (in case of values outside min/max)
    result['HR_NORMALIZED'] = result['HR_NORMALIZED'].clip(0, 1)
    
    return result

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def calculate_device_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistical metrics for each device"""
    stats_list = []

    for device in df['device_name'].unique():
        device_data = df[df['device_name'] == device]['HEART_RATE']

        stats_dict = {
            'Device': device,
            'Count': len(device_data),
            'Std Dev': device_data.std(),
            'Min HR': device_data.min(),
            'Max HR': device_data.max(),
            'Q1 (25%)': device_data.quantile(0.25),
            'Q3 (75%)': device_data.quantile(0.75),
        }
        stats_list.append(stats_dict)

    return pd.DataFrame(stats_list)


def compare_with_polar(df: pd.DataFrame, tolerance_seconds: int = 60) -> Optional[pd.DataFrame]:
    """
    Compare devices against Polar H10 (gold standard chest strap)

    Parameters:
    -----------
    df : pd.DataFrame
        Combined heart rate data
    tolerance_seconds : int
        Maximum time difference to consider measurements as simultaneous
    
    Returns:
    --------
    pd.DataFrame or None
        Comparison statistics for each device vs Polar H10
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

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_hr_distributions(df: pd.DataFrame, normalized: bool = False):
    """
    Plot heart rate distribution for each device
    
    Parameters:
    -----------
    df : pd.DataFrame
        Heart rate dataframe
    normalized : bool
        If True, use normalized HR values (0-1). Requires HR_NORMALIZED column.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    devices = sorted(df['device_name'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(devices)))
    
    # Select which column to use
    hr_column = 'HR_NORMALIZED' if normalized and 'HR_NORMALIZED' in df.columns else 'HEART_RATE'
    ylabel = 'Normalized Heart Rate (0-1)' if normalized else 'Heart Rate (bpm)'

    # Box plot
    data_for_box = [df[df['device_name'] == d][hr_column].values for d in devices]
    bp = axes[0].boxplot(data_for_box, labels=devices, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[0].set_title('Heart Rate Distribution by Device', fontsize=14, fontweight='bold')
    axes[0].set_ylabel(ylabel, fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=15)

    # Histogram overlay
    for device, color in zip(devices, colors):
        device_data = df[df['device_name'] == device][hr_column]
        axes[1].hist(device_data, bins=30, alpha=0.5, label=device, color=color, edgecolor='black')

    axes[1].set_title('Heart Rate Histogram Overlay', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(ylabel, fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_hr_timeline(df: pd.DataFrame, max_hours: int = 48, normalized: bool = False):
    """
    Plot heart rate over time for each device
    
    Parameters:
    -----------
    df : pd.DataFrame
        Heart rate dataframe
    max_hours : int
        Number of recent hours to plot
    normalized : bool
        If True, use normalized HR values (0-1). Requires HR_NORMALIZED column.
    """
    recent_date = df['datetime'].max() - timedelta(hours=max_hours)
    df_recent = df[df['datetime'] >= recent_date].copy()

    if df_recent.empty:
        print(f"No data in the last {max_hours} hours")
        return

    fig, ax = plt.subplots(figsize=(16, 6))

    devices = sorted(df_recent['device_name'].unique())
    colors_map = {
        'Garmin Instinct 2X Solar': '#1f77b4',
        'Bangle.js': '#ff7f0e',
        'Polar H10': '#2ca02c',
        'Custom App': '#d62728'
    }
    
    # Select which column to use
    hr_column = 'HR_NORMALIZED' if normalized and 'HR_NORMALIZED' in df_recent.columns else 'HEART_RATE'
    ylabel = 'Normalized Heart Rate (0-1)' if normalized else 'Heart Rate (bpm)'

    for device in devices:
        device_data = df_recent[df_recent['device_name'] == device]
        color = colors_map.get(device, 'gray')
        
        ax.scatter(
            device_data['datetime'], 
            device_data[hr_column],
            label=device, alpha=0.6, s=15, color=color)

    ax.set_title(f'Heart Rate Timeline (Last {max_hours} Hours)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date/Time', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_bland_altman(df: pd.DataFrame, tolerance_seconds: int = 60):
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

    n_devices = len(other_devices)
    fig, axes = plt.subplots(1, n_devices, figsize=(8*n_devices, 6))

    if n_devices == 1:
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


def plot_scatter_comparison(df: pd.DataFrame, tolerance_seconds: int = 60):
    """Scatter plot showing device HR vs Polar H10 HR"""
    polar_data = df[df['device_name'] == 'Polar H10'][['datetime', 'HEART_RATE']].copy()

    if polar_data.empty:
        print("No Polar H10 data available")
        return

    polar_data.columns = ['datetime', 'polar_hr']
    other_devices = [d for d in df['device_name'].unique() if d != 'Polar H10']

    if not other_devices:
        return

    n_devices = len(other_devices)
    fig, axes = plt.subplots(1, n_devices, figsize=(8*n_devices, 6))

    if n_devices == 1:
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

# =============================================================================
# MAIN COMPARISON SCRIPT
# =============================================================================

def main():
    # Configuration
    DB_PATH = "./data/Gadgetbridge.db"
    CUSTOM_APP_CSV = "./data/loglog.csv"
    
    # Check for CSV path in command line arguments
    if len(sys.argv) > 1:
        CUSTOM_APP_CSV = sys.argv[1]
        print(f"Using custom app CSV: {CUSTOM_APP_CSV}")
    
    try:
        print("="*80)
        print("LOADING DATA FROM ALL SOURCES")
        print("="*80)
        
        # Load Gadgetbridge data
        print(f"\n1. Loading Gadgetbridge data from: {DB_PATH}")
        gb_data = getGadgetbridgeData(DB_PATH, devices=['garmin', 'bangle', 'polar'])
        print(f"   Loaded {len(gb_data)} Gadgetbridge measurements")
        
        # Load custom app data if path provided
        if CUSTOM_APP_CSV:
            print(f"\n2. Loading custom app data from: {CUSTOM_APP_CSV}")
            loglog_data = getLoglogData(CUSTOM_APP_CSV, device_name='Custom App')
            print(f"   Loaded {len(loglog_data)} custom app measurements")
        else:
            print("\n2. No custom app CSV provided (use: python script.py /path/to/data.csv)")
            loglog_data = pd.DataFrame()
        
        # Combine all data
        print("\n3. Combining all data sources...")
        hr_data = combineHrData(gb_data, loglog_data)
        
        if hr_data.empty:
            print("No heart rate data found!")
            return
        
        # Normalize the data to 0-1 scale (0-200 bpm range)
        print("\n4. Normalizing heart rate data (0-200 bpm → 0-1 scale)...")
        hr_data = normalizeHrData(hr_data, min_hr=0, max_hr=200)
        print(f"   Added HR_NORMALIZED column")

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

        print("\n1. Heart rate distributions (raw BPM)...")
        plot_hr_distributions(hr_data, normalized=False)
        
        print("2. Heart rate distributions (normalized 0-1)...")
        plot_hr_distributions(hr_data, normalized=True)

        print("3. Heart rate timeline (raw BPM)...")
        plot_hr_timeline(hr_data, max_hours=48, normalized=False)
        
        print("4. Heart rate timeline (normalized 0-1)...")
        plot_hr_timeline(hr_data, max_hours=48, normalized=True)

        if 'Polar H10' in hr_data['device_name'].values:
            print("5. Bland-Altman comparison (medical-grade accuracy plot)...")
            plot_bland_altman(hr_data, tolerance_seconds=60)

            print("6. Scatter plot comparison...")
            plot_scatter_comparison(hr_data, tolerance_seconds=60)

        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        print(f"Please check DB_PATH ({DB_PATH}) and CSV path")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()