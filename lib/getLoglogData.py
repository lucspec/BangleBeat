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

def getLoglogData(csv_path: str,
                 device_name: str = 'loglog',
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load heart rate data from Loglog-format CSV
    
    CSV Format:
    Epoch (ms), Battery, X, Y, Z, Total, BPM, Confidence
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file
    device_name : str
        Name to use for this device in comparisons
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'
    end_date : str, optional
        End date in format 'YYYY-MM-DD'
    
    Returns:
    --------
    pd.DataFrame
        Heart rate data with columns:
        ['device_name', 'TIMESTAMP', 'HEART_RATE', 'datetime', 
         'battery', 'x', 'y', 'z', 'total_accel', 'confidence']
    """
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return pd.DataFrame()
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Clean up column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Rename columns to standard format
        column_mapping = {
            'Epoch (ms)': 'EPOCH_MS',
            'Battery': 'battery',
            'X': 'x',
            'Y': 'y', 
            'Z': 'z',
            'Total': 'total_accel',
            'BPM': 'HEART_RATE',
            'Confidence': 'confidence'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Filter out invalid heart rate readings
        df = df[df['HEART_RATE'].notna() & (df['HEART_RATE'] > 0)]
        
        # Convert epoch milliseconds to seconds for TIMESTAMP
        df['TIMESTAMP'] = df['EPOCH_MS'] / 1000
        
        # Convert to datetime
        df['datetime'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
        
        # Add device name
        df['device_name'] = device_name
        
        # Filter by date if provided
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            df = df[df['datetime'] >= start_dt]
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            df = df[df['datetime'] <= end_dt]
        
        return df
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame()
