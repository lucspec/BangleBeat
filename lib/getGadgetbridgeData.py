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

def getGadgetbridgeData(db_path: str, 
                    devices: Optional[List[str]] = None,
                    device_ids: Optional[dict] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load heart rate data from Gadgetbridge database
    
    Parameters:
    -----------
    db_path : str
        Path to Gadgetbridge.db SQLite database
    devices : list of str, optional
        List of device types to load. Options: 'garmin', 'bangle', 'polar'
        If None, loads all available devices
    device_ids : dict, optional
        Mapping of device type to device_id in database
        Example: {'garmin': 1, 'bangle': 3, 'polar': 4}
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'
    end_date : str, optional
        End date in format 'YYYY-MM-DD'
    
    Returns:
    --------
    pd.DataFrame
        Combined heart rate data with columns:
        ['device_name', 'TIMESTAMP', 'HEART_RATE', 'datetime']
    """
    if device_ids is None:
        device_ids = {'garmin': 1, 'bangle': 3, 'polar': 4}
    
    if devices is None:
        devices = ['garmin', 'bangle', 'polar']
    
    conn = sqlite3.connect(db_path)
    dataframes = []
    
    device_config = {
        'garmin': {
            'table': 'GARMIN_ACTIVITY_SAMPLE',
            'name': 'Garmin Instinct 2X Solar',
            'id': device_ids.get('garmin', 1)
        },
        'bangle': {
            'table': 'BANGLE_JSACTIVITY_SAMPLE',
            'name': 'Bangle.js',
            'id': device_ids.get('bangle', 3)
        },
        'polar': {
            'table': 'POLAR_H10_ACTIVITY_SAMPLE',
            'name': 'Polar H10',
            'id': device_ids.get('polar', 4)
        }
    }
    
    for device in devices:
        if device not in device_config:
            print(f"Warning: Unknown device '{device}', skipping")
            continue
        
        config = device_config[device]
        
        query = f"""
        SELECT
            ? as device_name,
            TIMESTAMP,
            HEART_RATE
        FROM {config['table']}
        WHERE DEVICE_ID = ? AND HEART_RATE IS NOT NULL AND HEART_RATE > 0
        """
        
        params = [config['name'], config['id']]
        
        if start_date:
            timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            query += " AND TIMESTAMP >= ?"
            params.append(timestamp)
        if end_date:
            timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
            query += " AND TIMESTAMP <= ?"
            params.append(timestamp)
        
        query += " ORDER BY TIMESTAMP"
        
        try:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
                dataframes.append(df)
        except Exception as e:
            print(f"Warning: Could not load {device} data: {e}")
    
    conn.close()
    
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return pd.DataFrame()