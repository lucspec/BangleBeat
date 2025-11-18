#!/usr/bin/env python3
"""
Heart Rate Data Loading Library
Functions for loading HR data from various sources
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path

try:
    import pandas as pd
except ImportError as e:
    print(f"Error: {e}")
    print("Run: pip install pandas")
    exit(1)


def getGadgetbridgeDevices(db_path: str) -> pd.DataFrame:
    """
    Get list of all devices in Gadgetbridge database
    
    Parameters:
    -----------
    db_path : str
        Path to Gadgetbridge.db SQLite database
    
    Returns:
    --------
    pd.DataFrame
        Device information with columns: id, name, mfg, identifier, type, type_name, model
    """
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        _id as id,
        NAME as name,
        MANUFACTURER as mfg,
        IDENTIFIER as identifier,
        TYPE as type,
        TYPE_NAME as type_name,
        MODEL as model
    FROM DEVICE
    ORDER BY _id
    """
    
    devices = pd.read_sql_query(query, conn)
    conn.close()
    
    return devices


def getGadgetbridgeData(db_path: str,
                        device_ids: Optional[List[int]] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        auto_detect: bool = True) -> pd.DataFrame:
    """
    Load heart rate data from Gadgetbridge database
    Automatically detects device types and loads from appropriate tables
    
    Parameters:
    -----------
    db_path : str
        Path to Gadgetbridge.db SQLite database
    device_ids : list of int, optional
        Specific device IDs to load. If None and auto_detect=True, loads all devices
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'
    end_date : str, optional
        End date in format 'YYYY-MM-DD'
    auto_detect : bool
        If True, automatically detect device types and tables
    
    Returns:
    --------
    pd.DataFrame
        Combined heart rate data with columns:
        ['device_id', 'device_name', 'device_type', 'TIMESTAMP', 'HEART_RATE', 'datetime']
    """
    conn = sqlite3.connect(db_path)
    
    # Get device information
    devices = getGadgetbridgeDevices(db_path)
    
    if devices.empty:
        print("No devices found in database")
        conn.close()
        return pd.DataFrame()
    
    # Filter to specific device IDs if provided
    if device_ids is not None:
        devices = devices[devices['id'].isin(device_ids)]
    
    # Map device types to their activity sample tables
    # Based on your devices and common Gadgetbridge patterns
    type_to_table = {
        'GARMIN_INSTINCT_2X_SOLAR': 'GARMIN_ACTIVITY_SAMPLE',
        'BANGLEJS': 'BANGLE_JSACTIVITY_SAMPLE',
        'POLAR': 'POLAR_H10_ACTIVITY_SAMPLE',
        # Add more mappings as needed
    }
    
    dataframes = []
    
    for _, device in devices.iterrows():
        device_id = device['id']
        device_name = device['name']
        device_type = device['type_name']
        
        # Determine table name
        table_name = None
        
        # Try exact match first
        if device_type in type_to_table:
            table_name = type_to_table[device_type]
        # Try partial matches for variations
        elif 'GARMIN' in device_type:
            table_name = 'GARMIN_ACTIVITY_SAMPLE'
        elif 'BANGLE' in device_type:
            table_name = 'BANGLE_JSACTIVITY_SAMPLE'
        elif 'POLAR' in device_type:
            table_name = 'POLAR_H10_ACTIVITY_SAMPLE'
        
        if not table_name:
            #print(f"Warning: Unknown device type '{device_type}' for device '{device_name}', skipping")
            continue
        
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        
        if not cursor.fetchone():
            print(f"Warning: Table '{table_name}' not found for device '{device_name}', skipping")
            continue
        
        # Build query
        query = f"""
        SELECT
            ? as device_id,
            ? as device_name,
            ? as device_type,
            TIMESTAMP,
            HEART_RATE
        FROM {table_name}
        WHERE DEVICE_ID = ? AND HEART_RATE IS NOT NULL AND HEART_RATE > 0
        """
        
        params = [device_id, device_name, device_type, device_id]
        
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
                print(f"Loaded {len(df):,} measurements from {device_name} ({device_type})")
        except Exception as e:
            print(f"Warning: Could not load data for {device_name}: {e}")
    
    conn.close()
    
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return pd.DataFrame()


def getLoglogCsv(csv_path: str,
                 device_name: str = 'loglog-banglejs',
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
        
        # Add device name and type
        df['device_name'] = device_name
        df['device_type'] = 'LOGLOG_CSV'
        df['device_id'] = 999  # Arbitrary ID for CSV data
        
        # Filter by date if provided
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            df = df[df['datetime'] >= start_dt]
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            df = df[df['datetime'] <= end_dt]
        
        print(f"Loaded {len(df):,} measurements from {device_name} (CSV)")
        
        return df
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame()


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
            standardized_dfs.append(df)
        else:
            print(f"Warning: Dataframe missing required columns, skipping")
    
    if standardized_dfs:
        combined = pd.concat(standardized_dfs, ignore_index=True)
        # Sort by timestamp
        combined = combined.sort_values('datetime').reset_index(drop=True)
        return combined
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


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hr_data_loaders.py <path_to_Gadgetbridge.db> [csv_path]")
        print("\nThis will display all devices in the database and load their data")
        sys.exit(1)
    
    db_path = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("="*80)
    print("GADGETBRIDGE DEVICES")
    print("="*80)
    
    devices = getGadgetbridgeDevices(db_path)
    print(devices.to_string(index=False))
    
    print("\n" + "="*80)
    print("LOADING HEART RATE DATA")
    print("="*80)
    
    gb_data = getGadgetbridgeData(db_path)
    
    if csv_path:
        csv_data = getLoglogCsv(csv_path)
        all_data = combineHrData(gb_data, csv_data)
    else:
        all_data = gb_data
    
    if not all_data.empty:
        print(f"\nTotal measurements: {len(all_data):,}")
        print(f"Date range: {all_data['datetime'].min()} to {all_data['datetime'].max()}")
        print("\nBy device:")
        for device in all_data['device_name'].unique():
            count = len(all_data[all_data['device_name'] == device])
            print(f"  - {device}: {count:,}")
    else:
        print("\nNo data loaded!")