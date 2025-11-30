#!/usr/bin/env python
"""
Gadgetbridge SQLite3 Database Parser
Extracts and analyzes data from Gadgetbridge fitness tracker database
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Database path - update this to your Gadgetbridge database location
DB_PATH = "/home/luc/Documents/gadgetbridge_20251109112942/database/Gadgetbridge"

DEVICE_TABLES = [ 
    "GARMIN_HEART_RATE_RESTING_SAMPLE"
]

def connect_database(db_path):
    """Connect to the Gadgetbridge SQLite database"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
    return sqlite3.connect(db_path)

def list_tables(conn):
    """List all tables in the database"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

def get_table_schema(conn, table_name):
    """Get schema information for a specific table"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    return cursor.fetchall()

def get_devices(conn):
    """Get list of all devices"""
    query = "SELECT * FROM DEVICE"
    return pd.read_sql_query(query, conn)

def get_activity_data(conn, device_id=None, start_date=None, end_date=None):
    """
    Get activity data (steps, heart rate, etc.)
    
    Parameters:
    - device_id: Filter by specific device ID
    - start_date: Start date (YYYY-MM-DD)
    - end_date: End date (YYYY-MM-DD)
    """
    query = "SELECT * FROM MI_BAND_ACTIVITY_SAMPLE"
    conditions = []
    
    if device_id is not None:
        conditions.append(f"DEVICE_ID = {device_id}")
    if start_date:
        timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        conditions.append(f"TIMESTAMP >= {timestamp}")
    if end_date:
        timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        conditions.append(f"TIMESTAMP <= {timestamp}")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    df = pd.read_sql_query(query, conn)
    
    # Convert timestamp to datetime
    if 'TIMESTAMP' in df.columns:
        df['datetime'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
    
    return df

def get_sleep_data(conn, device_id=None):
    """Get sleep tracking data"""
    try:
        query = "SELECT * FROM MI_BAND_ACTIVITY_SAMPLE WHERE RAW_KIND IN (4, 5)"
        if device_id is not None:
            query += f" AND DEVICE_ID = {device_id}"
        
        df = pd.read_sql_query(query, conn)
        if 'TIMESTAMP' in df.columns:
            df['datetime'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
        return df
    except:
        return pd.DataFrame()

def calculate_daily_stats(activity_df):
    """Calculate daily statistics from activity data"""
    if activity_df.empty or 'datetime' not in activity_df.columns:
        return pd.DataFrame()
    
    activity_df['date'] = activity_df['datetime'].dt.date
    
    daily_stats = activity_df.groupby('date').agg({
        'STEPS': 'sum',
        'HEART_RATE': 'mean',
        'RAW_INTENSITY': 'mean'
    }).reset_index()
    
    return daily_stats

def plot_steps(daily_stats):
    """Plot daily steps over time"""
    if daily_stats.empty or 'STEPS' not in daily_stats.columns:
        print("No step data available")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_stats['date'], daily_stats['STEPS'], marker='o')
    plt.title('Daily Steps Over Time')
    plt.xlabel('Date')
    plt.ylabel('Steps')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_heart_rate(activity_df):
    """Plot heart rate over time"""
    if activity_df.empty or 'HEART_RATE' not in activity_df.columns:
        print("No heart rate data available")
        return
    
    # Filter out invalid heart rate values
    hr_data = activity_df[activity_df['HEART_RATE'] > 0].copy()
    
    if hr_data.empty:
        print("No valid heart rate data available")
        return
    
    plt.figure(figsize=(12, 6))
    plt.scatter(hr_data['datetime'], hr_data['HEART_RATE'], alpha=0.5, s=10)
    plt.title('Heart Rate Over Time')
    plt.xlabel('Date/Time')
    plt.ylabel('Heart Rate (BPM)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Main analysis workflow
if __name__ == "__main__":
    try:
        # Connect to database
        print(f"Connecting to database: {DB_PATH}")
        conn = connect_database(DB_PATH)
        
        # List all tables
        print("\nAvailable tables:")
        tables = list_tables(conn)
        for table in tables:
            print(f"  - {table}")
        
        # Get devices
        print("\nDevices in database:")
        devices = get_devices(conn)
        print(devices)
        
        # Get activity data
        print("\nFetching activity data...")
        activity_data = get_activity_data(conn)
        print(f"Total activity records: {len(activity_data)}")
        
        if not activity_data.empty:
            print("\nActivity data columns:")
            print(activity_data.columns.tolist())
            print("\nFirst few records:")
            print(activity_data.head())
            
            # Calculate daily statistics
            print("\nCalculating daily statistics...")
            daily_stats = calculate_daily_stats(activity_data)
            print(daily_stats.head(10))
            
            # Create visualizations
            print("\nGenerating plots...")
            plot_steps(daily_stats)
            plot_heart_rate(activity_data)
        
        # Get sleep data
        print("\nFetching sleep data...")
        sleep_data = get_sleep_data(conn)
        print(f"Total sleep records: {len(sleep_data)}")
        
        conn.close()
        print("\nAnalysis complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease update DB_PATH to point to your Gadgetbridge database file.")
        print("Common locations:")
        print("  - Android backup: /data/data/nodomain.freeyourgadget.gadgetbridge/databases/Gadgetbridge")
        print("  - Extracted: ./Gadgetbridge or ./gadgetbridge.db")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()