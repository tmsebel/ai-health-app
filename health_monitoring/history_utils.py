# health_monitoring/history_utils.py

import os
import pandas as pd
from datetime import datetime


def save_session_data(df, user_id):
    """
    Save the session data for a specific user to a CSV file, appending if file exists.
    """
    filename = f"{user_id}_history.csv"
    file_exists = os.path.exists(filename)
    df.to_csv(
        filename,
        mode='a',
        header=not file_exists,
        index=False
    )
    return filename


def generate_historical_summary(file_path):
    """
    Generate a summary from historical health data.
    - Average heart rate
    - Average oxygen level
    - Average temperature
    - Average respiration rate
    - Count of anomalies detected
    """
    df = pd.read_csv(file_path)

    if 'anomaly' not in df.columns:
        df['anomaly'] = 'Normal'  # If missing, assume all normal

    summary = df.groupby('user_id').agg({
        'heart_rate': 'mean',
        'blood_oxygen': 'mean',
        'temperature': 'mean',
        'respiration_rate': 'mean',
        'anomaly': lambda x: (x == 'Anomaly').sum()
    }).rename(columns={'anomaly': 'anomaly_count'})

    return summary.reset_index()


def generate_overall_summary(directory="."):
    """
    Generate a combined historical summary for all users found in CSVs in the directory.
    """
    all_summaries = []

    for file in os.listdir(directory):
        if file.endswith("_history.csv"):
            user_summary = generate_historical_summary(os.path.join(directory, file))
            all_summaries.append(user_summary)

    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        return combined_summary
    else:
        return pd.DataFrame()
