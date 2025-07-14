import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

def simulate_health_data(num_users=5, minutes=500):
    user_ids = [f'user_{i+1}' for i in range(num_users)]
    start_time = datetime.now()
    data = []

    for user in user_ids:
        timestamp = start_time
        for _ in range(minutes):
            data.append({
                'user_id': user,
                'timestamp': timestamp,
                'heart_rate': np.random.randint(60, 100),
                'blood_oxygen': np.random.randint(90, 100),
                'temperature': np.random.normal(36.5, 0.5),
                'respiration_rate': np.random.randint(12, 20),
                'activity_level': np.random.choice(['low', 'moderate', 'high'])
            })
            timestamp += timedelta(minutes=1)

    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    df = df.copy()
    df['activity_level'] = df['activity_level'].map({'low': 0, 'moderate': 1, 'high': 2})
    features = ['heart_rate', 'blood_oxygen', 'temperature', 'respiration_rate', 'activity_level']
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    return df, df_scaled, features