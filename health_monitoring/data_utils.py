import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

def simulate_health_data(num_users=5, minutes=120):
    user_ids = [f'user_{i+1}' for i in range(num_users)]
    start_time = datetime.now()
    data = []

    for idx, user in enumerate(user_ids):
    # Assign unique base values for each user
    base_heart_rate = np.random.randint(65, 85) + idx * 2
    base_oxygen = np.random.randint(93, 98) - idx
    base_temp = 36.5 + (idx * 0.1)
    base_respiration = np.random.randint(13, 18)

    timestamp = start_time

    for _ in range(minutes):
        data.append({
            'user_id': user,
            'timestamp': timestamp,
            'heart_rate': np.random.normal(base_heart_rate, 3),
            'blood_oxygen': np.random.normal(base_oxygen, 1),
            'temperature': np.random.normal(base_temp, 0.2),
            'respiration_rate': np.random.normal(base_respiration, 1),
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