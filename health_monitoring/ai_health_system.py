# health_monitoring/data_utils.py

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

# health_monitoring/health_model.py

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import pandas as pd

def detect_anomalies(df_scaled, contamination=0.05):
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(df_scaled)
    return preds, model

def evaluate_model(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    return pd.DataFrame(report).transpose()

# Basic health recommendations based on thresholds
def generate_recommendations(df):
    recommendations = []
    for _, row in df.iterrows():
        advice = []
        if row['heart_rate'] > 100:
            advice.append("High heart rate detected. Consider resting.")
        if row['blood_oxygen'] < 92:
            advice.append("Low blood oxygen. Seek medical attention if persistent.")
        if row['temperature'] > 37.5:
            advice.append("Fever detected. Monitor temperature.")
        if not advice:
            advice.append("Vitals are within normal range.")
        recommendations.append(" ".join(advice))
    return recommendations

# health_monitoring/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import simulate_health_data, preprocess_data
from health_model import detect_anomalies, evaluate_model, generate_recommendations

st.set_page_config(page_title="AI Health Monitoring System", layout="wide")
st.title("🩺 AI-Powered Health Monitoring Dashboard")

# Sidebar Controls
st.sidebar.header("Configuration")
num_users = st.sidebar.slider("Number of Users", 1, 10, 3)
num_minutes = st.sidebar.slider("Minutes of Data per User", 100, 1000, 300)
contamination = st.sidebar.slider("Anomaly Rate", 0.01, 0.2, 0.05)

# Simulate or Upload Data
st.header("Simulated Health Data")
df = simulate_health_data(num_users, num_minutes)
st.dataframe(df.head(100))

# Preprocess and Detect Anomalies
df_processed, df_scaled, feature_cols = preprocess_data(df)
preds, model = detect_anomalies(df_scaled, contamination)
df_processed['anomaly'] = ['Anomaly' if x == -1 else 'Normal' for x in preds]

# Display Anomalies
st.header("Anomaly Detection Results")
st.dataframe(df_processed[['user_id', 'timestamp', 'heart_rate', 'blood_oxygen', 'temperature', 'anomaly']].head(100))

# Recommendations
st.header("Health Recommendations")
df_processed['recommendations'] = generate_recommendations(df_processed)
st.dataframe(df_processed[['user_id', 'timestamp', 'recommendations']].head(100))

# Visualize Anomalies
st.header("Anomaly Visualization")
fig, ax = plt.subplots()
anomaly_points = df_processed[df_processed['anomaly'] == 'Anomaly']
sns.lineplot(data=df_processed, x='timestamp', y='heart_rate', hue='user_id', ax=ax)
plt.scatter(anomaly_points['timestamp'], anomaly_points['heart_rate'], color='red', label='Anomaly')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig)

# Export Option
st.sidebar.markdown("---")
if st.sidebar.button("Export Anomaly Report"):
    df_processed.to_csv("anomaly_report.csv", index=False)
    st.success("Anomaly report saved as anomaly_report.csv")
