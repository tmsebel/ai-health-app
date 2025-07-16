import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import simulate_health_data, preprocess_data
from health_model import detect_anomalies, evaluate_model, generate_recommendations

st.set_page_config(page_title="AI Health Monitoring System", layout="wide")
st.title("🩺 AI-Powered Health Monitoring Dashboard")

st.sidebar.header("Configuration")
num_users = st.sidebar.slider("Number of Users", 1, 10, 3)
num_minutes = st.sidebar.slider("Minutes of Data per User", 100, 1000, 300)
contamination = st.sidebar.slider("Anomaly Rate", 0.01, 0.2, 0.05)

st.header("Simulated Health Data")
df = simulate_health_data(num_users, num_minutes)
st.dataframe(df.head(100))

df_processed, df_scaled, feature_cols = preprocess_data(df)
preds, model = detect_anomalies(df_scaled, contamination)
df_processed['anomaly'] = ['Anomaly' if x == -1 else 'Normal' for x in preds]

st.header("Anomaly Detection Results")
st.dataframe(df_processed[['user_id', 'timestamp', 'heart_rate', 'blood_oxygen', 'temperature', 'anomaly']].head(100))

st.header("Health Recommendations")
df_processed['recommendations'] = generate_recommendations(df_processed)
st.dataframe(df_processed[['user_id', 'timestamp', 'recommendations']].head(100))

st.header("Anomaly Visualization")
fig, ax = plt.subplots()
anomaly_points = df_processed[df_processed['anomaly'] == 'Anomaly']
sns.lineplot(data=df_processed, x='timestamp', y='heart_rate', hue='user_id', ax=ax)
plt.scatter(anomaly_points['timestamp'], anomaly_points['heart_rate'], color='red', label='Anomaly')
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig)

st.sidebar.markdown("---")
if st.sidebar.button("Export Anomaly Report"):
    df_processed.to_csv("anomaly_report.csv", index=False)
    st.success("Anomaly report saved as anomaly_report.csv")