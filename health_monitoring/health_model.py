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