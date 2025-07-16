# AI-Powered Health Monitoring System

A real-time AI-powered system for simulating, monitoring, and analyzing health vitals, designed to detect potential health anomalies and generate actionable recommendations. The system is visualized through a simple and interactive Streamlit web dashboard.

## 🚀 Live Demo

Access the deployed app here: [https://ai-health-app-fyx5noe7pleabe6mduqjmw.streamlit.app/](https://ai-health-app-fyx5noe7pleabe6mduqjmw.streamlit.app/)

## 📊 Features

- Real-time health data simulation (heart rate, blood oxygen, temperature, respiration rate, activity level).
- Anomaly detection using Isolation Forest.
- Health recommendations based on AI analysis.
- Interactive dashboard built with Streamlit.
- Visualization of vital signs and detected anomalies.
- Exportable anomaly reports (CSV).

## 🛠 Technologies Used

- **Python 3**
- **Streamlit**
- **Scikit-learn**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**

## 📂 Project Structure

```
├── app.py                # Streamlit dashboard app
├── data_utils.py         # Data simulation and preprocessing
├── health_model.py       # AI model and recommendations engine
├── requirements.txt      # Project dependencies
```

## ⚙️ Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/ai-health-monitoring.git
cd ai-health-monitoring
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the app locally:**

```bash
streamlit run app.py
```

## 📈 Deployment

This project is deployed on Streamlit Cloud. To deploy your own instance:

1. Upload your project to GitHub.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud).
3. Connect your GitHub account.
4. Deploy directly from your repo specifying `app.py` as the entry point.

## 📋 License

This project is open-source and free to use under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please fork this repo and submit pull requests.

---

*Developed by Tawana Msebele*

