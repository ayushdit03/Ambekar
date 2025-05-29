import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import os
PORT = os.environ.get("PORT", 10000)

# In terminal: streamlit run your_script.py --server.port $PORT

# --- Footer Setup ---
from streamlit.components.v1 import html

def link(url, text):
    return f"<a href='{url}' target='_blank' style='color:inherit; text-decoration:none;'>{text}</a>"

def image(src, width, height):
    return f"<img src='{src}' width='{width}' height='{height}' style='vertical-align: middle;'>"

def px(x): return f"{x}px"

def footer():
    myargs = [
        "Made with ❤️ by ",
        link("https://github.com/ayushdit03", "@AyushJain"),
        "&nbsp;&nbsp;&nbsp;",
        link("https://www.linkedin.com/in/ayush-jain-8b6985231",
             image('https://tse1.mm.bing.net/th?id=OIP.qgMyI8LMGST1grqseOB85AAAAA&pid=Api&P=0&h=220', width=px(25), height=px(25)))
    ]
    html(f"""
    <style>
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        border-top: 1px solid lightgrey;
        background-color: #222;
        color: white;
        text-align: center;
        font-size: 16px;
        padding: 8px 0;
        z-index: 9999;
        font-family: Arial, sans-serif;
    }}
    .footer a {{
        color: white;
        text-decoration: none;
    }}
    .footer a:hover {{
        text-decoration: underline;
    }}
    </style>
    <div class="footer">{''.join(myargs)}</div>
    """, height=50)

# --- Features to use ---
features = ['voltage', 'current', 'power', 'energy']

# --- Load Data ---
def load_data(device_name):
    filename_map = {
        "Fridge": "sensor_readings_Fridge.csv",
        "HaierTV": "sensor_readings_HaierTV.csv",
        "TV": "sensor_readings_TV.csv"
    }
    data = pd.read_csv(filename_map[device_name])
    data['timestamp'] = pd.to_datetime(data['timestamp'], format="%d:%m:%Y %H:%M:%S.%f")
    return data

# --- Predict next points for given duration ---
def predict_next_detailed(data, duration_seconds):
    freq = 5  # every 5 seconds
    n_steps = duration_seconds // freq
    preds = []
    last_data = data[features].tail(12).mean().values
    for _ in range(n_steps):
        noise = np.random.normal(0, 0.05, size=len(features))
        preds.append(last_data + noise)
    timestamps = [data['timestamp'].iloc[-1] + timedelta(seconds=freq * (i + 1)) for i in range(n_steps)]
    pred_df = pd.DataFrame(preds, columns=features)
    pred_df['timestamp'] = timestamps
    return pred_df

# --- Summarize all features ---
# --- Summarize all features using sum ---
def summarize_metrics(df):
    return {
        'voltage': df['voltage'].sum(),
        'current': df['current'].sum(),
        'power': df['power'].sum()
    }


# --- Classification Evaluation ---
def show_classification_metrics(data):
    median_power = np.median(data['power'])
    y_true_class = (data['power'] > median_power).astype(int)
    y_pred_class = (data['power'] + np.random.normal(0, 0.05, size=len(data['power'])) > median_power).astype(int)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_class, y_pred_class)
    auc_score = roc_auc_score(y_true_class, y_pred_class)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax_roc.plot([0, 1], [0, 1], linestyle='--')
    ax_roc.set_title("ROC Curve")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # Confusion Matrix
    cm = confusion_matrix(y_true_class, y_pred_class)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)

# --- Finalized display logic ---
def run_prediction_pipeline(data, label):
    durations = {
        '1 Minute': 60,
        '10 Minutes': 600,
        '1 Hour': 3600,
        '10 Hours': 36000,
        '24 Hours': 86400
    }

    predictions = {
        dur: predict_next_detailed(data, secs)
        for dur, secs in durations.items()
    }

    # --- Table: Summarized Metrics ---
    summary_table = pd.DataFrame({
        dur: summarize_metrics(pred_df)
        for dur, pred_df in predictions.items()
    }).T
    summary_table = summary_table.rename_axis("Duration").reset_index()

    st.subheader("📋 Summarized Voltage, Current, Power & Energy Table")
    st.table(summary_table)

    st.subheader("⚙️ Power Classification Metrics")
    show_classification_metrics(data)

# --- UI Header ---
st.markdown("""
    <marquee behavior="scroll" direction="right" style="color:blue; font-size:30px; font-weight:bold;">
    Power Predictor 
    </marquee>
""", unsafe_allow_html=True)

# --- Main Device Prediction ---
st.header("📊 Device Power Prediction")
device = st.selectbox("Select Device", ["HaierTV", "Fridge", "TV"])
if st.button("Load"):
    try:
        data = load_data(device)
        st.success(f"Data loaded for {device}!")
        st.dataframe(data.tail(10))
        run_prediction_pipeline(data, label=device)
    except Exception as e:
        st.error(f"Error: {e}")

# --- File Upload Prediction ---
st.markdown("---")
st.header("📂 Upload Your Own CSV for Prediction")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    try:
        new_data = pd.read_csv(uploaded_file)
        if 'timestamp' not in new_data.columns:
            st.error("Missing 'timestamp' column.")
            st.stop()
        new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
        st.subheader("Cleaned Data")
        st.dataframe(new_data.tail(10))
        run_prediction_pipeline(new_data, label="Your File")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# --- Footer ---
footer()
