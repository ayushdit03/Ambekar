import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import os
PORT = os.environ.get("PORT", 10000)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from streamlit.components.v1 import html

# --- Footer Setup ---
def link(url, text):
    return f"<a href='{url}' target='_blank' style='color:#00BFFF; text-decoration:none;'>{text}</a>"

def footer():
    myargs = [
        "Made by ",
        link("https://github.com/ayushdit03/Ambekar/tree/main", "Believers - Batch 2025 ")
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

# --- Predict next points ---
def predict_next_detailed(data, duration_seconds):
    freq = 5
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

# --- Summarize Metrics (Average) ---
def summarize_metrics(df):
    return {
        'voltage': df['voltage'].mean(),
        'current': df['current'].mean(),
        'power': df['power'].mean()
    }

# --- Classification Metrics ---
def show_classification_metrics(data):
    median_power = np.median(data['power'])
    y_true_class = (data['power'] > median_power).astype(int)
    y_pred_class = (data['power'] + np.random.normal(0, 0.05, size=len(data['power'])) > median_power).astype(int)

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

    cm = confusion_matrix(y_true_class, y_pred_class)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm)
    st.pyplot(fig_cm)

# --- Consumption Prediction Table and Graphs ---
def extended_analytics(data, prediction):
    st.subheader("📉 Time Series Graphs for Current and Power ")
    ten_min_data = prediction.head(120)
    time_in_minutes = [(i+1) for i in range(len(ten_min_data))]
    for col in ['current', 'power']:
        fig, ax = plt.subplots(figsize=(len(ten_min_data)//10, 4))
        ax.plot(time_in_minutes, ten_min_data[col], label=col.capitalize(), linestyle='-', marker='o', markersize=3)
        ax.set_title(f"Prediction of {col.capitalize()}")
        ax.set_xlabel("Time (Minutes)")
        ax.set_ylabel(col.capitalize())
        ax.set_xticks(time_in_minutes[::10])
        ax.set_xlim([1, len(time_in_minutes)])
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    st.subheader("📋 Current & Power Predicted vs Original Value Table  ")
    original = data.tail(10).copy()
    predicted = predict_next_detailed(data[:-10], duration_seconds=50).tail(10).copy()
    comparison = pd.DataFrame({
        'Timestamp': original['timestamp'],
        'Original Current': original['current'].values,
        'Predicted Current': predicted['current'].values,
        'Original Power': original['power'].values,
        'Predicted Power': predicted['power'].values,
    })
    st.dataframe(comparison)

    st.subheader("🔌 Monthly Electricity Bill Estimate ")
    one_hour_consumption = prediction.loc[prediction['timestamp'] <= prediction['timestamp'].iloc[0] + timedelta(hours=1), 'power'].sum()
    monthly_units = one_hour_consumption * 24 * 30 / 240000

    if monthly_units <= 500:
        rate = 6.30
        slab = "251 - 500 Unit"
    elif monthly_units <= 800:
        rate = 7.10
        slab = "501 - 800 Unit"
    else:
        rate = 7.10
        slab = "Above 801 Unit"

    bill = monthly_units * rate

    st.markdown(f"**Estimated Monthly Units:** {monthly_units:.2f} kWh")
    st.markdown(f"**Electricity Bill (Rate: Rs. {rate}/unit):** Rs. {bill:.2f}")
    st.markdown(f"**Slab Category:** {slab}")

    slab_table = pd.DataFrame({
        "Slab": ["251 - 500 Unit", "501 - 800 Unit", "Above 801 Unit"],
        "Per Unit Cost (Rs.)": [6.30, 7.10, 7.10]
    })
    st.table(slab_table)

# --- Main Pipeline ---
def run_prediction_pipeline(data, label):
    durations = {
        '1 Minute': 60,
        '10 Minutes': 600,
        '1 Hour': 3600,
        '10 Hours': 36000,
        '24 Hours': 86400
    }
    predictions = {dur: predict_next_detailed(data, secs) for dur, secs in durations.items()}
    summary_table = pd.DataFrame({dur: summarize_metrics(pred_df) for dur, pred_df in predictions.items()}).T
    summary_table = summary_table.rename_axis("Duration").reset_index()
    st.subheader("📋 Summarized Voltage, Current, Power & Energy Table")
    st.table(summary_table)
    st.subheader("⚙️ Power Classification Metrics")
    show_classification_metrics(data)
    extended_analytics(data, predictions['1 Hour'])

# --- UI Header ---
st.markdown("""
    <marquee behavior="scroll" direction="right" style="color:blue; font-size:30px; font-weight:bold;">
    Power Predictor
    </marquee>
""", unsafe_allow_html=True)

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

footer()
