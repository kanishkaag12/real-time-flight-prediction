# ==================== STEP 6: BUILD STREAMLIT DASHBOARD ====================
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flight Price Analytics", layout="wide")

st.title("✈️ Real-Time Flight Price Prediction & Anomaly Detection")

# Load models and data
price_model = joblib.load('best_price_prediction_model.pkl')
anomaly_model = joblib.load('anomaly_detection_model.pkl')
data = pd.read_csv('flight_data_with_anomalies.csv')

# Sidebar for user input
st.sidebar.header("Flight Details")
airline = st.sidebar.selectbox("Airline", data['airline'].unique())
duration = st.sidebar.slider("Duration (hours)", 1.0, 10.0, 2.5)
days_left = st.sidebar.slider("Days until departure", 1, 50, 7)

# Prediction
if st.sidebar.button("Predict Price & Check Anomaly"):
    # Here you would convert inputs to PCA features first
    st.success("Prediction functionality would be implemented here!")
    
# Display analytics
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    ax.hist(data['price'], bins=30, alpha=0.7)
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

with col2:
    st.subheader("Anomaly Detection Results")
    anomaly_count = data['is_anomaly'].sum()
    st.metric("Anomalies Detected", anomaly_count)
    st.metric("Anomaly Rate", f"{anomaly_count/len(data)*100:.2f}%")

st.subheader("Sample Data with Anomalies")
st.dataframe(data.head(10))

st.write("""
**Features:**
- Real-time price prediction
- Fare anomaly detection
- Airline comparison
- Booking recommendations
""")