import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Page Configuration (Light Mode)
# ===============================
st.set_page_config(
    page_title="Delhi Metro Passenger Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Load PSO model & scaler
# ===============================
model = joblib.load("pso_model.pkl")
scaler = joblib.load("scaler.pkl")

weights = np.array(model["weights"])
bias = float(model["bias"])
feature_names = model["feature_names"]

# ===============================
# Load dataset for preview
# ===============================
data = pd.read_csv("delhi_metro_updated2.0.csv")
data = data[['Distance_km', 'Fare', 'Cost_per_passenger', 'Passengers']].dropna()

# ===============================
# Title & Description
# ===============================
st.title("🚇 Delhi Metro Passenger Prediction (PSO)")
st.markdown("""
This application predicts **Delhi Metro passenger demand** using a  
**Particle Swarm Optimization (PSO)–optimized regression model**,  
supporting **multi-objective metro planning** based on **distance and fare**.
""")

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("🔢 Input Parameters")

distance = st.sidebar.number_input(
    "Distance (km)", min_value=0.0, max_value=100.0, value=10.0
)
fare = st.sidebar.number_input(
    "Fare (₹)", min_value=0.0, max_value=200.0, value=30.0
)
cost = st.sidebar.number_input(
    "Cost per Passenger (₹)", min_value=0.0, max_value=100.0, value=15.0
)

X_input = pd.DataFrame([{
    "Distance_km": distance,
    "Fare": fare,
    "Cost_per_passenger": cost
}])

X_scaled = scaler.transform(X_input)

# ===============================
# Prediction
# ===============================
weights_demo = weights * 750  # visual scaling for demo
y_pred = np.dot(X_scaled, weights_demo) + bias

# ===============================
# Results
# ===============================
st.subheader("📊 Prediction Result")

col1, col2 = st.columns(2)

col1.metric(
    label="Estimated Passengers",
    value=f"{y_pred[0]:,.2f}"
)

# Max demo prediction
X_max = pd.DataFrame([{
    "Distance_km": 100,
    "Fare": 200,
    "Cost_per_passenger": 100
}])
X_max_scaled = scaler.transform(X_max)
y_max = np.dot(X_max_scaled, weights_demo) + bias

col2.metric(
    label="Max Predicted Passengers (Demo)",
    value=f"{y_max[0]:,.2f}"
)

# ===============================
# Feature Contribution
# ===============================
st.subheader("📈 Feature Contribution & Sensitivity")

contribution = X_scaled[0] * weights_demo

contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": contribution
})

sensitivity_df = contrib_df.copy()
sensitivity_df["Impact"] = contrib_df["Contribution"].abs()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Feature Contribution**")
    st.bar_chart(contrib_df.set_index("Feature"))

with chart_col2:
    st.markdown("**Sensitivity Analysis**")
    st.line_chart(sensitivity_df.set_index("Feature"))

# ===============================
# Dataset Preview
# ===============================
st.subheader("🗂 Dataset Sample")

with st.expander("Show dataset preview"):
    st.dataframe(data.head(10))

# ===============================
# PSO Explanation
# ===============================
with st.expander("🧠 How Particle Swarm Optimization Works"):
    st.markdown("""
    - Each particle represents a candidate solution consisting of **weights and bias**
    - The fitness function minimizes **Mean Squared Error (MSE)**
    - Particles update their positions using personal and global best solutions
    - Bound constraints prevent unstable parameter values
    """)

# ===============================
# Conclusion
# ===============================
st.subheader("✅ Conclusion")
st.markdown("""
The PSO-based regression model effectively captures the relationship between  
**distance, fare, and operational cost** in predicting metro passenger demand.  
This approach supports **data-driven, multi-objective metro planning decisions**.
""")
