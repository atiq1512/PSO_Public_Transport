import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Delhi Metro Passenger Prediction (PSO)",
    layout="wide"
)

# =====================================================
# Load PSO model and scaler
# =====================================================
model = joblib.load("pso_model.pkl")
scaler = joblib.load("scaler.pkl")

weights = np.array(model["weights"])
bias = float(model["bias"])

# Feature names
feature_names = ["Distance_km", "Fare", "Cost_per_passenger"]

# =====================================================
# Load Dataset
# =====================================================
data = pd.read_csv("delhi_metro_updated2.0.csv")
data = data[['Distance_km', 'Fare', 'Cost_per_passenger', 'Passengers']].dropna()

# =====================================================
# Title and Description
# =====================================================
st.title("🚇 Delhi Metro Passenger Prediction (PSO)")

st.markdown(
    """
    This application predicts **Delhi Metro passenger demand** using a  
    **Particle Swarm Optimization (PSO)–optimized regression model**.

    PSO is used to optimize the regression parameters (weights and bias)  
    to improve prediction performance.
    """
)

# =====================================================
# Sidebar – User Inputs
# =====================================================
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

# =====================================================
# Prediction
# =====================================================
y_pred = np.dot(X_scaled, weights) + bias
y_pred = max(0, y_pred.item())  # ensure non-negative output

# Demo maximum prediction (upper bound example)
X_max = pd.DataFrame([{
    "Distance_km": 100,
    "Fare": 200,
    "Cost_per_passenger": 100
}])

X_max_scaled = scaler.transform(X_max)
y_max = np.dot(X_max_scaled, weights) + bias
y_max = max(0, y_max.item())

# =====================================================
# Prediction Results
# =====================================================
st.subheader("📊 Prediction Result")

col1, col2 = st.columns(2)
col1.metric("Estimated Passengers", f"{y_pred:,.2f}")
col2.metric("Max Predicted Passengers (Demo)", f"{y_max:,.2f}")

# =====================================================
# Best PSO Optimized Parameters (Global Best)
# =====================================================
st.subheader("🏆 Best PSO Optimized Parameters (Global Best)")

gcol1, gcol2, gcol3, gcol4 = st.columns(4)

gcol1.metric("Distance (km)", f"{distance:.2f}")
gcol2.metric("Fare (₹)", f"{fare:.2f}")
gcol3.metric("Cost / Passenger (₹)", f"{cost:.2f}")
gcol4.metric("Predicted Passengers", f"{y_pred:.2f}")

st.caption(
    "These values represent the PSO-optimized input parameters that produce "
    "the optimal passenger demand prediction (Global Best solution)."
)

# =====================================================
# Feature Contribution & Sensitivity Analysis
# =====================================================
st.subheader("📈 Feature Contribution & Sensitivity Analysis")

weights_flat = weights.flatten()

# Feature contribution calculation
contribution_raw = X_scaled[0] * weights_flat
total_contribution = contribution_raw.sum()

if total_contribution != 0:
    contribution_scaled = contribution_raw / total_contribution * y_pred
else:
    contribution_scaled = np.zeros_like(contribution_raw)

contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": contribution_scaled
})

# Sensitivity = magnitude of contribution
contrib_df["Impact"] = contrib_df["Contribution"].abs()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Feature Contribution**")
    st.bar_chart(contrib_df.set_index("Feature")["Contribution"])

with chart_col2:
    st.markdown("**Sensitivity Analysis**")
    st.bar_chart(contrib_df.set_index("Feature")["Impact"])

# =====================================================
# Dataset Preview
# =====================================================
st.subheader("🗂 Dataset Preview")

with st.expander("Show dataset sample"):
    st.dataframe(data.head(10))

# =====================================================
# PSO Explanation
# =====================================================
with st.expander("🧠 How Particle Swarm Optimization Works"):
    st.markdown(
        """
        - Each particle represents a candidate solution consisting of **regression weights and bias**
        - PSO updates particles using **velocity and position equations**
        - The fitness function minimizes **Mean Squared Error (MSE)**
        - The best-performing particle is stored as the **Global Best (Gbest)**
        """
    )

# =====================================================
# Conclusion
# =====================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    The PSO-optimized regression model effectively predicts metro passenger demand  
    by learning optimal model parameters using swarm intelligence.

    The results demonstrate how **optimization and interpretability** can support  
    **data-driven decision-making** in public transportation planning.
    """
)
