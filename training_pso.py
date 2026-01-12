#training_pso.py
import numpy as np
import pandas as pd
import joblib
import pyswarms as ps

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ===============================
# 1. Load dataset
# ===============================
data = pd.read_csv("delhi_metro_updated2.0.csv")

data = data.head(5000)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

print("Dataset shape:", data.shape)

# ===============================
# 2. Select features, target, objectives
# ===============================
X = data[['Distance_km', 'Fare', 'Cost_per_passenger']]
y = data['Passengers']

distance = data['Distance_km'].values
fare = data['Fare'].values

feature_names = X.columns.tolist()

# ===============================
# 3. Train-test split
# ===============================
X_train, X_test, y_train, y_test, dist_train, dist_test, fare_train, fare_test = train_test_split(
    X, y, distance, fare, test_size=0.2, random_state=42
)

# ===============================
# 4. Feature scaling
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 5. Multi-objective fitness function
# ===============================
def fitness_function(weights):
    if weights.size == 0:
        return np.array([np.inf])

    w = weights[:, :-1]
    b = weights[:, -1].reshape(1, -1)

    y_pred = X_train_scaled @ w.T + b

    mse = np.mean((y_train.values.reshape(-1, 1) - y_pred) ** 2, axis=0)

    distance_penalty = np.mean(dist_train)
    fare_penalty = np.mean(fare_train)

    w1, w2, w3 = 0.6, 0.2, 0.2

    fitness = w1 * mse + w2 * distance_penalty + w3 * fare_penalty
    return fitness

# ===============================
# 6. PSO Optimization
# ===============================
dimensions = X_train_scaled.shape[1] + 1

options = {"c1": 1.5, "c2": 1.5, "w": 0.7}
bounds = (np.array([-10]*dimensions), np.array([10]*dimensions))

optimizer = ps.single.GlobalBestPSO(
    n_particles=20,
    dimensions=dimensions,
    options=options,
    bounds=bounds
)

best_cost, best_position = optimizer.optimize(fitness_function, iters=50)

# ===============================
# 7. Extract best parameters
# ===============================
best_weights = best_position[:-1]
best_bias = best_position[-1]

# ===============================
# 8. Evaluate model
# ===============================
y_test_pred = X_test_scaled @ best_weights + best_bias
test_mse = mean_squared_error(y_test, y_test_pred)

print("\n✅ Training Complete")
print("Test MSE:", test_mse)

# ===============================
# 9. Save model and scaler
# ===============================
pso_model = {
    "model_type": "Multi-Objective PSO Optimized Linear Regression",
    "weights": best_weights,
    "bias": best_bias,
    "feature_names": feature_names
}

joblib.dump(pso_model, "pso_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Saved files:")
print("pso_model.pkl")
print("scaler.pkl")
