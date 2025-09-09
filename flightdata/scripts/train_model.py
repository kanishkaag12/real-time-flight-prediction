import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the saved model
model = joblib.load("models/best_price_prediction_model.pkl")

# Load training data
X_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")

# Predict on training set
y_train_pred = model.predict(X_train)

# Calculate RMSE manually by taking square root of MSE
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

print(f"Train RMSE: {train_rmse}")
print(f"Train R2: {train_r2}")
print(f"Train MAE: {train_mae}")
