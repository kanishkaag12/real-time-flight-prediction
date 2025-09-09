import joblib
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

# Load the saved model
model = joblib.load("models/best_price_prediction_model.pkl")  # Use correct relative path

# Load the test data
X_test = pd.read_csv("x_test.csv")   # Use correct relative path
y_test = pd.read_csv("y_test.csv")

# Predict directly (no scaling needed)
y_pred = model.predict(X_test)

# Evaluate
rmse = root_mean_squared_error(y_test, y_pred)  # removed squared=False here
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Test RMSE: {rmse}")
print(f"Test R2: {r2}")
print(f"Test MAE: {mae}")
