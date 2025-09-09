# ==================== STEP 3: BUILD PRICE PREDICTION MODELS ====================
print("="*70)
print("STEP 3: BUILDING PRICE PREDICTION MODELS")
print("="*70)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the split datasets
print("📁 Loading training and testing data...")
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

print(f"✅ Data loaded successfully!")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Convert y_train and y_test to Series (they were saved as DataFrames)
y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

print("\n🧠 Training models...")
results = {}
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'MAE': mae, 
        'RMSE': rmse, 
        'R²': r2,
        'model': model,
        'predictions': y_pred
    }
    
    print(f"✅ Training completed!")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R² Score: {r2:.4f}")

# Compare model performance
print("\n" + "="*50)
print("MODEL PERFORMANCE COMPARISON")
print("="*50)
best_model_name = None
best_r2 = -float('inf')

for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"   MAE: {metrics['MAE']:.4f}")
    print(f"   RMSE: {metrics['RMSE']:.4f}")
    print(f"   R²: {metrics['R²']:.4f}")
    print()
    
    if metrics['R²'] > best_r2:
        best_r2 = metrics['R²']
        best_model_name = model_name

print(f"🏆 BEST MODEL: {best_model_name} (R² = {best_r2:.4f})")

# Save the best model
best_model = results[best_model_name]['model']
joblib.dump(best_model, 'best_price_prediction_model.pkl')
print(f"💾 Best model saved as 'best_price_prediction_model.pkl'")

print("="*70)