# ==================== STEP 4: ANOMALY DETECTION ====================
print("="*60)
print("STEP 4: IMPLEMENTING FARE ANOMALY DETECTION")
print("="*60)

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the standardized data
print("📁 Loading standardized data...")
std_data = pd.read_csv('flight_data_standardized.csv')
print(f"Data shape: {std_data.shape}")

# Select features for anomaly detection (price + key features)
X_anomaly = std_data[['price', 'duration', 'days_left']]

# Initialize anomaly detection models
anomaly_models = {
    'Isolation Forest': IsolationForest(contamination=0.05, random_state=42),
    'One-Class SVM': OneClassSVM(nu=0.05),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=10)
}

print("\n🔍 Detecting anomalies...")
anomaly_results = {}

for name, model in anomaly_models.items():
    print(f"\n--- Using {name} ---")
    
    try:
        if name == 'DBSCAN':
            anomalies = model.fit_predict(X_anomaly)
            n_anomalies = sum(anomalies == -1)
        else:
            anomalies = model.fit_predict(X_anomaly)
            n_anomalies = sum(anomalies == -1)
        
        anomaly_results[name] = {
            'n_anomalies': n_anomalies,
            'model': model,
            'anomalies': anomalies
        }
        
        print(f"✅ Detected {n_anomalies} anomalies ({n_anomalies/len(std_data)*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Error with {name}: {e}")

# Use the best approach (Isolation Forest typically works best)
best_anomaly_model = IsolationForest(contamination=0.05, random_state=42)
std_data['is_anomaly'] = best_anomaly_model.fit_predict(X_anomaly)
std_data['is_anomaly'] = std_data['is_anomaly'].map({1: 0, -1: 1})  # 0=normal, 1=anomaly

print(f"\n🏆 Final anomaly detection:")
print(f"Total anomalies detected: {std_data['is_anomaly'].sum()}")
print(f"Anomaly rate: {std_data['is_anomaly'].mean()*100:.2f}%")

# Save results
std_data.to_csv('flight_data_with_anomalies.csv', index=False)
joblib.dump(best_anomaly_model, 'anomaly_detection_model.pkl')

print("💾 Anomaly data saved as 'flight_data_with_anomalies.csv'")
print("💾 Anomaly model saved as 'anomaly_detection_model.pkl'")

# Visualize anomalies
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
normal = std_data[std_data['is_anomaly'] == 0]
anomalous = std_data[std_data['is_anomaly'] == 1]
plt.scatter(normal['duration'], normal['price'], alpha=0.5, label='Normal', c='blue')
plt.scatter(anomalous['duration'], anomalous['price'], alpha=0.8, label='Anomaly', c='red')
plt.xlabel('Duration')
plt.ylabel('Price')
plt.legend()
plt.title('Anomalies: Duration vs Price')

plt.subplot(1, 3, 2)
plt.scatter(normal['days_left'], normal['price'], alpha=0.5, label='Normal', c='blue')
plt.scatter(anomalous['days_left'], anomalous['price'], alpha=0.8, label='Anomaly', c='red')
plt.xlabel('Days Left')
plt.ylabel('Price')
plt.legend()
plt.title('Anomalies: Days Left vs Price')

plt.subplot(1, 3, 3)
plt.hist([normal['price'], anomalous['price']], 
         bins=30, alpha=0.7, label=['Normal', 'Anomaly'], color=['blue', 'red'])
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.legend()
plt.title('Price Distribution: Normal vs Anomalous')

plt.tight_layout()
plt.savefig('anomaly_detection_visualization.png')
plt.show()

print("="*60)