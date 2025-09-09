# ==================== STEP 5: ANALYZE DETECTED ANOMALIES ====================
print("="*60)
print("STEP 5: ANALYZING DETECTED ANOMALIES")
print("="*60)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data with anomalies
data = pd.read_csv('flight_data_with_anomalies.csv')

# Separate normal and anomalous flights
normal_flights = data[data['is_anomaly'] == 0]
anomalous_flights = data[data['is_anomaly'] == 1]

print("📊 ANOMALY ANALYSIS REPORT")
print("="*40)
print(f"Total flights: {len(data)}")
print(f"Normal flights: {len(normal_flights)}")
print(f"Anomalous flights: {len(anomalous_flights)}")
print(f"Anomaly rate: {len(anomalous_flights)/len(data)*100:.2f}%")

print("\n💰 PRICE ANALYSIS:")
print(f"Normal price range: ₹{normal_flights['price'].min():.2f} to ₹{normal_flights['price'].max():.2f}")
print(f"Anomalous price range: ₹{anomalous_flights['price'].min():.2f} to ₹{anomalous_flights['price'].max():.2f}")
print(f"Avg normal price: ₹{normal_flights['price'].mean():.2f}")
print(f"Avg anomalous price: ₹{anomalous_flights['price'].mean():.2f}")

print("\n🏢 AIRLINE ANOMALY DISTRIBUTION:")
airline_anomalies = data.groupby('airline')['is_anomaly'].mean().sort_values(ascending=False)
print(airline_anomalies)

# Save analysis report
with open('anomaly_analysis_report.txt', 'w') as f:
    f.write("FLIGHT PRICE ANOMALY ANALYSIS REPORT\n")
    f.write("="*50 + "\n")
    f.write(f"Total anomalous flights: {len(anomalous_flights)}\n")
    f.write(f"Anomaly rate: {len(anomalous_flights)/len(data)*100:.2f}%\n")
    f.write(f"Average normal price: {normal_flights['price'].mean():.4f}\n")
    f.write(f"Average anomalous price: {anomalous_flights['price'].mean():.4f}\n")
    
print("💾 Analysis report saved as 'anomaly_analysis_report.txt'")
print("="*60)