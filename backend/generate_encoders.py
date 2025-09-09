import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def generate_encoders():
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        flightdata_dir = os.path.join(script_dir, '..', 'flightdata')
        data_file = os.path.join(flightdata_dir, 'data', 'Clean_Dataset.csv')
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(flightdata_dir, 'encoders')
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the dataset
        print(f"Loading dataset from {data_file}...")
        df = pd.read_csv(data_file)
        
        # Initialize encoders
        encoders = {
            'airline': LabelEncoder(),
            'source_city': LabelEncoder(),
            'departure_time': LabelEncoder(),
            'stops': LabelEncoder(),
            'arrival_time': LabelEncoder(),
            'destination_city': LabelEncoder(),
            'class': LabelEncoder()
        }
        
        # Fit encoders
        print("Fitting encoders...")
        for col, encoder in encoders.items():
            df[col] = encoder.fit_transform(df[col])
            
            # Save each encoder
            encoder_file = os.path.join(output_dir, f'le_{col}.pkl')
            joblib.dump(encoder, encoder_file)
            print(f"Saved {encoder_file}")
        
        # Initialize and fit the scaler
        print("Fitting scaler...")
        numeric_features = ['duration', 'days_left', 'price']
        scaler = StandardScaler()
        scaler.fit(df[numeric_features])
        
        # Save the scaler
        scaler_file = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_file)
        print(f"Saved {scaler_file}")
        
        print("\nEncoder generation completed successfully!")
        print(f"Encoders and scaler saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error generating encoders: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    generate_encoders()
