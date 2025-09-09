""" import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

class PricePredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.le_airline = joblib.load('le_airline.pkl')
        self.le_source = joblib.load('le_source.pkl')
        self.le_destination = joblib.load('le_destination.pkl')
        self.le_class = joblib.load('le_class.pkl')
        self.le_stops = joblib.load('le_stops.pkl')
        self.le_departure_time = joblib.load('le_departure_time.pkl')
        self.le_arrival_time = joblib.load('le_arrival_time.pkl')
        self.scaler = joblib.load('scaler.pkl')

    def preprocess(self, features_dict):
        df = pd.DataFrame([features_dict])
        df['airline_encoded'] = self.le_airline.transform([df['airline'].iloc[0]])
        df['source_city_encoded'] = self.le_source.transform([df['source_city'].iloc[0]])
        df['destination_city_encoded'] = self.le_destination.transform([df['destination_city'].iloc[0]])
        df['class_encoded'] = self.le_class.transform([df['class'].iloc[0]])
        df['stops_encoded'] = self.le_stops.transform([df['stops'].iloc[0]])
        df['departure_time_encoded'] = self.le_departure_time.transform([df['departure_time'].iloc[0]])
        df['arrival_time_encoded'] = self.le_arrival_time.transform([df['arrival_time'].iloc[0]])

        df['is_weekend'] = df['days_left'].apply(lambda x: 1 if (x % 7 in [0, 6]) else 0)
        df['is_peak'] = df['departure_time'].apply(lambda x: 1 if x in ['Morning', 'Evening'] else 0)

        df['competition_factor'] = 0
        if 'price' in df.columns:
            df['competition_factor'] = df.groupby(['source_city', 'destination_city', 'class', 'stops'])['price'].transform('mean')

        features = [
            'duration', 'days_left', 'airline_encoded', 'source_city_encoded',
            'destination_city_encoded', 'class_encoded', 'stops_encoded',
            'departure_time_encoded', 'arrival_time_encoded', 'is_weekend',
            'is_peak', 'competition_factor'
        ]
        X = df[features]

        return self.scaler.transform(X)

    def predict(self, features_dict):
        X = self.preprocess(features_dict)
        return self.model.predict(X)[0]


class AnomalyDetector:
    def __init__(self, model_path, is_nn=False, preprocessor=None):
        self.is_nn = is_nn
        self.preprocessor = preprocessor

        if is_nn:
            self.model = load_model(model_path)
            self.threshold = joblib.load(f"{model_path.split('.')[0]}_threshold.pkl")
        else:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load('scaler.pkl')

    def detect(self, features_dict):
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be passed to AnomalyDetector")

        X = self.preprocessor.preprocess(features_dict)

        if self.is_nn:
            recon_error = np.mean(np.power(X - self.model.predict(X), 2), axis=1)
            return recon_error > self.threshold  # True if anomaly
        else:
            return self.model.predict(X) == -1  # True if anomaly


# Example usage:
if __name__ == "__main__":
    sample_features = {
        'airline': 'Vistara',
        'source_city': 'Delhi',
        'destination_city': 'Mumbai',
        'class': 'Economy',
        'stops': 'zero',
        'departure_time': 'Morning',
        'arrival_time': 'Afternoon',
        'duration': 2.25,
        'days_left': 1
    }

    predictor = PricePredictor('xgboost.pkl')
    predicted_price = predictor.predict(sample_features)
    print(f'Predicted Price: {predicted_price}')

    detector = AnomalyDetector('dbscan.pkl', preprocessor=predictor)
    anomaly_flag = detector.detect(sample_features)
    print(f'Anomaly Detected: {anomaly_flag}')
 """


import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from config import *

logger = logging.getLogger(__name__)

class PricePredictor:
    """Enhanced price prediction wrapper"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load prediction model and preprocessor"""
        try:
            # Load the trained model
            self.model = joblib.load(model_path)
            
            # Load the preprocessor
            preprocessor_path = Path(__file__).parent / 'preprocessor.pkl'
            if preprocessor_path.exists():
                from preprocessing import FlightDataPreprocessor
                self.preprocessor = FlightDataPreprocessor()
                self.preprocessor.load(preprocessor_path)
            else:
                logger.warning("Preprocessor not found. Using basic preprocessing.")
                
            self.is_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_input(self, features_dict):
        """Preprocess input features for prediction"""
        try:
            df = pd.DataFrame([features_dict])
            
            if self.preprocessor and self.preprocessor.is_fitted:
                X = self.preprocessor.transform(df)
            else:
                logger.warning("Using fallback preprocessing")
                X = self._basic_preprocess(df)
            
            return X
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
    
    def _basic_preprocess(self, df):
        """Basic preprocessing fallback"""
        # Simplified mapping
        categorical_mappings = {
            'airline': {'AirAsia': 0, 'Air_India': 1, 'Vistara': 2, 'GO_FIRST': 3,
                        'Indigo': 4, 'SpiceJet': 5},
            'source_city': {'Bangalore': 0, 'Chennai': 1, 'Delhi': 2, 'Hyderabad': 3,
                            'Kolkata': 4, 'Mumbai': 5},
            'destination_city': {'Bangalore': 0, 'Chennai': 1, 'Delhi': 2, 'Hyderabad': 3,
                                 'Kolkata': 4, 'Mumbai': 5},
            'class': {'Economy': 0, 'Business': 1},
            'departure_time': {'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2,
                               'Evening': 3, 'Night': 4, 'Late_Night': 5},
            'arrival_time': {'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2,
                             'Evening': 3, 'Night': 4, 'Late_Night': 5}
        }
        
        processed_features = []
        
        for feature, mapping in categorical_mappings.items():
            if feature in df.columns:
                processed_features.append(df[feature].map(mapping).fillna(0).values[0])
        
        # Stops encoding
        if 'stops' in df.columns:
            stops_val = STOPS_MAPPING.get(df['stops'].iloc[0], 0)
            processed_features.append(stops_val)
        
        # Numerical features
        for feature in ['duration', 'days_left']:
            if feature in df.columns:
                processed_features.append(df[feature].iloc[0])
        
        return np.array(processed_features).reshape(1, -1)
    
    def predict(self, features_dict):
        """Make price prediction"""
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            X = self.preprocess_input(features_dict)
            prediction = self.model.predict(X)[0]
            
            logger.info(f"Prediction made: {prediction}")
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_batch(self, features_list):
        """Make batch predictions"""
        predictions = []
        for features in features_list:
            try:
                pred = self.predict(features)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                predictions.append(None)
        
        return predictions

class AnomalyDetector:
    """Enhanced anomaly detection wrapper"""
    
    def __init__(self, model_path=None, threshold_path=None, preprocessor=None):
        self.model = None
        self.threshold = None
        self.preprocessor = preprocessor
        self.is_nn = False
        self.is_loaded = False
        
        if model_path:
            self.load_model(model_path, threshold_path)
    
    def load_model(self, model_path, threshold_path=None):
        """Load anomaly detection model"""
        try:
            if str(model_path).endswith('.h5'):
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                self.is_nn = True
                
                if threshold_path:
                    self.threshold = joblib.load(threshold_path)
                else:
                    threshold_file = str(model_path).replace('.h5', '_threshold.pkl')
                    if Path(threshold_file).exists():
                        self.threshold = joblib.load(threshold_file)
            else:
                self.model = joblib.load(model_path)
                self.is_nn = False
            
            self.is_loaded = True
            logger.info(f"Anomaly model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading anomaly model: {e}")
            raise
    
    def detect(self, features_dict):
        """Detect if input is anomalous"""
        if not self.is_loaded:
            raise ValueError("Anomaly model not loaded.")
        
        if not self.preprocessor:
            raise ValueError("Preprocessor required for anomaly detection.")
        
        try:
            X = self.preprocessor.preprocess_input(features_dict)
            
            if self.is_nn:
                reconstruction = self.model.predict(X, verbose=0)
                mse = np.mean(np.power(X - reconstruction, 2), axis=1)
                is_anomaly = mse[0] > self.threshold if self.threshold else False
            else:
                prediction = self.model.predict(X)
                is_anomaly = prediction[0] == -1
            
            logger.info(f"Anomaly detection result: {is_anomaly}")
            return bool(is_anomaly)
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return False  # Default to non-anomalous on error

class FlightPricePipeline:
    """Complete prediction and anomaly detection pipeline"""
    
    def __init__(self, price_model_path=None, anomaly_model_path=None):
        self.price_predictor = None
        self.anomaly_detector = None
        
        if price_model_path:
            self.load_models(price_model_path, anomaly_model_path)
    
    def load_models(self, price_model_path, anomaly_model_path=None):
        """Load both prediction and anomaly detection models"""
        try:
            self.price_predictor = PricePredictor(price_model_path)
            
            if anomaly_model_path:
                self.anomaly_detector = AnomalyDetector(
                    anomaly_model_path,
                    preprocessor=self.price_predictor
                )
            
            logger.info("Pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            raise
    
    def predict_with_anomaly_check(self, features_dict):
        """Make prediction and check for anomalies"""
        result = {
            'predicted_price': None,
            'is_anomaly': False,
            'confidence': 'high',
            'error': None
        }
        
        try:
            if self.price_predictor:
                result['predicted_price'] = self.price_predictor.predict(features_dict)
            
            if self.anomaly_detector:
                result['is_anomaly'] = self.anomaly_detector.detect(features_dict)
                if result['is_anomaly']:
                    result['confidence'] = 'low'
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Pipeline prediction error: {e}")
        
        return result

# Testing example
if __name__ == "__main__":
    sample_features = {
        'airline': 'AirAsia',
        'source_city': 'Delhi',
        'destination_city': 'Mumbai',
        'class': 'Economy',
        'stops': 'zero',
        'departure_time': 'Morning',
        'arrival_time': 'Afternoon',
        'duration': 2.25,
        'days_left': 7
    }
    
    try:
        predictor = PricePredictor()
        print("Model classes defined successfully!")
    except Exception as e:
        print(f"Error in testing: {e}")
