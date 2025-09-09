from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

model = None

le_airline = None
le_source_city = None
le_departure_time = None
le_stops = None
le_arrival_time = None
le_destination_city = None
le_class = None
scaler = None


def load_model_and_encoders():
    global model, le_airline, le_source_city, le_departure_time, le_stops, le_arrival_time, le_destination_city, le_class, scaler
    
    logger.info("Starting to load model and encoders...")
    
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        flightdata_dir = os.path.abspath(os.path.join(script_dir, '..', 'flightdata'))
        
        # Define paths
        model_path = os.path.join(flightdata_dir, 'scripts', 'models', 'best_price_prediction_model.pkl')
        encoders_dir = os.path.join(flightdata_dir, 'encoders')
        
        logger.info(f"Model path: {model_path}")
        logger.info(f"Encoders directory: {encoders_dir}")
        
        # Verify paths exist
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return False
            
        if not os.path.exists(encoders_dir):
            logger.error(f"Encoders directory not found at: {encoders_dir}")
            return False
        
        # Load model and encoders with validation
        logger.info("Loading model and encoders...")
        
        try:
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
            
            # Load all encoders - using the exact filenames found in the encoders directory
            encoders_to_load = {
                'le_airline.pkl': 'le_airline',
                'le_source_city.pkl': 'le_source_city',
                'le_departure_time.pkl': 'le_departure_time',
                'le_stops.pkl': 'le_stops',
                'le_arrival_time.pkl': 'le_arrival_time',
                'le_destination_city.pkl': 'le_destination_city',
                'le_class.pkl': 'le_class',
                'scaler.pkl': 'scaler'
            }
            
            for file, var_name in encoders_to_load.items():
                file_path = os.path.join(encoders_dir, file)
                if not os.path.exists(file_path):
                    logger.error(f"Encoder file not found: {file_path}")
                    return False
                    
                logger.info(f"Loading {file}...")
                loaded_obj = joblib.load(file_path)
                globals()[var_name] = loaded_obj
                
                # Log encoder classes if it's a label encoder
                if hasattr(loaded_obj, 'classes_'):
                    logger.info(f"{file} classes: {loaded_obj.classes_}")
            
            # Log success
            logger.info("All model and encoders loaded successfully!")
            
            # Log encoder classes for debugging
            if le_airline is not None:
                logger.debug(f"Airline classes: {le_airline.classes_}")
            if le_source_city is not None:
                logger.debug(f"Source city classes: {le_source_city.classes_}")
            if le_departure_time is not None:
                logger.debug(f"Departure time classes: {le_departure_time.classes_}")
            if le_stops is not None:
                logger.debug(f"Stops classes: {le_stops.classes_}")
            if le_arrival_time is not None:
                logger.debug(f"Arrival time classes: {le_arrival_time.classes_}")
            if le_destination_city is not None:
                logger.debug(f"Destination city classes: {le_destination_city.classes_}")
            if le_class is not None:
                logger.debug(f"Class classes: {le_class.classes_}")
                
            return True
            
        except Exception as load_error:
            logger.error(f"Error loading model or encoders: {str(load_error)}", exc_info=True)
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in load_model_and_encoders: {str(e)}", exc_info=True)
        return False


def safe_transform(le, items):
    try:
        known_classes = set(le.classes_)
        default_class = le.classes_[0]  # Use first known class as fallback
        result = []
        
        # Convert items to list if it's a single value
        if not isinstance(items, (list, pd.Series, np.ndarray)):
            items = [items]
            
        for item in items:
            if item in known_classes:
                result.append(le.transform([item])[0])
            else:
                logger.warning(f"Unknown category '{item}' replaced with default '{default_class}'")
                result.append(le.transform([default_class])[0])
        
        # Return a single value if input was a single value
        return result[0] if len(result) == 1 else result
        
    except Exception as e:
        logger.error(f"Error in safe_transform: {str(e)}", exc_info=True)
        # Return default encoding in case of error
        return [le.transform([le.classes_[0]])[0] for _ in items] if len(items) > 1 else le.transform([le.classes_[0]])[0]


def preprocess_input(data):
    """Preprocess input data for prediction.
    
    Args:
        data: Dictionary containing flight details
        
    Returns:
        pd.DataFrame: Processed DataFrame ready for prediction
    """
    try:
        logger.info("\n=== Starting Preprocessing ===")
        logger.info(f"Input data type: {type(data)}")
        logger.info(f"Input data content: {data}")
        
        # Check if data is None or empty
        if data is None:
            logger.error("Error: Input data is None")
            return None
            
        if not isinstance(data, dict):
            logger.error(f"Error: Expected dict, got {type(data)}")
            return None
            
        if not data:
            logger.error("Error: Input data is empty")
            return None
            
        # Log all available encoders with error handling
        logger.info("\n=== Available Encoders ===")
        try:
            logger.info(f"Airline encoder classes: {le_airline.classes_}")
            logger.info(f"Source city encoder classes: {le_source_city.classes_}")
            logger.info(f"Departure time encoder classes: {le_departure_time.classes_}")
            logger.info(f"Stops encoder classes: {le_stops.classes_}")
            logger.info(f"Arrival time encoder classes: {le_arrival_time.classes_}")
            logger.info(f"Destination city encoder classes: {le_destination_city.classes_}")
            logger.info(f"Class encoder classes: {le_class.classes_}")
        except Exception as e:
            logger.error(f"Error logging encoder classes: {str(e)}", exc_info=True)
            return None
            
        # Create a DataFrame from the input data
        try:
            logger.info("\n=== Creating DataFrame from input data ===")
            df = pd.DataFrame([data])
            
            # Ensure all required columns are present
            required_columns = [
                'airline', 'source_city', 'departure_time', 'stops', 'arrival_time',
                'destination_city', 'class', 'duration', 'days_left', 'price'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in input data: {missing_columns}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                return None
                
            logger.info("\nInitial DataFrame:")
            logger.info(df)
            logger.info(f"\nInitial DataFrame columns: {df.columns.tolist()}")
            logger.info(f"Initial DataFrame dtypes: {df.dtypes}")
            
            # Log each column's value with type information
            logger.info("\n=== Input Data Values ===")
            for col in df.columns:
                try:
                    val = df[col].values[0]
                    logger.info(f"{col}: {val} (type: {type(val).__name__})")
                except Exception as e:
                    logger.error(f"Error getting value for column {col}: {str(e)}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error creating DataFrame: {str(e)}", exc_info=True)
            return None
        
        # Encode categorical variables
        logger.info("\n=== Encoding Categorical Variables ===")
        try:
            # Log all available columns before encoding
            logger.info(f"Available columns before encoding: {df.columns.tolist()}")
            
            # Create a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Define the encoding mapping
            encoding_map = {
                'airline': le_airline,
                'source_city': le_source_city,
                'departure_time': le_departure_time,
                'stops': le_stops,
                'arrival_time': le_arrival_time,
                'destination_city': le_destination_city,
                'class': le_class
            }
            
            # Encode each categorical variable
            for col, encoder in encoding_map.items():
                try:
                    logger.info(f"\nEncoding column: {col}")
                    logger.info(f"Original values: {df[col].unique()}")
                    
                    # Check if column exists
                    if col not in df.columns:
                        logger.error(f"Column '{col}' not found in DataFrame")
                        return None
                        
                    # Encode the column
                    encoded_col = f"{col}_encoded"
                    df[encoded_col] = safe_transform(encoder, df[col].astype(str).values)
                    
                    # Check if encoding was successful
                    if df[encoded_col].isna().any():
                        logger.error(f"Encoding failed for column '{col}'. NaN values detected.")
                        return None
                        
                    logger.info(f"Encoded values: {df[encoded_col].unique()}")
                    
                except Exception as e:
                    logger.error(f"Error encoding column '{col}': {str(e)}", exc_info=True)
                    return None
            
            logger.info("\nCategorical variables encoded successfully")
            logger.info("DataFrame after encoding:")
            logger.info(df.head().to_string())
            
        except Exception as e:
            logger.error(f"Error during categorical encoding: {str(e)}", exc_info=True)
            return None
        
        # Feature engineering
        logger.info("\n=== Feature Engineering ===")
        try:
            # Log before adding features
            logger.info("Original columns before feature engineering: %s", df.columns.tolist())
            
            # Add is_weekend feature (1 if Saturday or Sunday, else 0)
            df['is_weekend'] = 0
            logger.info("Added is_weekend feature")
            
            # Add is_peak feature (1 if days_left <= 7, else 0)
            if 'days_left' not in df.columns:
                logger.error("'days_left' column not found for feature engineering")
                return None
                
            df['is_peak'] = (df['days_left'] <= 7).astype(int)
            logger.info("Added is_peak feature")
            
            # Add competition factor (simplified)
            df['competition_factor'] = df['days_left'] * 0.1
            logger.info("Added competition_factor feature")
            
            # Log the engineered features
            logger.info("\n=== Engineered Features ===")
            logger.info("is_weekend values: %s", df['is_weekend'].unique())
            logger.info("is_peak values: %s", df['is_peak'].unique())
            logger.info("competition_factor values: %s", df['competition_factor'].unique())
            
            logger.info("\n=== DataFrame After Feature Engineering ===")
            logger.info("Columns: %s", df.columns.tolist())
            logger.info("Shape: %s", df.shape)
            logger.info("\nSample data after feature engineering:")
            logger.info(df[['days_left', 'is_weekend', 'is_peak', 'competition_factor']].to_string())
            
        except Exception as e:
            logger.error("Error in feature engineering: %s", str(e), exc_info=True)
            return None
        
        # Get the feature names from the scaler (these are the expected column names)
        try:
            feature_order = list(scaler.feature_names_in_)
            print(f"\nScaler expects features in this order: {feature_order}")
        except AttributeError:
            # Fallback to default order if feature_names_in_ is not available
            feature_order = [
                'airline_encoded', 'source_city_encoded', 'departure_time_encoded',
                'stops_encoded', 'arrival_time_encoded', 'destination_city_encoded',
                'class_encoded', 'duration', 'days_left', 'is_weekend', 'is_peak',
                'competition_factor'
            ]
            print("\nUsing default feature order for scaling")
        
        print("\nPreparing features for scaling...")
        
        # Create a mapping from our current column names to the expected ones
        current_columns = df.columns.tolist()
        print(f"Current columns: {current_columns}")
        
        # Ensure all required columns are present
        required_columns = [
            'airline_encoded', 'source_city_encoded', 'departure_time_encoded',
            'stops_encoded', 'arrival_time_encoded', 'destination_city_encoded',
            'class_encoded', 'duration', 'days_left', 'is_weekend', 'is_peak',
            'competition_factor'
        ]
        
        missing_columns = [col for col in required_columns if col not in current_columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None
            
        # Create a DataFrame with the exact feature order expected by the scaler
        X = df[required_columns].copy()
        
        # Ensure all columns are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill any remaining NaN values with 0
        X = X.fillna(0)
        print("\nFinal features for prediction:")
        for col in X.columns:
            print(f"- {col}: {X[col].values[0]} (type: {type(X[col].values[0])})")
        
        # Scale features
        print("\nScaling features...")
        try:
            X_scaled = scaler.transform(X)
            print("Feature scaling completed")
            return X_scaled
        except Exception as e:
            print(f"Error scaling features: {str(e)}")
            print("Feature values:", X.values)
            raise
            
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/')
def home():
    return jsonify({
        "message": "Flight Price Prediction API",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Make price prediction",
            "/health": "GET - Check API health"
        }
    })


@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("\n=== New Prediction Request ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content type: {request.content_type}")
        
        if model is None or le_airline is None or scaler is None:
            error_msg = 'Model or encoders not loaded. Please check server logs.'
            logger.error(error_msg)
            return jsonify({'error': error_msg, 'type': 'initialization_error'}), 500
        
        try:
            # Try to parse JSON data
            try:
                data = request.get_json()
                logger.info("Successfully parsed JSON data")
            except Exception as e:
                error_msg = f'Error parsing JSON data: {str(e)}'
                logger.error(error_msg, exc_info=True)
                return jsonify({'error': error_msg, 'type': 'invalid_json'}), 400
                
            if data is None:
                error_msg = 'No JSON data received in request or empty JSON'
                logger.error(error_msg)
                return jsonify({'error': error_msg, 'type': 'invalid_input'}), 400
                
            logger.info(f"Parsed input data: {data}")
            
            # Log each field in the input data
            for key, value in data.items():
                logger.info(f"Input field - {key}: {value} (type: {type(value).__name__})")
            
            # Validate required fields
            required_fields = [
                'airline', 'source_city', 'departure_time', 'stops',
                'arrival_time', 'destination_city', 'class', 'duration', 'days_left'
            ]
            
            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                error_msg = f'Missing required fields: {missing_fields}'
                print(error_msg)
                return jsonify({
                    'error': error_msg,
                    'type': 'missing_fields',
                    'missing_fields': missing_fields
                }), 400
            
            # Add default price for competition factor calculation
            if 'price' not in data:
                data['price'] = 0
                print("Added default price for competition factor calculation")
            
            print("\nStarting preprocessing...")
            logger.info("\n=== Starting Preprocessing ===")
            try:
                processed_data = preprocess_input(data)
                logger.info("\n=== Preprocessing completed successfully ===")
                logger.info("Processed data columns: %s", processed_data.columns.tolist())
                logger.info("Processed data dtypes: \n%s", processed_data.dtypes)
                
                # Log the first few rows of processed data
                logger.info("\nProcessed data sample (first row):")
                for col in processed_data.columns:
                    logger.info(f"{col}: {processed_data[col].values[0]}")
                
                # Ensure we have the expected columns for scaling
                expected_columns = [
                    'airline_encoded', 'source_city_encoded', 'departure_time_encoded',
                    'stops_encoded', 'arrival_time_encoded', 'destination_city_encoded',
                    'class_encoded', 'duration', 'days_left', 'is_weekend', 'is_peak',
                    'competition_factor'
                ]
                
                missing_cols = [col for col in expected_columns if col not in processed_data.columns]
                if missing_cols:
                    error_msg = f'Missing expected columns after preprocessing: {missing_cols}'
                    logger.error(error_msg)
                    logger.error("Available columns: %s", processed_data.columns.tolist())
                    return jsonify({'error': error_msg, 'type': 'preprocessing_error'}), 500
                
                # Make prediction
                try:
                    logger.info("\n=== Making prediction ===")
                    logger.info("Input data shape: %s", processed_data.shape)
                    logger.info("Input data sample (first row):")
                    logger.info(processed_data.iloc[0].to_dict())
                    
                    prediction = model.predict(processed_data)
                    logger.info("\n=== Prediction successful ===")
                    logger.info("Prediction result shape: %s", prediction.shape)
                    logger.info("Prediction values: %s", prediction)
                    
                    # Convert numpy array to list for JSON serialization
                    prediction_list = prediction.tolist()
                    
                    return jsonify({
                        'predicted_price': float(prediction_list[0]) if hasattr(prediction, '__len__') else float(prediction),
                        'status': 'success',
                        'message': 'Prediction successful'
                    })
                    
                except Exception as e:
                    error_msg = f'Error during prediction: {str(e)}'
                    logger.error("\n=== Prediction failed ===")
                    logger.error(error_msg, exc_info=True)
                    logger.error("Input data that caused the error:")
                    logger.error(processed_data)
                    return jsonify({'error': error_msg, 'type': 'prediction_error'}), 500
                    
            except Exception as e:
                error_msg = f'Error during data preprocessing: {str(e)}'
                logger.error("\n=== Preprocessing failed ===")
                logger.error(error_msg, exc_info=True)
                logger.error("Input data that caused the error:")
                logger.error(data)
                return jsonify({'error': error_msg, 'type': 'preprocessing_error'}), 400
                
        except Exception as request_error:
            error_msg = f'Request processing error: {str(request_error)}'
            logger.error(error_msg, exc_info=True)
            return jsonify({
                'error': error_msg,
                'type': 'request_processing_error',
                'traceback': traceback.format_exc() if app.debug else None
            }), 400
            
    except Exception as unexpected_error:
        error_msg = f'Unexpected error: {str(unexpected_error)}'
        print(error_msg)
        import traceback
        print("Unexpected error traceback:", traceback.format_exc())
        return jsonify({
            'error': 'An unexpected error occurred',
            'type': 'unexpected_error',
            'details': str(unexpected_error),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500


if __name__ == '__main__':
    print("Starting Flight Price Prediction API...")
    if load_model_and_encoders():
        print("Model and encoders loaded successfully!")
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model or encoders. Exiting...")
