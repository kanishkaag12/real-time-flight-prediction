from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime

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
    try:
        model = joblib.load('../flightdata/xgboost.pkl')
        le_airline = joblib.load('../flightdata/le_airline.pkl')
        le_source_city = joblib.load('../flightdata/le_source.pkl')
        le_departure_time = joblib.load('../flightdata/le_departure_time.pkl')
        le_stops = joblib.load('../flightdata/le_stops.pkl')
        le_arrival_time = joblib.load('../flightdata/le_arrival_time.pkl')
        le_destination_city = joblib.load('../flightdata/le_destination.pkl')
        le_class = joblib.load('../flightdata/le_class.pkl')
        scaler = joblib.load('../flightdata/scaler.pkl')

        # Print encoder classes for debugging
        print("Airline classes:", le_airline.classes_)
        print("Source classes:", le_source_city.classes_)
        print("Departure classes:", le_departure_time.classes_)
        print("Stops classes:", le_stops.classes_)
        print("Arrival classes:", le_arrival_time.classes_)
        print("Destination classes:", le_destination_city.classes_)
        print("Class classes:", le_class.classes_)

        print("Model and encoders/scaler loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model/encoders/scaler: {str(e)}")
        return False


def safe_transform(le, items):
    known_classes = set(le.classes_)
    default_class = le.classes_[0]  # Use first known class as fallback
    result = []
    for item in items:
        if item in known_classes:
            result.append(le.transform([item])[0])
        else:
            print(f"Warning: Unknown category '{item}' replaced with default '{default_class}'")
            result.append(le.transform([default_class])[0])
    return result


def preprocess_input(data):
    try:
        df = pd.DataFrame([data])
        # ... existing rename and encoding ...
        # After encoding categorical variables and scaling numeric ones:
        
        # Rename encoded columns to match model feature names
        df.rename(columns={
            'Airline': 'airline_encoded',
            'Source': 'source_encoded',
            'Dep_Time': 'departure_time_encoded',
            'Total_Stops': 'stops_encoded',
            'Arrival_Time': 'arrival_time_encoded',
            'Destination': 'destination_encoded',
            'Class': 'class_encoded'
        }, inplace=True)

        # Add missing engineered features with default/computed values
        df['competition_factor'] = 0  # Assign default or calculate if possible
        
        # Define final ordered features list exactly as model expects
        features = [
            'airline_encoded', 'arrival_time_encoded', 'class_encoded', 'competition_factor',
            'departure_time_encoded', 'destination_encoded', 'duration', 'days_left', 
            'source_encoded', 'stops_encoded'
        ]
        df = df[features]
        
        return df.values
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
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
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please check server logs.",
                "success": False
            }), 500

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No data provided. Please send JSON data.",
                "success": False
            }), 400

        required_fields = ['airline', 'source_city', 'departure', 'stops', 'arrival_time',
                           'destination_city', 'class', 'duration', 'days_left']

        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}",
                "success": False
            }), 400

        processed_data = preprocess_input(data)

        if processed_data is None:
            return jsonify({
                "error": "Error preprocessing input data",
                "success": False
            }), 400

        prediction = model.predict(processed_data)

        return jsonify({
            "success": True,
            "predicted_price": float(prediction[0]),
            "currency": "INR",
            "input_data": data,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "error": f"Prediction error: {str(e)}",
            "success": False
        }), 500


if __name__ == '__main__':
    print("Starting Flight Price Prediction API...")
    if load_model_and_encoders():
        print("Model and encoders loaded successfully!")
    else:
        print("Warning: Failed to load model or encoders. API predictions will fail.")
    app.run(debug=True, host='0.0.0.0', port=5000)
