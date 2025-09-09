""" from flask import Flask, request, jsonify
from models import PricePredictor, AnomalyDetector

app = Flask(__name__)

# Initialize models once at startup
predictor = PricePredictor('xgboost.pkl')
anomaly_detector = AnomalyDetector('isolation_forest.pkl', preprocessor=predictor)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.json

        # Validate required features keys
        required_keys = [
            'airline', 'source_city', 'destination_city', 'class',
            'stops', 'departure_time', 'arrival_time', 'duration', 'days_left'
        ]
        missing = [k for k in required_keys if k not in features]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Get prediction and anomaly detection results
        price = predictor.predict(features)
        is_anomaly = anomaly_detector.detect(features)

        return jsonify({
            'predicted_price': float(price),
            'is_anomaly': bool(is_anomaly)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
 """



from flask import Flask, request, jsonify, render_template_string
import logging
from models import FlightPricePipeline
from config import *
print("API_HOST =", API_HOST)
print("API_PORT =", API_PORT)
print("API_DEBUG =", API_DEBUG)

import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

pipeline = None

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Flight Price Prediction API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        select, input { width: 100%; padding: 8px; margin-bottom: 10px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; }
        .success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
    </style>
</head>
<body>
    <div class="container">
        <h1>✈️ Flight Price Prediction API</h1>
        
        <form id="predictionForm">
            <div class="form-group">
                <label for="airline">Airline:</label>
                <select id="airline" name="airline" required>
                    <option value="">Select Airline</option>
                    <option value="AirAsia">AirAsia</option>
                    <option value="Air_India">Air India</option>
                    <option value="Vistara">Vistara</option>
                    <option value="Indigo">Indigo</option>
                    <option value="SpiceJet">SpiceJet</option>
                    <option value="GO_FIRST">GO FIRST</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="source_city">Source City:</label>
                <select id="source_city" name="source_city" required>
                    <option value="">Select Source</option>
                    <option value="Delhi">Delhi</option>
                    <option value="Mumbai">Mumbai</option>
                    <option value="Bangalore">Bangalore</option>
                    <option value="Chennai">Chennai</option>
                    <option value="Kolkata">Kolkata</option>
                    <option value="Hyderabad">Hyderabad</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="destination_city">Destination City:</label>
                <select id="destination_city" name="destination_city" required>
                    <option value="">Select Destination</option>
                    <option value="Delhi">Delhi</option>
                    <option value="Mumbai">Mumbai</option>
                    <option value="Bangalore">Bangalore</option>
                    <option value="Chennai">Chennai</option>
                    <option value="Kolkata">Kolkata</option>
                    <option value="Hyderabad">Hyderabad</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="class">Class:</label>
                <select id="class" name="class" required>
                    <option value="">Select Class</option>
                    <option value="Economy">Economy</option>
                    <option value="Business">Business</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="stops">Stops:</label>
                <select id="stops" name="stops" required>
                    <option value="">Select Stops</option>
                    <option value="zero">Non-stop</option>
                    <option value="one">1 Stop</option>
                    <option value="two_or_more">2+ Stops</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="departure_time">Departure Time:</label>
                <select id="departure_time" name="departure_time" required>
                    <option value="">Select Departure Time</option>
                    <option value="Early_Morning">Early Morning</option>
                    <option value="Morning">Morning</option>
                    <option value="Afternoon">Afternoon</option>
                    <option value="Evening">Evening</option>
                    <option value="Night">Night</option>
                    <option value="Late_Night">Late Night</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="arrival_time">Arrival Time:</label>
                <select id="arrival_time" name="arrival_time" required>
                    <option value="">Select Arrival Time</option>
                    <option value="Early_Morning">Early Morning</option>
                    <option value="Morning">Morning</option>
                    <option value="Afternoon">Afternoon</option>
                    <option value="Evening">Evening</option>
                    <option value="Night">Night</option>
                    <option value="Late_Night">Late Night</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="duration">Duration (hours):</label>
                <input type="number" id="duration" name="duration" step="0.25" min="0.5" max="20" required>
            </div>
            
            <div class="form-group">
                <label for="days_left">Days Until Departure:</label>
                <input type="number" id="days_left" name="days_left" min="0" max="365" required>
            </div>
            
            <button type="submit">Predict Price & Check Anomaly</button>
        </form>
        
        <div id="result"></div>
        
        <h2>API Documentation</h2>
        <h3>POST /predict</h3>
        <p>Make predictions using JSON data:</p>
        <pre>{
  "airline": "AirAsia",
  "source_city": "Delhi", 
  "destination_city": "Mumbai",
  "class": "Economy",
  "stops": "zero",
  "departure_time": "Morning",
  "arrival_time": "Afternoon", 
  "duration": 2.25,
  "days_left": 7
}</pre>
        
        <h3>GET /health</h3>
        <p>Check API health status</p>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = Object.fromEntries(formData.entries());
            
            // Convert numeric fields
            data.duration = parseFloat(data.duration);
            data.days_left = parseInt(data.days_left);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                let resultHtml = '';
                let resultClass = 'success';
                
                if (result.error) {
                    resultClass = 'error';
                    resultHtml = `<strong>Error:</strong> ${result.error}`;
                } else {
                    if (result.is_anomaly) {
                        resultClass = 'warning';
                        resultHtml = `
                            <strong>⚠️ Anomalous Price Detected!</strong><br>
                            Predicted Price: ₹${result.predicted_price.toFixed(2)}<br>
                            This price appears unusual for the given route and conditions.
                        `;
                    } else {
                        resultHtml = `
                            <strong>✅ Normal Price Prediction</strong><br>
                            Predicted Price: ₹${result.predicted_price.toFixed(2)}<br>
                            Confidence: ${result.confidence || 'High'}
                        `;
                    }
                }
                
                document.getElementById('result').innerHTML = `<div class="result ${resultClass}">${resultHtml}</div>`;
                
            } catch (error) {
                document.getElementById('result').innerHTML = `<div class="result error"><strong>Error:</strong> ${error.message}</div>`;
            }
        });
    </script>
</body>
</html>
'''

def initialize_pipeline():
    global pipeline
    try:
        price_model_path = Path(__file__).parent.parent / 'models' / 'best_price_prediction_model.pkl'
        anomaly_model_path = Path(__file__).parent.parent / 'models' / 'isolation_forest.pkl'
        
        if not price_model_path.exists():
            model_files = list((Path(__file__).parent.parent / 'models').glob('*.pkl'))
            price_models = [f for f in model_files if 'price' in f.name or 'xgboost' in f.name or 'lightgbm' in f.name]
            price_model_path = price_models[0] if price_models else None
        
        if not anomaly_model_path.exists():
            anomaly_files = [f for f in (Path(__file__).parent.parent / 'models').glob('*.pkl') if any(x in f.name for x in ['isolation', 'anomaly', 'dbscan'])]
            anomaly_model_path = anomaly_files[0] if anomaly_files else None
        
        if price_model_path:
            anomaly_path = str(anomaly_model_path) if anomaly_model_path else None
            pipeline = FlightPricePipeline(str(price_model_path), anomaly_path)
            logger.info("Pipeline initialized successfully")
            return True
        else:
            logger.error("Price model not found; pipeline initialization failed")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return False

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        'status': 'healthy' if pipeline else 'unhealthy',
        'pipeline_loaded': pipeline is not None,
        'version': '1.0.0'
    }
    
    return jsonify(status), 200 if pipeline else 503

@app.route('/predict', methods=['POST'])
def predict():
    if not pipeline:
        return jsonify({'error': 'Prediction pipeline not available'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        required_fields = [
            'airline', 'source_city', 'destination_city', 'class',
            'stops', 'departure_time', 'arrival_time', 'duration', 'days_left'
        ]
        missing = [field for field in required_fields if field not in data]
        if missing:
            return jsonify({'error': f'Missing required fields: {missing}'}), 400
        
        try:
            data['duration'] = float(data['duration'])
            data['days_left'] = int(data['days_left'])
        except (ValueError, TypeError):
            return jsonify({'error': 'Duration must be a number and days_left must be an integer'}), 400
        
        if not (0.5 <= data['duration'] <= 20):
            return jsonify({'error': 'Duration must be between 0.5 and 20 hours'}), 400
        
        if not (0 <= data['days_left'] <= 365):
            return jsonify({'error': 'Days left must be between 0 and 365'}), 400
        
        result = pipeline.predict_with_anomaly_check(data)
        
        if result['error']:
            return jsonify({'error': result['error']}), 500
        
        return jsonify({
            'predicted_price': result['predicted_price'],
            'is_anomaly': result['is_anomaly'],
            'confidence': result['confidence'],
            'input_data': data
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if not pipeline:
        return jsonify({'error': 'Prediction pipeline not available'}), 503
    
    try:
        data = request.get_json()
        if not data or 'flights' not in data:
            return jsonify({'error': 'Expected JSON with flights array'}), 400
        
        flights = data['flights']
        if not isinstance(flights, list):
            return jsonify({'error': 'flights must be an array'}), 400
        
        if len(flights) > 100:
            return jsonify({'error': 'Maximum 100 flights per batch'}), 400
        
        results = []
        for i, flight_data in enumerate(flights):
            try:
                result = pipeline.predict_with_anomaly_check(flight_data)
                results.append({
                    'index': i,
                    'predicted_price': result['predicted_price'],
                    'is_anomaly': result['is_anomaly'],
                    'confidence': result['confidence']
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if initialize_pipeline():
        logger.info(f"Starting Flask API on {API_HOST}:{API_PORT}")
        app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
    else:
        logger.error("Failed to start API - pipeline initialization failed")
