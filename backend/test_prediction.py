import requests
import json
import sys

# Test data sample
sample_data = {
    "airline": "AirAsia",
    "source_city": "Delhi",
    "departure_time": "Evening",
    "stops": "one",
    "arrival_time": "Night",
    "destination_city": "Mumbai",
    "class": "Economy",
    "duration": 2.5,
    "days_left": 14,
    "price": 0  # Add default price for competition factor
}

def test_prediction():
    print("=== Starting Prediction Test ===")
    print("Sending request to /predict endpoint...")
    
    try:
        # Print the request data
        print("\nRequest data:")
        print(json.dumps(sample_data, indent=2))
        
        # Make a POST request to the prediction endpoint
        response = requests.post(
            "http://localhost:5000/predict",
            headers={"Content-Type": "application/json"},
            json=sample_data  # Use json parameter instead of data for automatic serialization
        )
        
        # Print the response
        print("\nResponse Status Code:", response.status_code)
        
        try:
            response_data = response.json()
            print("Response JSON:")
            print(json.dumps(response_data, indent=2))
            
            if response.status_code == 200:
                print("\n✅ Prediction successful!")
                print(f"Predicted price: {response_data.get('predicted_price')}")
            else:
                print("\n❌ Prediction failed!")
                
        except ValueError:
            print("Response content (not JSON):")
            print(response.text)
            
        return response.status_code == 200
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response content: {e.response.text}")
        return False
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction()
    sys.exit(0 if success else 1)
