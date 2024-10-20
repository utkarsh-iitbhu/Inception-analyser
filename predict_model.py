# Make a prediction on some input data
import requests
import json

def send_prediction_request(text):
    # Define the URL of the Flask app
    url = 'http://127.0.0.1:8080/predict' 
    
    # Create the JSON payload
    payload = {
        'text': text
    }
    
    # Set the headers
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        # Send the POST request to the Flask app
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            print("Success! Response from server:")
            print(response.json())
        else:
            print(f"Failed with status code {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {str(e)}")

if __name__ == "__main__":
    # Input text for prediction
    text_to_predict = "How much longer are we going to wait for them to take care of themselves ?"
    
    # Call the function to send a request to the Flask server
    send_prediction_request(text_to_predict)
