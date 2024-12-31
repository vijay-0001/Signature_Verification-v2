""" import requests

url = "http://127.0.0.1:8000/verify_signature"
files = {
    "signature_to_verify": open("C:/Users/vijay/Desktop/Code/Signature_Verification/sign_data/test/050/01_050.png", "rb"),
    "original_signature": open("C:/Users/vijay/Desktop/Code/Signature_Verification/sign_data/test/050/02_050.png", "rb")
}

response = requests.post(url, files=files)

print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}") """


import requests
import json
import base64
from PIL import Image
from io import BytesIO

url = "http://127.0.0.1:8000/verify_signature"
files = {
    "signature_to_verify": open("C:/Users/vijay/Desktop/Code/Signature_Verification/sign_data/test/057_forg/01_0117057.png", "rb"),
    "original_signature": open("C:/Users/vijay/Desktop/Code/Signature_Verification/sign_data/test/056/01_056.png", "rb")
}

response = requests.post(url, files=files)

# Print the status code in a readable format
print(f"Status Code: {response.status_code}")

# Check if the response is JSON
if response.headers.get("Content-Type") == "application/json":
    try:
        response_json = response.json()
        # Pretty print the JSON response
        print(f"Response JSON: {json.dumps(response_json, indent=4)}")
        
        # Extract prediction and score
        prediction = response_json.get("prediction", "N/A")
        score = response_json.get("score", "N/A")
        
        # Display prediction and score nicely
        print(f"\nPrediction: {prediction}")
        print(f"Score: {score:.3f}" if isinstance(score, (int, float)) else f"Score: {score}")
        
        # If there's a base64 image, decode it and display it
        if "combined_image" in response_json:
            img_data = response_json["combined_image"]
            
            # Decode the image from base64
            img_bytes = base64.b64decode(img_data)
            
            # Convert the bytes into a PIL Image
            img = Image.open(BytesIO(img_bytes))
            
            # Display the image
            img.show()
        
    except json.JSONDecodeError:
        print("Response is not a valid JSON.")
else:
    # If it's not JSON, just print the raw text
    print(f"Response Text: {response.text}")

# Close the opened files
files["signature_to_verify"].close()
files["original_signature"].close()