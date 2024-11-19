from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Route to display the webcam feed
@app.route('/')
def index():
    return render_template('index.html')

# Route to process webcam frame
@app.route('/process-frame', methods=['POST'])
def process_frame():
    # Get the base64-encoded image from the request
    data = request.get_json()
    img_data = data['image']
    
    # Decode the base64 image
    img_data = img_data.split(',')[1]  # Remove the data URL part
    img_bytes = base64.b64decode(img_data)

    # Convert the bytes into a NumPy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform some processing on the image (e.g., convert to grayscale)
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert the processed image back to base64 to send to the frontend
    _, buffer = cv2.imencode('.jpg', processed_img)
    processed_img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'processed_image': f"data:image/jpeg;base64,{processed_img_base64}"})

if __name__ == '__main__':
    app.run(debug=True)
