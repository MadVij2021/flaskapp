import base64
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import io

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    # Serve the HTML page that displays the webcam and processed feed
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(data):
    """
    Handle incoming webcam frame data from the frontend, process it,
    and send back the processed frame.
    """
    # Extract the base64-encoded image from the data
    img_data = data['image']
    
    # Decode the base64 string into bytes
    img_data = img_data.split(',')[1]  # Remove the "data:image/jpeg;base64," part
    img_bytes = base64.b64decode(img_data)

    # Convert the bytes to a NumPy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img=cv2.flip(img,0)

    # Process the image (for example, convert to grayscale)
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the processed image back to JPEG
    _, buffer = cv2.imencode('.jpg', processed_img)
    processed_img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Send the processed image back to the frontend
    emit('processed_frame', {'image': f"data:image/jpeg;base64,{processed_img_base64}"})

if __name__ == '__main__':
    socketio.run(app, debug=True)
