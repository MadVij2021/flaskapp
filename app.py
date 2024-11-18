import cv2
from flask import Flask, Response

app = Flask(__name__)

# Initialize the camera (0 is the default webcam, you may need to change it depending on your system)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Read frame-by-frame from the webcam
        success, frame = camera.read()
        frame=cv2.flip(frame,1)
        if not success:
            break
        else:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Yield the frame as a byte stream for the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            
    

@app.route('/')
def homepage():
    return "harshit chopra kaisa hai bhai"

@app.route('/video')
def video_feed():
    # Return the streaming response to the browser
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask app on localhost and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
