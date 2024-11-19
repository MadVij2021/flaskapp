import base64
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import io
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from PIL import Image


MODEL_PATH = r"C:\Users\lenovo\Downloads\pose_landmarker_lite.task"

app = Flask(__name__)
socketio = SocketIO(app)



# Drawing landmarks
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for pose_landmarks in pose_landmarks_list:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

base_options = python.BaseOptions(delegate=0,model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

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
    rgb_frame=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rgb_frame=cv2.flip(rgb_frame,1)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

# # Pose detection
    detection_result = detector.detect(mp_image)

    # Draw landmarks
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

    # Convert the processed image back to JPEG
    _, buffer = cv2.imencode('.jpg', annotated_image)
    processed_img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Send the processed image back to the frontend
    emit('processed_frame', {'image': f"data:image/jpeg;base64,{processed_img_base64}"})

if __name__ == '__main__':
    socketio.run(app, debug=True)
