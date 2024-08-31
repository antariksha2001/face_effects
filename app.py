from flask import Flask, render_template, Response, request
import cv2
import numpy as np

app = Flask(__name__)

# Initialize the camera
camera = cv2.VideoCapture(0)

# Function to apply face blur
def apply_face_blur(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.GaussianBlur(face, (99, 99), 30)
        frame[y:y+h, x:x+w] = face
    return frame

# Function to apply edge detection
def apply_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_colored

# Function to apply sepia filter
def apply_sepia_filter(frame):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_frame = cv2.transform(frame, sepia_filter)
    sepia_frame = np.clip(sepia_frame, 0, 255)
    return sepia_frame

# Function to flip video horizontally
def flip_video(frame):
    return cv2.flip(frame, 1)

# Function to process the video based on user selection
def process_frame(frame, function):
    if function == "Face Blur":
        return apply_face_blur(frame)
    elif function == "Edge Detection":
        return apply_edge_detection(frame)
    elif function == "Sepia Filter":
        return apply_sepia_filter(frame)
    elif function == "Flip Video":
        return flip_video(frame)
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    function = request.args.get('function', 'None')

    def generate():
        while True:
            success, frame = camera.read()
            if not success:
                break
            frame = cv2.resize(frame, (800, 600))  # Resize to 800x600
            frame = process_frame(frame, function)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
