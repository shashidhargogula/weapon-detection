import os
import cv2
import numpy as np
import time
import threading
import pyttsx3
from flask import Flask, render_template, request, Response, redirect, flash, url_for
from ultralytics import YOLO

# Flask setup
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("C:/Users/shash/OneDrive/Desktop/MPAI09/CODE2/best.pt")  # Adjust to your path

# Text-to-Speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech rate
engine.setProperty('volume', 1)  # Max volume

last_alarm_time = 0

# Async text-to-speech
def speak_async(message):
    threading.Thread(target=lambda: [engine.say(message), engine.runAndWait()]).start()

# Detection function
def detect_weapons(frame):
    global last_alarm_time
    results = model.predict(source=frame, imgsz=640, conf=0.5, device='cpu', verbose=False)
    boxes = results[0].boxes
    annotated_frame = frame.copy()

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Alert
        if cls_id in [0, 1] and conf >= 0.5:  # Customize class IDs as needed
            current_time = time.time()
            if current_time - last_alarm_time > 5:
                speak_async(f"{class_name} detected. Please be cautious!")
                last_alarm_time = current_time
            cv2.putText(annotated_frame, f"{class_name.upper()} DETECTED!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return annotated_frame

# Video from file
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = detect_weapons(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

# Live camera
def generate_camera():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = detect_weapons(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

# Routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password!', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            return render_template("result.html", filename=file.filename)
    return render_template("index.html")

@app.route("/video_feed/<filename>")
def video_feed(filename):
    return Response(generate_frames(os.path.join(app.config["UPLOAD_FOLDER"], filename)),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/camera_feed")
def camera_feed():
    return Response(generate_camera(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/open_camera")
def open_camera():
    return render_template("camera.html")

@app.route("/charts")
def charts():
    return render_template("charts.html")

if __name__ == "__main__":
    
    print(">>> Flask app is starting on http://127.0.0.1:5050")
    app.run(debug=True, port=5050)
