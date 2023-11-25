# app.py
from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import threading
import time
import os

app = Flask(__name__)

# Folder to store uploaded videos
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Video recording variables
video_feed = None
video_thread = None
stop_recording = False

# Route to the main page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for uploading video
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return render_template('index.html', message='Video uploaded successfully', video_path=file_path)

# Route for live feed
@app.route('/live_feed')
def live_feed():
    return render_template('live_feed.html')

# Video streaming generator function
def generate_frames():
    global video_feed, stop_recording

    camera = cv2.VideoCapture(0)

    while not stop_recording:
        success, frame = camera.read()
        if not success:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)

    camera.release()

# Route for starting webcam
@app.route('/start_webcam')
def start_webcam():
    global video_feed, video_thread, stop_recording

    if video_thread is None or not video_thread.is_alive():
        video_feed = generate_frames()
        video_thread = threading.Thread(target=app.run, args=('0.0.0.0', 5001), kwargs={'debug': False, 'use_reloader': False})
        video_thread.start()

    stop_recording = False

    return render_template('live_feed.html', video_feed=video_feed)

# Route for stopping webcam
@app.route('/stop_webcam')
def stop_webcam():
    global stop_recording
    stop_recording = True
    return redirect(url_for('live_feed'))

if __name__ == '__main__':
    app.run(debug=True)
