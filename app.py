from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import time
from model import predict_frame

app = Flask(__name__)

# Home
@app.route('/')
def home():
    return render_template('index.html')

# Upload
@app.route('/upload')
def upload():
    return render_template('upload.html')

# About
@app.route('/about')
def about():
    return render_template('about.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['video']
    filename = file.filename

    filepath = os.path.join('static', filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)

    frame_count = 0
    fake_score = 0
    temporal_score = 0

    prev_frame = None
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames
        if frame_count % 3 != 0:
            continue

        # Spatial
        prediction = predict_frame(frame)
        fake_score += prediction

        # Temporal
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            motion = np.mean(diff)

            if motion < 2:
                temporal_score += 0.3

        prev_frame = frame

        if frame_count > 100:
            break

    cap.release()

    processing_time = round(time.time() - start_time, 2)

    # ✅ Correct normalization
    effective_frames = frame_count // 3
    if effective_frames == 0:
        effective_frames = 1

    total_score = fake_score + temporal_score
    ratio = total_score / effective_frames

    # ✅ FINAL DECISION
    if ratio >= 0.5:
        result = "Fake"
        confidence = round(min(ratio, 1.0) * 100, 2)
    else:
        result = "Real"
        confidence = round((1 - min(ratio, 1.0)) * 100, 2)

    return render_template('result.html',
                           result=result,
                           confidence=confidence,
                           filename=filename,
                           frames=effective_frames,
                           time=processing_time)


if __name__ == "__main__":
    app.run(debug=True)