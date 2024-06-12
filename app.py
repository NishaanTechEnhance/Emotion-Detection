import streamlit as st
import cv2
import numpy as np
import time
from deepface import DeepFace
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_video_frames_from_file(video_file_path):
    capture = cv2.VideoCapture(video_file_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps
    while capture.isOpened():
        start_time = time.time()
        ret, frame = capture.read()
        if not ret:
            break
        yield frame, frame_time - (time.time() - start_time)
    capture.release()

st.title('Real-Time Emotion Detection')

video_file = st.file_uploader('Upload Video', type=['mp4', 'mov', 'avi'])

if video_file:
    start_detection = st.button('Start Emotion Detection')
    
    if start_detection:
        stframe = st.empty()
        stop_detection_placeholder = st.empty()

        with open('temp_video.mp4', 'wb') as f:
            f.write(video_file.read())

        stop_detection_button = stop_detection_placeholder.button('Stop Emotion Detection')

        for frame, delay in get_video_frames_from_file('temp_video.mp4'):
            if stop_detection_button:
                break

            try:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face_roi = frame[y:y + h, x:x + w]

                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                    emotion = result[0]['dominant_emotion']

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                stframe.image(frame, channels='BGR', use_column_width=True)

                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                st.error(f"Error analyzing frame: {str(e)}")
                continue
