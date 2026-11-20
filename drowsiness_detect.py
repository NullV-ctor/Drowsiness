'''This script detects if a person is drowsy or not, using dlib and eye aspect ratio (EAR)
calculations. Uses webcam video feed as input.

Features:
- Detects face and eye landmarks in real-time.
- Triggers alert sound for 2.5 - 3 seconds when eyes stay closed beyond threshold.
- Automatically captures and saves a screenshot of the drowsy moment with timestamp.
- Avoids audio stuttering by managing playback state cleanly.
'''

import os
import time
import cv2
import dlib
import numpy as np
import pygame
from imutils import face_utils
from scipy.spatial import distance

# Ensure screenshots directory exists
os.makedirs("screenshots", exist_ok=True)

# Initialize Pygame audio mixer and load sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('audio/alert.wav')

# Minimum threshold of eye aspect ratio below which alarm is triggered
EYE_ASPECT_RATIO_THRESHOLD = 0.3

# Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered (~1.5s at 30 FPS)
EYE_ASPECT_RATIO_CONSEC_FRAMES = 45

# Duration for which alarm sounds (2.5 - 3 seconds)
ALARM_DURATION = 3.0

# Cooldown between consecutive automatic screenshots (in seconds)
SCREENSHOT_COOLDOWN = 6.0

# State variables
COUNTER = 0
ALARM_ON = False
alarm_start_time = 0
last_screenshot_time = 0
screenshot_msg_timer = 0

# Load Haar cascade for drawing bounding rectangle around faces
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# Calculates and returns eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load face detector and dlib 68-point shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Extract indexes of facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Start webcam video capture
video_capture = cv2.VideoCapture(0)
time.sleep(1)

print("[INFO] Drowsiness detector started. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[WARNING] Could not read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces via Haar cascade (for blue bounding box)
    face_rectangles = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face_rectangles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Detect facial points via dlib
    faces = detector(gray, 0)
    current_time = time.time()

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)
        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2.0

        # Draw green contours around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Display real-time EAR
        cv2.putText(frame, f"EAR: {eyeAspectRatio:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Check if eyes are closed
        if eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD:
            COUNTER += 1

            # Trigger alert when eyes have been closed long enough
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    alarm_start_time = current_time
                    alert_sound.play()

                    # Save screenshot automatically on drowsiness
                    if current_time - last_screenshot_time > SCREENSHOT_COOLDOWN:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        screenshot_path = f"screenshots/drowsy_{ts}.png"
                        cv2.imwrite(screenshot_path, frame)
                        last_screenshot_time = current_time
                        screenshot_msg_timer = current_time + 2.5
                        print(f"[ALERT] Drowsiness detected! Saved screenshot: {screenshot_path}")
        else:
            COUNTER = 0

    # Stop alarm after ALARM_DURATION (2.5 - 3 seconds)
    if ALARM_ON:
        cv2.putText(frame, "DROWSY ALERT!", (frame.shape[1] // 2 - 140, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        # Visual alert border
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 6)

        if time.time() - alarm_start_time >= ALARM_DURATION:
            alert_sound.stop()
            ALARM_ON = False

    # Display screenshot saved notification
    if time.time() < screenshot_msg_timer:
        cv2.putText(frame, "Screenshot Saved!", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Status indicator
    status_text = "DROWSY" if (ALARM_ON or COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES) else "ALERT"
    status_color = (0, 0, 255) if status_text == "DROWSY" else (0, 255, 0)
    cv2.putText(frame, f"Status: {status_text}", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Show video feed
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
alert_sound.stop()
video_capture.release()
cv2.destroyAllWindows()
print("[INFO] Camera released and windows closed.")
