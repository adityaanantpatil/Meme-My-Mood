#!/usr/bin/env python3
"""
Real-time emoji display based on camera pose and facial expression detection.
Stable version with normalized features and anti-flickering.
"""

import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Configuration
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Tuned thresholds (normalized by eye distance)
SMILE_THRESHOLD = 0.11       # Reliable slight smile detection
SURPRISE_THRESHOLD = 0.22    # Clear open mouth (O shape)

# Path to images folder
IMAGES_DIR = "images"

def load_emoji(filename, display_name):
    path = os.path.join(IMAGES_DIR, filename)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"{display_name} not found at {path}")
    return cv2.resize(img, EMOJI_WINDOW_SIZE)

# Load emoji images
try:
    smiling_emoji = load_emoji("smile.jpg", "smile.jpg")
    straight_face_emoji = load_emoji("plain.jpg", "plain.jpg")
    hands_up_emoji = load_emoji("air.jpg", "air.jpg")
    surprised_emoji = load_emoji("surprised.jpg", "surprised_emoji.jpg")  # Fixed typo

except Exception as e:
    print("Error loading emoji images!")
    print(f"Details: {e}")
    print("\nPlease ensure the following files exist in the 'images' folder:")
    print("   smile.jpg")
    print("   plain.jpg")
    print("   air.jpg")
    print("   surprised_emoji.jpg")
    exit()

blank_emoji = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Window setup
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Emoji Output', WINDOW_WIDTH + 150, 100)

print("Real-time Emoji Detector Started")
print("   Raise both hands above shoulders   → Hands Up")
print("   Smile naturally                     → Smiling Face")
print("   Open mouth wide (say 'wow')         → Surprised Face")
print("   Neutral expression                  → Straight Face")
print("   Press 'q' to quit\n")

# Initialize history buffer for smoothing (attached to face_mesh to persist)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    # History buffer for mouth openness (anti-flicker)
    face_mesh.mouth_open_history = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        current_state = "STRAIGHT_FACE"

        # 1. Pose detection (hands up has highest priority)
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            left_shoulder_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_wrist_y = lm[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_wrist_y = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y

            if left_wrist_y < left_shoulder_y or right_wrist_y < right_shoulder_y:
                current_state = "HANDS_UP"

        # 2. Face expression analysis (only if hands not up)
        if current_state != "HANDS_UP":
            face_results = face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark

                # Key facial landmarks
                left_eye = np.array([landmarks[33].x, landmarks[33].y])      # Left eye outer corner
                right_eye = np.array([landmarks[263].x, landmarks[263].y])   # Right eye outer corner
                left_mouth = np.array([landmarks[61].x, landmarks[61].y])    # Left mouth corner
                right_mouth = np.array([landmarks[291].x, landmarks[291].y]) # Right mouth corner
                upper_lip = np.array([landmarks[13].x, landmarks[13].y])     # Center top inner lip
                lower_lip = np.array([landmarks[14].x, landmarks[14].y])     # Center bottom inner lip

                # Distances
                eye_distance = np.linalg.norm(right_eye - left_eye)
                mouth_open_distance = np.linalg.norm(lower_lip - upper_lip)

                if eye_distance > 0:
                    normalized_mouth_open = mouth_open_distance / eye_distance
                else:
                    normalized_mouth_open = 0

                # Update smoothing buffer
                face_mesh.mouth_open_history.append(normalized_mouth_open)
                if len(face_mesh.mouth_open_history) > 10:
                    face_mesh.mouth_open_history.pop(0)

                # Use average over last few frames
                avg_mouth_open = sum(face_mesh.mouth_open_history) / len(face_mesh.mouth_open_history)

                # Decision logic with hysteresis
                if avg_mouth_open > SURPRISE_THRESHOLD:
                    current_state = "SURPRISED"
                elif avg_mouth_open > SMILE_THRESHOLD:
                    current_state = "SMILING"
                else:
                    current_state = "STRAIGHT_FACE"

        # Select emoji
        if current_state == "HANDS_UP":
            emoji_display = hands_up_emoji
        elif current_state == "SMILING":
            emoji_display = smiling_emoji
        elif current_state == "SURPRISED":
            emoji_display = surprised_emoji
        else:
            emoji_display = straight_face_emoji

        # Display camera feed with status
        display_frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.putText(display_frame, f'STATE: {current_state}', (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display_frame, 'Press Q to quit', (10, WINDOW_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow('Camera Feed', display_frame)
        cv2.imshow('Emoji Output', emoji_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Emoji detector stopped.")