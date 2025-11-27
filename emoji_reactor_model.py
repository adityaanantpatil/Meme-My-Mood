# emoji_reactor_fer.py
# Real-time Emoji Reactor using FER (much faster and smoother than DeepFace)

import cv2
import mediapipe as mp
import numpy as np
import os
from fer import FER   # pip install fer

# ================================
# CONFIG
# ================================
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)
IMAGES_DIR = "images"

def load_emoji(name):
    path = os.path.join(IMAGES_DIR, name)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Emoji not found: {path}")
    return cv2.resize(img, EMOJI_SIZE)

# Load emojis (make sure these files exist in ./images/)
smiling_emoji   = load_emoji("smile.jpg")      # 😊
straight_emoji  = load_emoji("plain.jpg")      # 😐
handsup_emoji   = load_emoji("air.jpg")        # 🙌
surprised_emoji = load_emoji("suprised.jpg")   # 😲
blank_emoji = np.zeros((EMOJI_SIZE[1], EMOJI_SIZE[0], 3), dtype=np.uint8)

# ================================
# MediaPipe Pose (for hands-up detection)
# ================================
mp_pose = mp.solutions.pose

# Try multiple camera indices (some systems use 0, others 1 or 2)
cap = None
for idx in [1, 0, 2]:
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        print(f"Using camera index {idx}")
        break
if cap is None or not cap.isOpened():
    print("ERROR: Could not open any webcam.")
    exit()

cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Feed", WINDOW_WIDTH, WINDOW_HEIGHT)

cv2.namedWindow("Emoji Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Emoji Output", WINDOW_WIDTH, WINDOW_HEIGHT)

print("\n=== Emoji Reactor Ready ===")
print("Controls: q = quit")
print("😊 Smile → smile.jpg")
print("😐 Neutral → plain.jpg")
print("🙌 Hands up → air.jpg (overrides emotion)")
print("😲 Surprise → suprised.jpg\n")

# Initialize FER detector
# mtcnn=True  → more accurate face detection (slightly slower)
# mtcnn=False → faster, uses OpenCV Haar cascade (recommended for real-time)
emo_detector = FER(mtcnn=False)

# MediaPipe Pose context
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)                   # Mirror
        display_frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ================================
        # 1. Hands-up detection (overrides everything)
        # ================================
        hands_up = False
        pose_result = pose.process(rgb_frame)

        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks.landmark
            left_shoulder_y  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_wrist_y     = lm[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_wrist_y    = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y

            # If any wrist is clearly above its shoulder → hands up
            if left_wrist_y < left_shoulder_y - 0.10 or right_wrist_y < right_shoulder_y - 0.10:
                hands_up = True

        # ================================
        # 2. Emotion detection using FER
        # ================================
        emotion = "neutral"   # fallback

        try:
            # detect_emotions works on BGR frame directly
            result = emo_detector.detect_emotions(frame)
            if result:  # at least one face found
                emotions_dict = result[0]["emotions"]
                dominant = max(emotions_dict, key=emotions_dict.get)

                # Map FER labels to your desired ones
                if dominant in ["happy"]:
                    emotion = "happy"
                elif dominant in ["surprise"]:
                    emotion = "surprise"
                elif dominant in ["neutral"]:
                    emotion = "neutral"
                else:
                    emotion = "neutral"   # everything else → neutral face
        except Exception as e:
            print(f"FER error (rare): {e}")
            emotion = "neutral"

        # ================================
        # 3. Select final emoji
        # ================================
        if hands_up:
            emoji = handsup_emoji
            label = "HANDS UP 🙌"
        else:
            if emotion == "happy":
                emoji = smiling_emoji
                label = "SMILING 😊"
            elif emotion == "surprise":
                emoji = surprised_emoji
                label = "SURPRISED 😲"
            else:
                emoji = straight_emoji
                label = "NEUTRAL 😐"

        # ================================
        # 4. Display
        # ================================
        cv2.putText(display_frame, f"STATE: {label}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Optional: show detected emotion scores (debug)
        # cv2.putText(display_frame, f"Emotion: {emotion}", (10, 70),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Camera Feed", display_frame)
        cv2.imshow("Emoji Output", emoji)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Emoji Reactor stopped.")