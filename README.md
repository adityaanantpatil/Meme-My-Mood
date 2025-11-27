🎭 Emoji Reactor
Real-Time Facial Expression & Pose-Based Emoji Generator
<p align="center"> <img src="https://img.shields.io/badge/Python-3.8–3.10-3776AB?logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/OpenCV-Enabled-5C3EE8?logo=opencv&logoColor=white" /> <img src="https://img.shields.io/badge/MediaPipe-FaceMesh%20%7C%20Pose-FF6F00?logo=google" /> <img src="https://img.shields.io/badge/Status-Stable-brightgreen?style=flat" /> </p> <p align="center"> <b>Live camera feed → Detect Your Expression/Hands → Show Matching Emoji</b><br> Fast • Stable • Anti-Flicker • Real-Time </p>
🌟 Features



🎯 Real-Time Emotion Detection
Smile 😀
Surprised 😮
Neutral 😐


🙌 Pose Detection
Hands-up detection using full body Pose landmarks

🧠 Stabilized Output
10-frame smoothing buffer
Normalized landmark distances


🪟 Dual Window UI
Camera Feed with live state
Emoji Output



📁 Fully Offline — No Internet required

Raise both hands → 🙌
Smile → 😀
Say "wow" with wide mouth → 😮
Neutral face → 😐

🧩 Project Structure
emoji-reactor/
│── emoji_reactor.py
│── images/
│     ├── smile.jpg
│     ├── plain.jpg
│     ├── air.jpg
│     ├── surprised.jpg
│── README.md

🔧 Installation
1️⃣ Install dependencies
pip install opencv-python mediapipe numpy


⚠️ MediaPipe requires Python 3.10 or lower — 3.11/3.12+ may cause import errors.


2️⃣ Run the program
python emoji_reactor.py

🖼️ Required Emoji Files
Inside the /images folder, include:
smile.jpg → for smiling
plain.jpg → neutral
air.jpg → hands up
surprised.jpg → surprised face


All are automatically resized to fit the emoji window.

🧠 How It Works (Technical Breakdown)
Pose Module
Wrist Y-coordinate < Shoulder Y-coordinate
→ triggers HANDS UP
Face Mesh Module


Extracts:
Eye corners
Mouth corners
Upper & lower inner lips


Computes:
eye_distance → normalization
mouth_open_distance
Rolling average (10-frame anti-flicker)



Decision Logic

State	Condition
🙌 HANDS_UP	Wrist above shoulder
😀 SMILING	mouth_open > 0.11
😮 SURPRISED	mouth_open > 0.22
😐 STRAIGHT_FACE	everything else
🖥️ Controls
Key	Action

q	Quit the program



🧪 Upcoming: Custom Facial Expression Model (WIP)

A new deep-learning–based Facial Expression Recognition (FER) model is currently under development and will soon replace/augment the MediaPipe mouth-distance logic.

🚀 What This Model Will Do

Detect 7+ emotions with higher accuracy
😀 Happy
😐 Neutral
😮 Surprise
😡 Angry
😢 Sad
😤 Disgust
😠 Contempt (optional)

Provide stable predictions using softmax smoothing.
Reduce false detections caused by lighting, angle, and head pose.
Fully offline — no cloud API needed.



🧱 Architecture (Planned)
Lightweight CNN or MobileNetV3-based classifier
Trained on FER-2013 / RAF-DB / custom dataset
Uses cropped 48×48 or 112×112 grayscale/RGB facial images
Optimized for real-time inference on CPU


🔄 Integration Plan

The pipeline will soon look like:
Camera → Face Detection → FER Model → Expression Label → Emoji Output

This will replace the current:
Camera → Face Mesh → Landmark Distances → Emoji Output


The system will auto-switch:

Engine	Status
MediaPipe landmark-based expressions	Active (Current)
Custom FER deep learning model	Coming Soon
🛠️ Experimental Mode (Optional)

A toggle USE_CUSTOM_MODEL = True will allow developers to test the new model once the .h5 or .pt file is added to:
/model/emotion_model.pt


Activation plan inside emoji_reactor.py:

# TODO: Enable when model is ready
USE_CUSTOM_MODEL = False  

if USE_CUSTOM_MODEL:
    # Predict using the custom FER model
    expression = fer_model.predict(face_crop)
else:
    # Fallback to MediaPipe expression logic
    expression = mediapipe_expression_logic()
