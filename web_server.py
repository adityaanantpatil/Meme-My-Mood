from flask import Flask, Response, render_template, request
import cv2
import numpy as np
import threading
import time
import emoji_reactor
from pyngrok import ngrok

app = Flask(__name__)

# ---------- Global camera state for laptop webcam ----------
latest_raw = None
latest_emoji = None
capture_lock = threading.Lock()
capture_thread_started = False


def camera_capture_loop():
    """
    Background thread:
    - Grabs frames from laptop webcam
    - Runs emoji_reactor.process_frame
    - Stores both raw + emoji in globals
    """
    global latest_raw, latest_emoji
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open laptop webcam.")
        return

    print("[web_server] Laptop webcam capture thread started.")
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue

        emoji_frame = emoji_reactor.process_frame(frame)

        with capture_lock:
            latest_raw = frame.copy()
            latest_emoji = emoji_frame.copy()

        time.sleep(0.01)

    cap.release()


def start_capture_thread():
    global capture_thread_started
    if not capture_thread_started:
        t = threading.Thread(target=camera_capture_loop, daemon=True)
        t.start()
        capture_thread_started = True


# ---------- MJPEG generators for laptop streams ----------
def gen_laptop_raw_stream():
    while True:
        with capture_lock:
            frame = None if latest_raw is None else latest_raw.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


def gen_laptop_emoji_stream():
    while True:
        with capture_lock:
            frame = None if latest_emoji is None else latest_emoji.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/laptop_input_stream")
def laptop_input_stream():
    return Response(
        gen_laptop_raw_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/laptop_output_stream")
def laptop_output_stream():
    return Response(
        gen_laptop_emoji_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------- Phone camera frame processing ----------
@app.route("/process_mobile_frame", methods=["POST"])
def process_mobile_frame():
    """
    Receives one frame from the phone (JPEG),
    runs emoji_reactor.process_frame,
    returns processed emoji frame as JPEG.
    """
    if "frame" not in request.files:
        return "No frame uploaded", 400

    file = request.files["frame"]
    file_bytes = file.read()

    np_arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return "Invalid image", 400

    emoji_frame = emoji_reactor.process_frame(frame)

    ret, buffer = cv2.imencode(".jpg", emoji_frame)
    if not ret:
        return "Encoding error", 500

    return Response(buffer.tobytes(), mimetype="image/jpeg")


# ---------- UI ----------
@app.route("/")
def index():
    return render_template("index.html")


# ---------- ngrok helper ----------
def start_ngrok(port=5000):
    """
    Starts an ngrok tunnel to the given port and prints the public URL.
    Requires:
        ngrok config add-authtoken YOUR_TOKEN_HERE
    to have been run once on this machine.
    """
    # Kill any existing tunnels just in case
    ngrok.kill()

    public_url = ngrok.connect(port, "http")
    print("\n[ngrok] Tunnel created:")
    print("   Public URL:", public_url)
    print("   Forwarding -> http://localhost:%d" % port)
    print("\nOPEN THIS ON YOUR PHONE (Chrome):")
    print("   ", public_url, "\n")
    return public_url


if __name__ == "__main__":
    print("[web_server] Starting laptop webcam thread...")
    start_capture_thread()

    print("[web_server] Starting ngrok tunnel...")
    start_ngrok(5000)

    print("[web_server] Starting Flask on 0.0.0.0:5000")
    # use_reloader=False so we don't start threads/tunnels twice
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
