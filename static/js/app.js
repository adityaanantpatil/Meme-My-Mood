// static/js/app.js

// ---------- ELEMENTS ----------
const laptopMode = document.getElementById("laptop-mode");
const phoneMode = document.getElementById("phone-mode");
const btnLaptopMode = document.getElementById("mode-laptop");
const btnPhoneMode = document.getElementById("mode-phone");

const videoEl = document.getElementById("mobile-video");
const processedImg = document.getElementById("mobile-processed");
const canvasEl = document.getElementById("mobile-canvas");
const startBtn = document.getElementById("start-mobile");
const stopBtn = document.getElementById("stop-mobile");
const switchCamBtn = document.getElementById("switch-camera");

// ---------- STATE ----------
let mobileStream = null;
let sendInterval = null;
// "user" = front cam, "environment" = back cam
let currentFacingMode = "user";

// ---------- MODE SWITCHING ----------
function setMode(mode) {
    if (mode === "laptop") {
        laptopMode.classList.remove("d-none");
        phoneMode.classList.add("d-none");

        btnLaptopMode.classList.add("btn-primary");
        btnLaptopMode.classList.remove("btn-outline-light");

        btnPhoneMode.classList.add("btn-outline-light");
        btnPhoneMode.classList.remove("btn-primary");

        stopMobileCamera();
    } else {
        laptopMode.classList.add("d-none");
        phoneMode.classList.remove("d-none");

        btnPhoneMode.classList.add("btn-primary");
        btnPhoneMode.classList.remove("btn-outline-light");

        btnLaptopMode.classList.add("btn-outline-light");
        btnLaptopMode.classList.remove("btn-primary");
    }
}

btnLaptopMode.addEventListener("click", () => setMode("laptop"));
btnPhoneMode.addEventListener("click", () => setMode("phone"));

// Default = laptop
setMode("laptop");

// ---------- PHONE CAMERA HANDLING ----------

async function startMobileCamera(forceFacingMode = null) {
    console.log("Starting phone camera...");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert(
            "This browser does not expose navigator.mediaDevices.getUserMedia.\n\n" +
            "On Android Chrome, make sure you opened the HTTPS ngrok URL."
        );
        return;
    }

    if (forceFacingMode) {
        currentFacingMode = forceFacingMode;
    }

    // Helper to actually call getUserMedia with the current facingMode
    async function tryGetStream(facingMode) {
        const constraints = {
            video: { facingMode: { exact: facingMode } },
            audio: false
        };
        console.log("Requesting camera with facingMode:", facingMode, constraints);
        return navigator.mediaDevices.getUserMedia(constraints);
    }

    try {
        let stream;
        try {
            // First attempt with currentFacingMode
            stream = await tryGetStream(currentFacingMode);
        } catch (err) {
            // If current facingMode fails, try the opposite once
            if (
                err.name === "OverconstrainedError" ||
                err.name === "NotFoundError"
            ) {
                console.warn(
                    "FacingMode",
                    currentFacingMode,
                    "failed, trying opposite. Error:",
                    err
                );
                const opposite = currentFacingMode === "user" ? "environment" : "user";
                stream = await tryGetStream(opposite);
                currentFacingMode = opposite;
            } else {
                throw err;
            }
        }

        mobileStream = stream;
        videoEl.srcObject = mobileStream;

        await new Promise((resolve) => {
            videoEl.onloadedmetadata = () => resolve();
        });

        canvasEl.width = videoEl.videoWidth || 640;
        canvasEl.height = videoEl.videoHeight || 480;

        if (sendInterval) clearInterval(sendInterval);
        sendInterval = setInterval(captureAndSendFrame, 200);
    } catch (err) {
        console.error("Error starting phone camera:", err);
        alert(
            "Camera error: " +
                err.name +
                "\n" +
                (err.message || "") +
                "\n\n" +
                "Make sure you:\n" +
                " • Opened the HTTPS ngrok URL\n" +
                " • Allowed camera permissions in Chrome\n"
        );
    }
}

function stopMobileCamera() {
    if (sendInterval) {
        clearInterval(sendInterval);
        sendInterval = null;
    }

    if (mobileStream) {
        mobileStream.getTracks().forEach((t) => t.stop());
        mobileStream = null;
    }

    videoEl.srcObject = null;
}

// Toggle front/back camera
async function switchPhoneCamera() {
    // Flip facing mode
    currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
    console.log("Switching camera, new facingMode:", currentFacingMode);

    stopMobileCamera();
    startMobileCamera(currentFacingMode);
}

function captureAndSendFrame() {
    if (!mobileStream) return;

    const ctx = canvasEl.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

    canvasEl.toBlob(
        async (blob) => {
            if (!blob) return;

            const formData = new FormData();
            formData.append("frame", blob, "frame.jpg");

            try {
                const res = await fetch("/process_mobile_frame", {
                    method: "POST",
                    body: formData
                });

                if (!res.ok) {
                    console.error("Server error:", await res.text());
                    return;
                }

                const outBlob = await res.blob();
                const url = URL.createObjectURL(outBlob);
                processedImg.src = url;
            } catch (err) {
                console.error("Error sending frame:", err);
            }
        },
        "image/jpeg",
        0.8
    );
}

// Button wiring
startBtn.addEventListener("click", () => {
    startMobileCamera();
});

stopBtn.addEventListener("click", () => {
    stopMobileCamera();
});

switchCamBtn.addEventListener("click", () => {
    switchPhoneCamera();
});

// Cleanup
window.addEventListener("beforeunload", () => {
    stopMobileCamera();
});
