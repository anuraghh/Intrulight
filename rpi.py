import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import time
import pigpio

# ───────────── RASPBERRY PI SETUP ─────────────
PI_IP = '192.168.137.78'  # Your Pi's IP address
PIR_PIN = 17              # GPIO pin for PIR sensor
BUZ_PIN = 27              # GPIO pin for Buzzer

print(f"Connecting to Raspberry Pi at {PI_IP}...")
try:
    pi = pigpio.pi(PI_IP)
    if not pi.connected:
        print("Connection Failed. Check IP address and if 'sudo pigpiod' is running on the Pi.")
        exit()
except Exception as e:
    print(f"Connection Failed: {e}")
    print("Please ensure 'pigpio' is installed (pip install pigpio) and 'sudo pigpiod' is running on the Pi.")
    exit()

print("Connection Successful")
pi.set_mode(PIR_PIN, pigpio.INPUT)
pi.set_mode(BUZ_PIN, pigpio.OUTPUT)
pi.set_pull_up_down(PIR_PIN, pigpio.PUD_DOWN)
pi.write(BUZ_PIN, 0)  # Ensure buzzer is off at start

# ───────────── AI MODEL SETUP ─────────────
print("[INFO] Loading AI models...")
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
INTRUDER_DIR = BASE_DIR / "intruders"
MODEL_DIR.mkdir(exist_ok=True)
INTRUDER_DIR.mkdir(exist_ok=True)

IMG_SIZE = 160
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# Load trained models
try:
    svm_path = MODEL_DIR / "svm_model.pkl"
    encoder_path = MODEL_DIR / "label_encoder.pkl"

    if not svm_path.exists() or not encoder_path.exists():
        raise FileNotFoundError("Model files missing")

    svm = joblib.load(svm_path)
    encoder = joblib.load(encoder_path)
    print("[INFO] Loaded SVM and LabelEncoder.")
except FileNotFoundError:
    print("❌ Error: Model files not found in 'models' directory.")
    pi.stop()
    exit()
except Exception as e:
    print(f"❌ Error loading models: {e}")
    pi.stop()
    exit()

# Initialize MTCNN and Facenet
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Start webcam
def open_camera():
    for idx in (0,1,2):
        try:
            cap_try = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        except Exception:
            cap_try = cv2.VideoCapture(idx)
        if cap_try.isOpened():
            print(f"[INFO] Opened camera index {idx}")
            return cap_try
    return None

cap = open_camera()
if cap is None or not cap.isOpened():
    print("❌ Cannot open camera. Make sure a webcam is attached and accessible.")
    pi.stop()
    exit()

print("✅ AI models loaded. Security system is active.")
print("[INFO] Waiting for motion...")

# Helper: beep buzzer for duration_seconds
def beep(duration_seconds=2):
    try:
        pi.write(BUZ_PIN, 1)
        time.sleep(duration_seconds)
    finally:
        pi.write(BUZ_PIN, 0)

# ───────────── MAIN SECURITY LOOP ─────────────
try:
    while True:
        try:
            motion_detected = pi.read(PIR_PIN)
        except Exception as e:
            print(f"[WARN] Failed to read PIR sensor: {e}")
            motion_detected = 0

        if motion_detected:
            print("[EVENT] Motion detected! Capturing image...")
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to capture frame from webcam.")
                continue

            try:
                boxes, _ = mtcnn.detect(frame)
            except Exception as e:
                print(f"[ERROR] MTCNN detection failed: {e}")
                boxes = None

            if boxes is not None and len(boxes) > 0:
                print(f"[INFO] Found {len(boxes)} face(s). Analyzing...")
                Intruder_alert = False

                for box in boxes:
                    x1,y1,x2,y2 = [int(b) for b in box]
                    x1 = max(0,x1); y1 = max(0,y1)
                    x2 = min(frame.shape[1]-1,x2); y2 = min(frame.shape[0]-1,y2)
                    if x2<=x1 or y2<=y1: continue

                    face = frame[y1:y2, x1:x2]
                    if face.size == 0: continue

                    # Preprocess face
                    face = cv2.resize(face,(IMG_SIZE,IMG_SIZE))
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                    face_tensor = torch.from_numpy(face_rgb).permute(2,0,1).unsqueeze(0).float().to(device)

                    with torch.no_grad():
                        embedding = facenet(face_tensor).cpu().numpy()[0]

                    # Predict
                    try:
                        probs = svm.predict_proba([embedding])[0]
                        confidence = float(np.max(probs))
                    except:
                        pred = svm.predict([embedding])[0]
                        confidence = None
                        name = encoder.inverse_transform([pred])[0]
                        print(f"[INFO] Known face detected: {name}, buzzer 2 sec")
                        beep(2)
                        continue

                    if confidence is None or confidence < 0.6:
                        print(">>> INTRUDER ALERT! Unknown Person Detected <<<")
                        Intruder_alert = True
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        img_path = INTRUDER_DIR / f"intruder_{ts}.jpg"
                        try: cv2.imwrite(str(img_path), frame)
                        except: pass
                        beep(6)
                        break
                    else:
                        pred = svm.predict([embedding])[0]
                        name = encoder.inverse_transform([pred])[0]
                        print(f"[INFO] Authorized person detected: {name} (confidence={confidence:.2f}), buzzer 2 sec")
                        beep(2)

            else:
                print("[INFO] Motion detected but no faces found.")

        # Sleep briefly
        time.sleep(0.3)

except KeyboardInterrupt:
    print("\n[INFO] Keyboard interrupt received. Shutting down...")

finally:
    print("[INFO] Cleaning up...")
    try: pi.write(BUZ_PIN, 0)
    except: pass
    try: pi.stop()
    except: pass
    try: cap.release()
    except: pass
    print("[INFO] System stopped. Exiting.")
