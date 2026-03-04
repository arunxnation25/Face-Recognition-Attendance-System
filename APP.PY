import os
import io
import base64
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_mail import Mail, Message
from PIL import Image
import numpy as np
import face_recognition
from scipy.spatial import distance as dist
import csv
import threading

# ================= APP CONFIG =================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_DIR = os.path.join(APP_DIR, "known_faces")
ATT_FILE = os.path.join(APP_DIR, "attendance.csv")

os.makedirs(KNOWN_DIR, exist_ok=True)

app = Flask(__name__, static_folder='.', static_url_path='')

# ================= EMAIL CONFIG =================

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'arunkoundal2005@gmail.com'
app.config['MAIL_PASSWORD'] = 'vbgaezatnhgiuywy'
app.config['MAIL_DEFAULT_SENDER'] = ('Face Recognition', 'arunkoundal2005@gmail.com')

mail = Mail(app)

# ================= IN-MEMORY =================

known_encodings = []
known_names = []
enc_lock = threading.Lock()
emailed_today = set()

# ================= HELPER FUNCTIONS =================

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def load_known_faces():
    encs = []
    names = []

    for fname in os.listdir(KNOWN_DIR):
        path = os.path.join(KNOWN_DIR, fname)
        if not os.path.isfile(path):
            continue

        name, _ = os.path.splitext(fname)

        try:
            img = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(img)
            if len(enc) > 0:
                encs.append(enc[0])
                names.append(name)
                print(f"Loaded: {name}")
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    with enc_lock:
        global known_encodings, known_names
        known_encodings = encs
        known_names = names

def log_attendance(name):
    exists = os.path.exists(ATT_FILE)
    with open(ATT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Name", "Timestamp"])
        writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

def send_email(subject, recipient, body):
    try:
        msg = Message(subject=subject, recipients=[recipient], body=body)
        mail.send(msg)
        print("Email sent!")
    except Exception as e:
        print("Email error:", e)

# ================= ROUTES =================

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/recognize', methods=['POST'])
def recognize():

    data = request.json
    img_b64 = data.get('image')

    if not img_b64:
        return jsonify({"success": False, "error": "No image provided"}), 400

    if img_b64.startswith('data:image'):
        img_b64 = img_b64.split(',', 1)[1]

    img_bytes = base64.b64decode(img_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(image)

    # Face detection
    face_locations = face_recognition.face_locations(img_np)
    face_encodings = face_recognition.face_encodings(img_np, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(img_np)

    if len(face_encodings) == 0:
        return jsonify({
            "success": False,
            "message": "No face detected",
            "blink": False
        })

    # ================= BLINK DETECTION =================

    blink_detected = False

    for landmarks in face_landmarks_list:
        leftEye = landmarks["left_eye"]
        rightEye = landmarks["right_eye"]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < 0.26:   # more flexible
            blink_detected = True

    # ================= FACE MATCHING =================

    with enc_lock:
        encs = known_encodings.copy()
        names = known_names.copy()

    results = []

    for enc in face_encodings:
     best_name = "Unknown"

    if len(encs) > 0:
        matches = face_recognition.compare_faces(encs, enc, tolerance=0.5)
        distances = face_recognition.face_distance(encs, enc)
        best_idx = np.argmin(distances)

        if matches[best_idx]:
            best_name = names[best_idx]

    results.append(best_name)

    # ✅ Attendance + Email (Every Time)
    if best_name != "Unknown" and blink_detected:
        log_attendance(best_name)

        send_email(
            subject=f"{best_name} detected!",
            recipient="officialarun92@gmail.com",
            body=f"{best_name} marked attendance at {datetime.now()}"
        )

    res = results[0] if len(results) == 1 else results

    return jsonify({
        "success": True,
        "result": res,
        "blink": blink_detected
    })

@app.route('/attendance')
def attendance():
    rows = []
    if os.path.exists(ATT_FILE):
        with open(ATT_FILE, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return jsonify(rows)

@app.route('/add_face', methods=['POST'])
def add_face():
    if 'name' not in request.form or 'file' not in request.files:
        return jsonify({"success": False, "error": "name and file required"}), 400

    name = request.form['name'].strip()
    f = request.files['file']

    if name == '' or f.filename == '':
        return jsonify({"success": False, "error": "Invalid name or file"}), 400

    save_name = f"{name}{os.path.splitext(f.filename)[1]}"
    save_path = os.path.join(KNOWN_DIR, save_name)
    f.save(save_path)

    load_known_faces()

    return jsonify({"success": True, "message": "Face added successfully"})

@app.route('/test_email')
def test_email():
    send_email("Test Email", "officialarun92@gmail.com", "This is a test email.")
    return "Test email sent!"

# ================= START =================

if __name__ == '__main__':
    print("Loading known faces...")
    load_known_faces()
    app.run(host='0.0.0.0', port=5000, debug=True)