import cv2
import sqlite3
from datetime import datetime
from mtcnn import MTCNN
from keras_facenet import FaceNet
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load model pengenalan wajah dan label encoder
model = load_model('face_recognition_model.keras')  # Pastikan path model sesuai
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Setup database SQLite untuk kehadiran
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
''')
conn.commit()

# Fungsi untuk log kehadiran
def log_attendance(name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        SELECT * FROM attendance 
        WHERE name = ? AND DATE(timestamp) = DATE(?)
    ''', (name, timestamp))
    result = cursor.fetchone()

    if result is None:
        cursor.execute('''
            INSERT INTO attendance (name, timestamp) 
            VALUES (?, ?)
        ''', (name, timestamp))
        conn.commit()
        print(f"Kehadiran dicatat untuk {name} pada {timestamp}")
    else:
        print(f"{name} sudah tercatat hadir hari ini.")

# Fungsi untuk pengenalan wajah
embedder = FaceNet()
def recognize_face(face_crop):
    embedding = embedder.embeddings([face_crop])[0]
    embedding = embedding.reshape(1, -1)
    prediction = model.predict(embedding)
    max_prob = np.max(prediction)
    if max_prob < 0.5:
        return "Unknown"
    else:
        return label_encoder.inverse_transform([np.argmax(prediction)])[0]

# Mulai deteksi wajah dari kamera
cap = cv2.VideoCapture(0)
detector = MTCNN()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Tidak dapat mengambil frame dari kamera")
            break

        # Deteksi wajah dalam frame
        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, width, height = face['box']
            face_crop = frame[y:y+height, x:x+width]
            face_crop = cv2.resize(face_crop, (160, 160))

            # Kenali wajah dan log kehadiran
            name = recognize_face(face_crop)
            log_attendance(name)

            # Gambar kotak dan nama pada frame
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
            text_y = y - 20 if y - 20 > 20 else y + 20
            for i in range(8, 0, -2):
                cv2.putText(frame, name, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), i)
            cv2.putText(frame, name, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

        # Tampilkan frame
        cv2.imshow("Classroom Attendance System", frame)
        
        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Pastikan kamera dan koneksi database ditutup setelah selesai
    cap.release()
    cv2.destroyAllWindows()
    conn.close()
