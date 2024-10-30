import streamlit as st
import cv2
import numpy as np
import os
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import sqlite3
import pandas as pd
from datetime import datetime
from mtcnn import MTCNN
import pickle

# Load model dan encoder
model = load_model('face_recognition_model.keras')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

embedder = FaceNet()
detector = MTCNN()

# Fungsi untuk mendeteksi wajah dan mengembalikan potongan wajah
def detect_faces(frame):
    faces = detector.detect_faces(frame)
    face_crops = []
    for face in faces:
        x, y, width, height = face['box']
        face_crop = frame[y:y+height, x:x+width]
        face_crop = cv2.resize(face_crop, (160, 160))
        face_crops.append(face_crop)
    return face_crops

# Fungsi untuk mendapatkan embedding wajah
def get_face_embeddings(face_crops):
    embeddings = [embedder.embeddings([face])[0] for face in face_crops]
    return np.array(embeddings)  # Pastikan output adalah numpy array 2D

# Fungsi untuk mengenali wajah
def recognize_faces(embeddings):
    if embeddings.ndim == 1:  # Jika hanya satu wajah yang terdeteksi
        embeddings = np.expand_dims(embeddings, axis=0)
    predictions = model.predict(embeddings)
    recognized_faces = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    return recognized_faces

# Fungsi untuk menyimpan kehadiran di database
def log_attendance(name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS attendance (name TEXT, timestamp TEXT)")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, timestamp))
    conn.commit()
    conn.close()

# Fungsi untuk menampilkan riwayat kehadiran
def show_attendance_history():
    conn = sqlite3.connect('attendance.db')
    df = pd.read_sql("SELECT * FROM attendance", conn)
    conn.close()
    return df

# Tampilan Streamlit
st.title("Classroom Attendance System")

# Menu navigasi
menu = ["Deteksi Wajah (Kamera)", "Deteksi Wajah (Gambar)", "Tambah Gambar ke Dataset", "Riwayat Kehadiran"]
choice = st.sidebar.selectbox("Pilih Menu", menu)

# Deteksi wajah menggunakan kamera
if choice == "Deteksi Wajah (Kamera)":
    st.subheader("Deteksi Wajah Menggunakan Kamera")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Kamera tidak terdeteksi.")
            break

        face_crops = detect_faces(frame)
        if face_crops:
            embeddings = get_face_embeddings(face_crops)
            if embeddings.shape[1] == 512:  # Pastikan embeddings sesuai dengan input model
                names = recognize_faces(embeddings)
                
                # Tambahkan kotak dan nama pada wajah yang terdeteksi
                for i, face in enumerate(detector.detect_faces(frame)):
                    x, y, width, height = face['box']
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Buat kotak hijau
                    cv2.putText(frame, names[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    log_attendance(names[i])  # Simpan nama ke database

        # Tampilkan frame di Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()

# Deteksi wajah dari gambar yang diunggah
elif choice == "Deteksi Wajah (Gambar)":
    st.subheader("Deteksi Wajah dari Gambar yang Diunggah")
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        face_crops = detect_faces(frame)
        if face_crops:
            embeddings = get_face_embeddings(face_crops)
            if embeddings.shape[1] == 512:
                names = recognize_faces(embeddings)
                
                # Tambahkan kotak dan nama pada wajah yang terdeteksi
                for i, face in enumerate(detector.detect_faces(frame)):
                    x, y, width, height = face['box']
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Buat kotak hijau
                    cv2.putText(frame, names[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    log_attendance(names[i])  # Simpan nama ke database

                # Tampilkan frame dengan kotak dan nama
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                st.write("Dimensi embeddings tidak sesuai dengan model.")
        else:
            st.write("Tidak ada wajah yang terdeteksi.")

# Menambah gambar ke dataset
elif choice == "Tambah Gambar ke Dataset":
    st.subheader("Tambah Gambar ke Dataset")
    name = st.text_input("Masukkan Nama Orang")
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

    if name and uploaded_file is not None:
        os.makedirs(f"dataset/{name}", exist_ok=True)
        file_path = f"dataset/{name}/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Gambar ditambahkan ke dataset untuk {name}. Silakan latih ulang model untuk menggunakannya.")

# Tampilkan riwayat kehadiran
elif choice == "Riwayat Kehadiran":
    st.subheader("Riwayat Kehadiran")
    attendance_df = show_attendance_history()
    st.dataframe(attendance_df)
