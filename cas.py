import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Inisialisasi FaceNet
embedder = FaceNet()

# Fungsi untuk mendeteksi dan memotong wajah
def extract_face_embedding(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = cv2.resize(img_rgb, (160, 160))  # Resize gambar wajah untuk FaceNet
    embedding = embedder.embeddings([face])[0]
    return embedding

# Variabel untuk menyimpan embeddings dan label
embeddings = []
names = []

# Akses semua folder siswa
dataset_path = 'dataset'  # Path ke folder dataset
for student_name in os.listdir(dataset_path):
    student_folder = os.path.join(dataset_path, student_name)
    if os.path.isdir(student_folder):
        for image_name in os.listdir(student_folder):
            image_path = os.path.join(student_folder, image_name)
            embedding = extract_face_embedding(image_path)
            embeddings.append(embedding)
            names.append(student_name)

# Konversi ke numpy array
embeddings = np.array(embeddings)

# Encode label nama
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(names)

# Simpan label_encoder ke file
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

model = Sequential()
model.add(Dense(128, input_shape=(embeddings.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(embeddings, labels, epochs=20, batch_size=8, validation_split=0.2)

model.save('face_recognition_model.keras')
