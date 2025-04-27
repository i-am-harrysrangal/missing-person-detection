# utils.py

import os
import cv2
import face_recognition
from config import KNOWN_FACES_DIR, UPLOAD_FOLDER

# Load known faces and their encodings
def load_known_faces():
    encodings = []
    names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
            face_enc = face_recognition.face_encodings(image)
            if face_enc:
                encodings.append(face_enc[0])
                names.append(os.path.splitext(filename)[0])
    return encodings, names

# Crop and save face image
def crop_face(image, top, right, bottom, left, filename):
    face_image = image[top:bottom, left:right]
    cropped_face_filename = f"{filename}_{top}_{left}.jpg"
    cropped_face_path = os.path.join(UPLOAD_FOLDER, cropped_face_filename)
    cv2.imwrite(cropped_face_path, face_image)
    return cropped_face_filename

# Draw bounding box and label on image
def draw_face_box(image, top, right, bottom, left, label):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, label, (left, top - 10), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
