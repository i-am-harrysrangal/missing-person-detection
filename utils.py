import os
import cv2
import numpy as np
import insightface
from config import KNOWN_FACES_DIR, UPLOAD_FOLDER

# Initialize InsightFace model
face_app = insightface.app.FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load known faces and their embeddings
def load_known_faces():
    encodings = []
    names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.avif')):
            img_path = os.path.join(KNOWN_FACES_DIR, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = face_app.get(img)
            if faces:
                encodings.append(faces[0].embedding)
                names.append(os.path.splitext(filename)[0])
    return np.array(encodings), names

# Crop and save face image
def crop_face(image, top, right, bottom, left, filename):
    face_image = image[top:bottom, left:right]
    if face_image.size == 0:
        return None  # if cropping failed
    cropped_face_filename = f"{filename}_{top}_{left}.jpg"
    cropped_face_path = os.path.join(UPLOAD_FOLDER, cropped_face_filename)
    cv2.imwrite(cropped_face_path, face_image)
    return cropped_face_filename

# Draw bounding box and label on image
def draw_face_box(image, top, right, bottom, left, label):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, label, (left, top - 10), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
