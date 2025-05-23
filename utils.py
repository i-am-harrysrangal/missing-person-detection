import os
import cv2
import insightface
import numpy as np
from config import *
from werkzeug.utils import secure_filename
from pymongo.mongo_client import MongoClient
from gridfs import GridFS
from bson import ObjectId


# Create a new client and connect to the server
client = MongoClient(DB_URL)
db = client["masterDb"]
known_faces_collection = db["known_faces"]
imagedb = GridFS(db)

# Initialize InsightFace model
face_app = insightface.app.FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

known_face_embeddings = []
known_face_names = []

def known_faces():
    for doc in known_faces_collection.find({}):
        known_face_embeddings.append(doc["embedding"])
        known_face_names.append(doc["name"])
    return known_face_embeddings,known_face_names

def crop_face(image, top, right, bottom, left, filename):
    face_image = image[top:bottom, left:right]
    if face_image.size == 0:
        return None
    cropped_face_filename = f"{filename}_{top}_{left}.jpg"
    cropped_face_path = os.path.join(UPLOAD_FOLDER, cropped_face_filename)
    cv2.imwrite(cropped_face_path, face_image)
    return cropped_face_filename

# Draw bounding box and label on image
def draw_face_box(image, top, right, bottom, left, label):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, label, (left, top - 10), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

def remove_uploaded_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def add_missing_person(full_name, person_id, gender, photo, upload_date,
                       label_status, tags, last_seen_location, last_seen_datetime,
                       emergency_contact, remarks):
    filename = secure_filename(photo.filename)
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    photo.save(filepath)

    photo.seek(0)
    image_id = imagedb.put(photo.stream, filename=filename)

    # Process image
    img = cv2.imread(filepath)
    if img is not None:
        faces = face_app.get(img)
        if faces:
            embedding = faces[0].embedding.tolist()

            # Save to MongoDB
            known_faces_collection.insert_one({
                "name": full_name,
                "person_id": person_id,
                "gender": gender,
                "embedding": embedding,
                "photo_filename": filename,
                "upload_date": upload_date,
                "label_status": label_status,
                "tags": tags,
                "last_seen_location": last_seen_location,
                "last_seen_datetime": last_seen_datetime,
                "emergency_contact": emergency_contact,
                "remarks": remarks,
                "image_id": image_id
            })

def getImage():
    return list(known_faces_collection.find({}))

def imagedbid(image_id):
    return imagedb.get(ObjectId(image_id))

def totalMissingPerson():
    return list(known_faces_collection.find({"label_status": "Missing"}))

def totalFoundPerson():
    return list(known_faces_collection.find({"label_status": "Found"}))