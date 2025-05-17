import numpy as np
from utils import crop_face, draw_face_box, known_faces_collection, cosine_similarity

def match_image(faces, img):
    matchesFound = []
    cropped_faces = []

    for face in faces:
        embedding = face.embedding

        # Find best match from MongoDB directly
        best_match_name = None
        best_match_score = 0

        for doc in known_faces_collection.find({}):
            known_embedding = np.array(doc["embedding"])
            similarity = cosine_similarity(embedding,known_embedding)

            if similarity > best_match_score:
                best_match_score = similarity
                best_match_name = doc["name"]

        confidence_percentage = round(best_match_score * 100, 2)

        if confidence_percentage > 49:
            matchesFound.append((best_match_name, confidence_percentage))

            bbox = face.bbox.astype(int)
            cropped_face_filename = crop_face(img, bbox[1], bbox[2], bbox[3], bbox[0], best_match_name)
            cropped_faces.append(cropped_face_filename)

            draw_face_box(img, bbox[1], bbox[2], bbox[3], bbox[0], f"{best_match_name} ({confidence_percentage}%)")
    return matchesFound, cropped_faces