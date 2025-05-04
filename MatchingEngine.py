import numpy as np
from utils import crop_face, draw_face_box

def match_image(known_face_embeddings, faces, known_face_names, img):

    matchesFound = []
    cropped_faces = []

    for face in faces:

        embedding = face.embedding

        # Calculate cosine similarity
        similarities = np.dot(known_face_embeddings, embedding) / (
            np.linalg.norm(known_face_embeddings, axis=1) * np.linalg.norm(embedding)
        )
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[best_match_index]
        matched_name = known_face_names[best_match_index]
        confidence_percentage = round(best_match_score * 100, 2)

        # add matched only when confidence is more than 50)
        if confidence_percentage > 49:
            matchesFound.append((matched_name, confidence_percentage))

        # Crop the detected face
        if confidence_percentage > 49:
            bbox = face.bbox.astype(int)
            cropped_face_filename = crop_face(img, bbox[1], bbox[2], bbox[3], bbox[0], matched_name)
            cropped_faces.append(cropped_face_filename)

            # Draw rectangle and label
            draw_face_box(img, bbox[1], bbox[2], bbox[3], bbox[0], f"{matched_name} ({confidence_percentage}%)")
    
    return matchesFound, cropped_faces