import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import insightface
from utils import load_known_faces, crop_face, draw_face_box

app = Flask(__name__)

# Set up paths
KNOWN_FACES_DIR = 'known_faces'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize InsightFace model
face_app = insightface.app.FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load known faces
known_face_embeddings, known_face_names = load_known_faces()

# Function to remove uploaded file after processing
def remove_uploaded_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Route: home page for upload
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('result.html', message="No file uploaded.")
        file = request.files['image']
        if file.filename == '':
            return render_template('result.html', message="No file selected.")
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        return handle_face_recognition(filepath, filename)

    return render_template('index.html')

# Core function: handle face recognition
def handle_face_recognition(filepath, filename):
    img = cv2.imread(filepath)
    if img is None:
        remove_uploaded_file(filepath)
        return render_template('result.html', message="Failed to load the uploaded image. Please try another image.")
    
    faces = face_app.get(img)

    matches = []
    cropped_faces = []

    if faces:
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
                matches.append((matched_name, confidence_percentage))

            # Crop the detected face
            if confidence_percentage > 49:
                bbox = face.bbox.astype(int)
                cropped_face_filename = crop_face(img, bbox[1], bbox[2], bbox[3], bbox[0], matched_name)
                cropped_faces.append(cropped_face_filename)

                # Draw rectangle and label
                draw_face_box(img, bbox[1], bbox[2], bbox[3], bbox[0], f"{matched_name} ({confidence_percentage}%)")

        # Save boxed image if faces were processed
        image_with_boxes_filename = f"boxed_{filename}"
        image_with_boxes_path = os.path.join(app.config['UPLOAD_FOLDER'], image_with_boxes_filename)
        cv2.imwrite(image_with_boxes_path, img)

        # Remove original uploaded file after processing
        remove_uploaded_file(filepath)

        # Pass the matches and cropped faces correctly
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            return render_template('result.html', matches=matches, image_path=image_with_boxes_filename, cropped_faces=cropped_faces)
        else:
            return render_template('result.html', message="No match found.")
    else:
        remove_uploaded_file(filepath)
        return render_template('result.html', message="No face detected.")

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route for adding a new known face
@app.route('/add_known_face', methods=['GET', 'POST'])
def add_known_face():
    if request.method == 'POST':
        name = request.form['name']
        files = request.files.getlist('known_faces')
        
        for file in files:
            image_path = os.path.join(KNOWN_FACES_DIR, secure_filename(file.filename))
            os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            file.save(image_path)

            # Process new face
            img = cv2.imread(image_path)
            if img is None:
                continue

            faces = face_app.get(img)
            if faces:
                known_face_embeddings.append(faces[0].embedding)
                known_face_names.append(name)

        return redirect(url_for('index'))

    return render_template('add_known_face.html')

if __name__ == '__main__':
    app.run(debug=True)
