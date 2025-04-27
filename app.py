import os
import cv2
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import face_recognition
from utils import load_known_faces, crop_face, draw_face_box

app = Flask(__name__)

# Set up paths for uploads and known faces
KNOWN_FACES_DIR = 'known_faces'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load known face encodings and names
known_face_encodings, known_face_names = load_known_faces()

# Function to remove uploaded file after processing
def remove_uploaded_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Route for uploading image and performing recognition
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

# Function to handle face recognition
def handle_face_recognition(filepath, filename):
    # Load the uploaded image
    unknown_image = face_recognition.load_image_file(filepath)
    unknown_encodings = face_recognition.face_encodings(unknown_image)
    unknown_face_locations = face_recognition.face_locations(unknown_image)

    matches = []
    cropped_faces = []
    matched_faces = set()

    image_with_boxes = cv2.imread(filepath)  # OpenCV reads in BGR format

    if unknown_encodings:
        for i, unknown_encoding in enumerate(unknown_encodings):
            # Get distances to all known faces
            face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
            
            # Find the best match
            best_match_index = face_distances.argmin()
            best_match_distance = face_distances[best_match_index]
            confidence_percentage = round((1 - best_match_distance) * 100, 2)
            matched_name = known_face_names[best_match_index]

            # Only process if not already matched
            if matched_name not in matched_faces:
                matches.append((matched_name, confidence_percentage))
                matched_faces.add(matched_name)

                # Crop the face and save
                top, right, bottom, left = unknown_face_locations[i]
                cropped_face_filename = crop_face(unknown_image, top, right, bottom, left, matched_name)
                cropped_faces.append(cropped_face_filename)

                # Draw bounding box and name with confidence
                draw_face_box(image_with_boxes, top, right, bottom, left, f"{matched_name} ({confidence_percentage}%)")

        # Save the boxed image
        image_with_boxes_filename = f"boxed_{filename}"
        image_with_boxes_path = os.path.join(app.config['UPLOAD_FOLDER'], image_with_boxes_filename)
        cv2.imwrite(image_with_boxes_path, image_with_boxes)

        # Remove the uploaded original image
        remove_uploaded_file(filepath)

        if matches:
            # Sort matches by highest confidence
            matches.sort(key=lambda x: x[1], reverse=True)
            best_match = matches[0]  # Take only the best match
            return render_template('result.html', name=[best_match], image_path=image_with_boxes_filename, cropped_faces=cropped_faces)
        else:
            return render_template('result.html', message="No match found.")
    else:
        remove_uploaded_file(filepath)
        return render_template('result.html', message="No face detected.")

# Route to serve uploaded and processed images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to add a new known face
@app.route('/add_known_face', methods=['GET', 'POST'])
def add_known_face():
    if request.method == 'POST':
        name = request.form['name']
        files = request.files.getlist('known_faces')
        
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        for file in files:
            image_path = os.path.join(KNOWN_FACES_DIR, secure_filename(file.filename))
            file.save(image_path)

            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)

        return redirect(url_for('index'))

    return render_template('add_known_face.html')

if __name__ == '__main__':
    app.run(debug=True)
