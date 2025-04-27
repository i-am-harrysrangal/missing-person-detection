import os
import cv2
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import face_recognition

app = Flask(__name__)

# Set up paths for uploads and known faces
KNOWN_FACES_DIR = 'known_faces'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load known face encodings
known_face_encodings = []
known_face_names = []
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

# Route for image upload and recognition
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

        # Perform face recognition
        return handle_face_recognition(filepath, filename)

    return render_template('index.html')

# Handle face recognition using face_recognition library
def handle_face_recognition(filepath, filename):
    # Load the uploaded image
    unknown_image = face_recognition.load_image_file(filepath)
    unknown_encodings = face_recognition.face_encodings(unknown_image)
    unknown_face_locations = face_recognition.face_locations(unknown_image)

    matches = []
    cropped_faces = []  # Store the cropped faces
    matched_faces = set()  # To keep track of the faces we've already processed

    # Use OpenCV to add bounding boxes
    image_with_boxes = cv2.imread(filepath)  # OpenCV reads the image in BGR format

    if unknown_encodings:
        # Compare faces and create cropped images
        for i, unknown_encoding in enumerate(unknown_encodings):
            results = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
            if True in results:
                match_index = results.index(True)
                matched_name = known_face_names[match_index]
                confidence = face_recognition.face_distance([known_face_encodings[match_index]], unknown_encoding)[0]
                confidence_percentage = round((1 - confidence) * 100, 2)

                # Check if the face was already processed for this person
                if matched_name not in matched_faces:
                    matches.append((matched_name, confidence_percentage))
                    matched_faces.add(matched_name)

                    # Crop the face and save it
                    top, right, bottom, left = unknown_face_locations[i]
                    face_image = unknown_image[top:bottom, left:right]
                    cropped_face_filename = f"{matched_name}_{filename}_{top}_{left}.jpg"
                    cropped_face_path = os.path.join(app.config['UPLOAD_FOLDER'], cropped_face_filename)

                    # Save the cropped face image
                    cv2.imwrite(cropped_face_path, face_image)
                    cropped_faces.append(cropped_face_filename)

                    # Draw a green rectangle around the face and add the name
                    cv2.rectangle(image_with_boxes, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Add the name next to the face
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = matched_name
                    cv2.putText(image_with_boxes, text, (left, top - 10), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Save the image with bounding boxes and names
        image_with_boxes_filename = f"boxed_{filename}"
        image_with_boxes_path = os.path.join(app.config['UPLOAD_FOLDER'], image_with_boxes_filename)
        cv2.imwrite(image_with_boxes_path, image_with_boxes)

        # Render result template with the processed data
        if matches:
            return render_template('result.html', name=matches, image_path=image_with_boxes_filename, cropped_faces=cropped_faces)
        else:
            return render_template('result.html', message="No match found.")
    else:
        return render_template('result.html', message="No face detected.")

# Route to serve the uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
