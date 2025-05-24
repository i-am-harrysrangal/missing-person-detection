import os
from urllib import response
import cv2
import datetime
import numpy as np
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import insightface
from config import *
from MatchingEngine import match_image
from utils import *
from flask import Response

app = Flask(__name__)

# Initialize InsightFace model
face_app = insightface.app.FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = face_app.get(img)
    if faces:
        return faces[0].embedding  # Use first face found
    return None

def process_video(video_path, input_embedding):
    timestamps = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return timestamps
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Process every 2 seconds
        if frame_num % int(fps * 2) == 0:
            faces = face_app.get(frame)
            for face in faces:
                similarity = cosine_similarity(input_embedding, face.embedding)
                if similarity > 0.5:  # Threshold for video matches
                    timestamp = str(datetime.timedelta(seconds=int(frame_num / fps)))
                    timestamps.append(timestamp)
                    break  # Avoid duplicate timestamps per frame
        if timestamps: #just check first match, if found break while loop
            break
        frame_num += 1
    cap.release()
    return timestamps

@app.route('/')
def index():
    return render_template('index.html')

def handle_face_recognition(filepath, filename):
    isImage = True
    img = cv2.imread(filepath)

    if img is None:
        remove_uploaded_file(filepath)
        return render_template('result.html', message="Failed to load the uploaded image. Please try another image.")
    
    faces = face_app.get(img)
    if faces:
        matches, cropped_faces = match_image(faces,img)
        # Save boxed image if faces were processed
        image_with_boxes_filename = f"boxed_{filename}"
        image_with_boxes_path = os.path.join(UPLOAD_FOLDER, image_with_boxes_filename)
        cv2.imwrite(image_with_boxes_path, img)

        # Remove original uploaded file after processing
        remove_uploaded_file(filepath)

        # Pass the matches and cropped faces correctly
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            return render_template('result.html', matches=matches, image_path=image_with_boxes_filename, cropped_faces=cropped_faces)
        else:
            return render_template('result.html', message="No match found.",isImage = isImage)
    else:
        remove_uploaded_file(filepath)
        return render_template('result.html', message="No face detected.")

@app.route('/videorecognition', methods=['GET', 'POST'])
def videorecognition():
    if 'image' not in request.files:
        return render_template('result.html', message="No file uploaded.")
    file = request.files['image']
    if file.filename == '':
        return render_template('result.html', message="No file selected.")
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Get input face embedding
    input_embedding = get_face_embedding(filepath)
    if input_embedding is None:
        remove_uploaded_file(filepath)
        return render_template('result.html', message="No face detected in image.")

    # Find first available video
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi'))]
    if not video_files:
        remove_uploaded_file(filepath)
        return render_template('result.html', message="No videos found.")
    
    timestamps = []
    videos_scanned = 0
    for video_file in video_files:    
        
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        video_timestamps = process_video(video_path, input_embedding)
        videos_scanned += 1
        for ts in video_timestamps:
            timestamps.append(f"{video_file} - {ts}")
        if video_timestamps:
            break

    remove_uploaded_file(filepath)

    if not timestamps:
        return render_template('result.html', message="No matches in video.",video_count = videos_scanned)
    return render_template('result.html', timestamps=timestamps,video_count = videos_scanned)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/add_person', methods=['POST'])
def add_person():
    # Collect form data
    full_name = request.form['full_name']
    person_id = request.form['person_id']
    gender = request.form['gender']
    upload_date = request.form['upload_date']
    label_status = request.form.get('label_status', '')
    tags = request.form.get('tags', '')
    last_seen_location = request.form.get('last_seen_location', '')
    last_seen_datetime = request.form.get('last_seen_datetime', '')
    emergency_contact = request.form['emergency_contact']
    remarks = request.form.get('remarks', '')

    # Photo file
    photo = request.files['photo']

    # Call method to process and store
    add_missing_person(
        full_name=full_name,
        person_id=person_id,
        gender=gender,
        photo=photo,
        upload_date=upload_date,
        label_status=label_status,
        tags=tags,
        last_seen_location=last_seen_location,
        last_seen_datetime=last_seen_datetime,
        emergency_contact=emergency_contact,
        remarks=remarks
    )

    return redirect(url_for('index'))


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    persons = getImage()
    missin_count = totalMissingPerson()
    found_count = totalFoundPerson()
    return render_template('dashboard.html', persons=persons,missin_count=missin_count,found_count=found_count)

@app.route('/get-image/<image_id>')
def get_image(image_id):
    print("jdsbfjhdbfhdbfhjdbhfb")
    try:
        image = imagedbid(image_id)
        return Response(image.read(), mimetype='image/jpeg')
    except Exception as e:
        return f"Image not found: {str(e)}", 404

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/openvideorecognition')
def openvideorecognition():
    return render_template('videorecognition.html')

@app.route('/AddPerson')
def AddPerson():
    return render_template('AddPerson.html')

@app.route('/imagerecognition', methods=['GET', 'POST'])
def imagerecognition():
    if request.method == 'POST':
            if 'image' not in request.files:
                return render_template('result.html', message="No file uploaded.")
            file = request.files['image']
            if file.filename == '':
                return render_template('result.html', message="No file selected.")
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            return handle_face_recognition(filepath, filename)
    return render_template('imagerecognition.html')

if __name__ == '__main__':
    app.run(debug=True)