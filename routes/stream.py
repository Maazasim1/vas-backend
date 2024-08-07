# routes/stream.py
from flask import Blueprint, Response, request, current_app, json
from tqdm import tqdm
import requests
import os
import time
import base64
from PIL import Image
import numpy as np
import io
from firebase_admin import storage
import cv2
from config import Config
from ultralytics import YOLO
from services.utils import face_detection
from services.embeddings import extract_embeddings, compare_embeddings
from mongo_client import insert_metadata
bp = Blueprint('stream', __name__)

@bp.route('/stream/image', methods=['GET'])
def stream_image():
    with current_app.app_context():
        mongo_client = current_app.config['mongo_client']
        image_id = request.args.get('image_id')
        email_id = request.args.get('email')
        if not image_id:
            return "No image ID provided", 400

        # Check if the file exists in Firebase Storage
        bucket = storage.bucket()
        print(image_id)
        blob = bucket.blob(f'upload_file/images/{image_id}.jpg')
        if not blob.exists():
            return "Image not found", 404

        image_url = blob.public_url

        insertion_bool = False
        db = mongo_client.vas
        collection = db['users']
        result = collection.find_one({'email': email_id})
        if not result:
            return "No user with this email found", 404

        # Access the utility functions and classes from the app context
        generate_unique_id = current_app.config['generate_unique_id']
        extract_embeddings = current_app.config['extract_embeddings']
        compare_embeddings = current_app.config['compare_embeddings']
        VideoProcessor = current_app.config['VideoProcessor']

        # Use the image URL instead of the local path
        reference_embedding = extract_embeddings(image_url)
        if reference_embedding is None:
            return "No face detected in the reference image", 400

        def generate():
            video_count = 0
            blobs = bucket.list_blobs(prefix='NVR/')
            nvr_total = len([blob for blob in blobs if blob.name.endswith('.mp4')])
            for blob in blobs:
                if blob.name.endswith('.mp4'):
                    video_count += 1
                    video_url = blob.public_url
                    video_processor = VideoProcessor(video_url, mongo_client)

                    res = yield from video_processor.process_video(
                        reference_embedding=reference_embedding,
                        email_id=email_id,
                        image_id=image_id,
                        image_path=image_url,  # Pass the URL here
                        insertion_bool=insertion_bool,
                        video_count=video_count
                    )
                    if res:
                        yield "data: Duplicate key detected\n\n"
                        break
            final_response = {'completed': "Processing on NVR completed!"}
            yield f'data: {json.dumps(final_response)}\n\n'

        return Response(generate(), mimetype='text/event-stream')

    
@bp.route('/stream/video', methods=["GET"])
def stream_video():
    with current_app.app_context():
        mongo_client = current_app.config['mongo_client']
        video_id = request.args.get('image_id')
        email_id = request.args.get('email')
        if not video_id:
            return "No image ID provided", 400

        # Check if the file exists in Firebase Storage
        bucket = storage.bucket()
        print(video_id)
        blob = bucket.blob(f'upload_file/videos/{video_id}.mp4')
        if not blob.exists():
            return "Video not found", 404
        
        video_url = blob.public_url

        insertion_bool = False
        db = mongo_client.vas
        collection = db['users']
        result = collection.find_one({'email': email_id})
        if not result:
            return "No user with this email found", 404

        # Access the utility functions and classes from the app context
        generate_unique_id = current_app.config['generate_unique_id']
        extract_embeddings_from_video = current_app.config['extract_embeddings_from_video']
        compare_embeddings = current_app.config['compare_embeddings']
        VideoProcessor = current_app.config['VideoProcessor']

        # Use the image URL instead of the local path
        

        video_directory = './videos/'

        # Download the video using requests
        response = requests.get(video_url)
        response.raise_for_status()  # Raise an error for bad responses
        if response.status_code == 200:
            video_path = os.path.join('./Firebasevideos/', f'{video_id}.mp4')
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            return "Failed to download video", 500

    cap = cv2.VideoCapture(video_path)
    model = YOLO(Config.FACE_DETECTION_MODEL_PATH)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    distinct_faces = []
    with tqdm(total=total_frames, desc="Processing Video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                results = model.predict(frame, verbose = False)
                face_bboxes = face_detection(results)
                for face in face_bboxes:
                    face_img = frame[int(face[1]):int(face[3]), int(face[0]):int(face[2])]
                    emb = extract_embeddings(face_img)
                    if emb is not None:
                        is_unique = True
                        for faces in distinct_faces:
                            if compare_embeddings(emb, faces):
                                is_unique = False
                                break
                
                        if is_unique:
                            distinct_faces.append(emb)
                pbar.update(1)
            else:
                break
    
    print(f"Found {len(distinct_faces)} distinct faces")
    print(f"Fetching videos from NVR")
    buc = storage.bucket()
    blobs = buc.list_blobs(prefix="NVR/")
    for file in blobs:
        if not file.name.endswith(".mp4"):
            print(f"Skipping {file.name}, as it does not appear to be a video file.")
            continue

        local_file_path = os.path.join(video_directory, os.path.basename(file.name))
        try:
            file.download_to_filename(local_file_path)
            print(f'Downloaded {file.name} to {local_file_path}')
        except Exception as e:
            print(f'Failed to download {file.name}: {e}')
    print("Download successful")
    def generate():
        start_time = time.time()
        video_count = 0
        frame_count = 0
        insertion_bool=False
        nvr_total = len([entry for entry in os.listdir(video_directory) if entry.endswith(".mp4")])
        metadata_insertion = {
            'up_image_id': video_id,
            'up_image': video_url,
            'detected': []
        }
        for video_filename in os.listdir(video_directory):
            if video_filename.endswith(".mp4"):
                video_count += 1
                video_path = os.path.join(video_directory, video_filename)
                video_processor = VideoProcessor(video_path, mongo_client)

                for i,faces in enumerate(distinct_faces):
                    print(f"Face {i}")
                    metadata = yield from video_processor.process_frame_on_video(
                        reference_embedding=faces,
                        email_id=email_id,
                        image_id=video_id,
                        image_path=frame,
                        video_path = video_url,  # Pass the URL here
                        video_count=video_count
                    )
                    frame_count += 1
                    if 'detected' in metadata:
                        metadata_insertion['detected'].extend(metadata['detected'])
        inserted_document = insert_metadata(metadata_insertion, email_id, mongo_client)
        if inserted_document and inserted_document.modified_count > 0:
            metadata['_id'] = str(inserted_document.upserted_id) if inserted_document.upserted_id else "existing_document"
        else:
            yield "data: Failed to insert metadata\n\n"
            insertion_bool = True                   
        if insertion_bool:
            yield "data: Duplicate key detected\n\n"
        end_time = time.time()
        
        execution_time_seconds = end_time - start_time

        minutes = int(execution_time_seconds // 60)
        seconds = int(execution_time_seconds % 60)

        print(f"Execution time: {minutes} minutes and {seconds} seconds")

        frame_response = {'completed': f"Processing of frame: {frame_count} completed across NVR"}
        yield f'data: {json.dumps(frame_response)}\n\n'

        cap.release()
        final_response = {'Video Completion': "Video processed!"}
        yield f'data: {json.dumps(final_response)}\n\n'

    return Response(generate(), mimetype='text/event-stream')
