# routes/stream.py
from flask import Blueprint, Response, request, current_app, json
import requests
import os
import base64
from PIL import Image
import numpy as np
import io
from firebase_admin import storage
import cv2
from config import Config
from ultralytics import YOLO
from services.utils import face_detection
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

        video_directory = './videos/'
        def generate():
            video_count = 0
            nvr_total = len([entry for entry in os.listdir(video_directory) if entry.endswith(".mp4")])
            for video_filename in os.listdir(video_directory):
                if video_filename.endswith(".mp4"):
                    video_count+=1
                    video_path = os.path.join(video_directory, video_filename)
                    video_processor = VideoProcessor(video_path, mongo_client)

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
            final_response = {'completed':"Processing on NVR completed!"}
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

    def generate():
        video_count = 0
        frame_count = 0
        nvr_total = len([entry for entry in os.listdir(video_directory) if entry.endswith(".mp4")])

        while True:
            success, frame = cap.read()
            if not success:
                break
            reference_embedding = extract_embeddings_from_video(frame)
            if reference_embedding is None:
                return "No face detected in the reference image", 400
            results = model(frame)
            face_bboxes = face_detection(results)

            for video_filename in os.listdir(video_directory):
                if video_filename.endswith(".mp4"):
                    video_count += 1
                    video_path = os.path.join(video_directory, video_filename)
                    video_processor = VideoProcessor(video_path, mongo_client)

                    res = yield from video_processor.process_frame_on_video(
                        reference_embedding=reference_embedding,
                        email_id=email_id,
                        image_id=video_id,
                        image_path=frame,
                        video_path = video_url,  # Pass the URL here
                        insertion_bool=insertion_bool,
                        video_count=video_count
                    )
                    frame_count += 1
                    if res:
                        yield "data: Duplicate key detected\n\n"
                        break

            frame_response = {'completed': f"Processing of frame: {frame_count} completed across NVR"}
            yield f'data: {json.dumps(frame_response)}\n\n'

        cap.release()
        final_response = {'Video Completion': "Video processed!"}
        yield f'data: {json.dumps(final_response)}\n\n'

    return Response(generate(), mimetype='text/event-stream')
