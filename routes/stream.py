# routes/stream.py
from flask import Blueprint, Response, request, current_app
import os
import base64
from PIL import Image
import numpy as np
import io
from firebase_admin import storage

bp = Blueprint('stream', __name__)

@bp.route('/stream', methods=['GET'])
def stream():
    with current_app.app_context():
        mongo_client = current_app.config['mongo_client']
        image_id = request.args.get('image_id')
        email_id = request.args.get('email')
        if not image_id:
            return "No image ID provided", 400

        # Check if the file exists in Firebase Storage
        bucket = storage.bucket()
        print(image_id)
        blob = bucket.blob(f'upload_file/{image_id}.jpg')
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
            for video_filename in os.listdir(video_directory):
                if video_filename.endswith(".mp4"):
                    video_path = os.path.join(video_directory, video_filename)
                    video_processor = VideoProcessor(video_path, mongo_client)

                    res = yield from video_processor.process_video(
                        reference_embedding=reference_embedding,
                        email_id=email_id,
                        image_id=image_id,
                        image_path=image_url,  # Pass the URL here
                        insertion_bool=insertion_bool
                    )
                    if res:
                        yield "data: Duplicate key detected\n\n"
                        break

        return Response(generate(), mimetype='text/event-stream')