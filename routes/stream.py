# routes/stream.py
from flask import Blueprint, Response, request, current_app
import os
import base64
from PIL import Image
import numpy as np

bp = Blueprint('stream', __name__)

@bp.route('/stream', methods=['GET'])
def stream():
    image_id = request.args.get('image_id')
    email_id = request.args.get('email')
    if not image_id:
        return "No image ID provided", 400

    image_path = os.path.join('ReceivedImages', f'{image_id}.jpg')
    if not os.path.exists(image_path):
        return "Image not found", 404

    try:
        # Load the reference image
        image = Image.open(image_path)
        image = np.array(image)
    except Exception as e:
        return f"Error processing image: {e}", 500

    # Access the utility functions and classes from the app context
    generate_unique_id = current_app.config['generate_unique_id']
    extract_embeddings = current_app.config['extract_embeddings']
    compare_embeddings = current_app.config['compare_embeddings']
    VideoProcessor = current_app.config['VideoProcessor']

    reference_embedding = extract_embeddings(image)
    if reference_embedding is None:
        return "No face detected in the reference image", 400

    mongo_client = current_app.config['mongo_client']
    video_directory = 'C:/Users/rohaa/Desktop/VAS/videos/'

    def generate():
        for video_filename in os.listdir(video_directory):
            if video_filename.endswith(".mp4"):
                video_path = os.path.join(video_directory, video_filename)
                video_processor = VideoProcessor(video_path, mongo_client)

                yield from video_processor.process_video(reference_embedding=reference_embedding, email_id=email_id, image_id=image_id, image_path=image_path)

    return Response(generate(), mimetype='text/event-stream')
