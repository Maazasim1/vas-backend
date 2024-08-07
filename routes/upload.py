from flask import Blueprint, request, jsonify, current_app
import os
import base64
from firebase_admin import storage
from config import Config

bp = Blueprint('upload', __name__)

# Ensure the directory exists
os.makedirs('ReceivedImages', exist_ok=True)

@bp.route('/upload_file', methods=['POST'])
def upload_file():
    req_data = request.get_json()
    if req_data['image']:
        image_base64 = req_data['image']
        image_id = req_data['image_id']
        received_images_path = Config.RECEIVED_IMAGES_PATH
        print("Image id: ", image_id)
        if not image_base64 or not image_id:
            return "Both 'image' and 'image_id' must be provided", 400

        try:
            # Decode the base64 string
            image_data = base64.b64decode(image_base64)
            image_path = os.path.join(received_images_path, f'{image_id}.jpg')

            # # Save the image locally
            # with open(image_path, 'wb') as image_file:
            #     image_file.write(image_data)

            # Upload the image to Firebase Storage
            bucket = storage.bucket()
            blob = bucket.blob(f"upload_file/images/{image_id}.jpg")
            blob.upload_from_string(image_data, content_type='image/jpg')

            # Get the public URL
            blob.make_public()
            image_url = blob.public_url

            return jsonify({"message": "Image uploaded successfully", "image_url": image_url}), 200
        except Exception as e:
            return f"Error uploading image to Firebase: {e}", 500
        
    elif req_data['video']:
        video_base64 = req_data['video']
        video_id = req_data['video_id']
        print("Video_ID: ", video_id)
        if not video_base64 or not video_id:
            return "Both 'video' and 'video_id' must be provided", 400
        try:
            video_data = base64.b64decode(video_base64)
            bucket = storage.bucket()
            blob = bucket.blob(f"upload_file/videos/{video_id}.mp4")
            blob.upload_from_string(video_data, content_type='video/mp4')
            blob.make_public()
            video_url = blob.public_url
            return jsonify({"message":"Video uploaded successfully", "video_url":video_url}), 200
        except Exception as e:
            return f"Error uploading video to Firebase: {e}", 500
