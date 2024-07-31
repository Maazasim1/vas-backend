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
        blob = bucket.blob(f"upload_file/{image_id}.jpg")
        blob.upload_from_string(image_data, content_type='image/jpg')

        # Get the public URL
        blob.make_public()
        image_url = blob.public_url

        return jsonify({"message": "Image uploaded successfully", "image_url": image_url}), 200
    except Exception as e:
        return f"Error uploading image to Firebase: {e}", 500
