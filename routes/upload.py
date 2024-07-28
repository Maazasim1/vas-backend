from flask import Blueprint, request
import os
import base64
from config import Config

bp = Blueprint('upload', __name__)

@bp.route('/upload_file', methods=['POST'])
def upload_file():
    req_data = request.get_json()
    image_base64 = req_data['image']
    image_id = req_data['image_id']
    print("Image id: ", image_id)
    if not image_base64 or not image_id:
        return "Both 'image' and 'image_id' must be provided", 400

    try:
        # Decode the base64 string and save the image
        image_data = base64.b64decode(image_base64)
        image_path = os.path.join(Config.RECEIVED_IMAGES_PATH, f'{image_id}.jpg')
        with open(image_path, 'wb') as file:
            file.write(image_data)
    except Exception as e:
        return f"Error saving image: {e}", 500

    return "Image uploaded successfully", 200
