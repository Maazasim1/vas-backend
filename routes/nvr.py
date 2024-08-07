from flask import Blueprint, jsonify, current_app,request
from firebase_admin import storage
import datetime
import base64

bp = Blueprint('nvr', __name__)

@bp.route('/view_all_videos', methods=['GET'])
def view_all_videos():
    try:
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix="NVR/")
        video_urls = []
        
        for blob in blobs:
            if blob.name.endswith('.mp4'):
                # Generate a signed URL for the blob
                url = blob.generate_signed_url(expiration=datetime.timedelta(hours=1))  # URL valid for 1 hour
                video_urls.append(url)
        
        return jsonify({"urls": video_urls})
    except Exception as e:
        current_app.logger.error(f"Error fetching video URLs: {e}")
        return jsonify({"error": "Failed to retrieve video URLs"}), 500


@bp.route('/add_videos', methods=['POST'])
def add_videos():
    req_data = request.get_json()
    video_base64 = req_data['video']
    video_id = req_data['video_id']
    print("Video_ID: ", video_id)
    if not video_base64 or not video_id:
        return "Both 'video' and 'video_id' must be provided", 400
    try:
        video_data = base64.b64decode(video_base64)
        bucket = storage.bucket()
        blob = bucket.blob(f"NVR/{video_id}.mp4")
        blob.upload_from_string(video_data, content_type='video/mp4')
        blob.make_public()
        video_url = blob.public_url
        return jsonify({"message":"Video uploaded successfully", "video_url":video_url}), 200
    except Exception as e:
        return f"Error uploading video to NVR: {e}", 500