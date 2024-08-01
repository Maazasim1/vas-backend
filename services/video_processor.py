import cv2
import os
import base64
import numpy as np
import json
import uuid
import io
from PIL import Image
import requests
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
from services.embeddings import extract_embeddings, compare_embeddings
from mongo_client import insert_metadata
from config import Config
from services.utils import face_detection
from firebase_admin import storage
from datetime import datetime
from flask import current_app

class VideoProcessor:
    def __init__(self, video_path, mongo_client):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.video_id = str(uuid.uuid4())  # Unique ID for the video
        self.mongo_client = mongo_client

        # Load YOLO model for face detection
        self.model = YOLO(Config.FACE_DETECTION_MODEL_PATH)
        self.mtcnn = MTCNN()
        self.resnet = InceptionResnetV1(pretrained='casia-webface').eval()

    def process_video(self, reference_embedding, email_id, image_id, image_path, insertion_bool):
        yield 'data: Streaming API initialized\n\n'

        frame_count = 0
        # Loading received image from URL
        try:
            response = requests.get(image_path)
            response.raise_for_status()  # Raise an error for bad responses
            image_ref = Image.open(io.BytesIO(response.content))
            image_ref = image_ref.convert("RGBA")
            image_ref = np.array(image_ref)
            # Convert RGBA to BGR
            image_ref = cv2.cvtColor(image_ref, cv2.COLOR_RGBA2BGR)
            _, image_ref_b64 = cv2.imencode('.jpg', image_ref)
            image_ref_b64 = base64.b64encode(image_ref_b64).decode('utf-8')
        except Exception as e:
            yield f"data: Error loading received image: {str(e)}\n\n"
            return

        image_path = f'{image_id}.jpg'
        bucket = storage.bucket()

        try:
            image_ref_b64_bytes = base64.b64decode(image_ref_b64)
            blob_up = bucket.blob(f"stream/uploaded/{image_id}.jpg")
            blob_up.upload_from_string(image_ref_b64_bytes, content_type='image/jpg')
            blob_up.make_public()
            up_image_url = blob_up.public_url
        except Exception as e:
            yield f"data: Error while uploading to stream/uploaded: {str(e)}\n\n"
            return

        metadata = {
            'up_image_id': image_id,
            'up_image': up_image_url,
            'detected': []
        }

        while True:
            success, frame = self.cap.read()
            frame_count += 1
            if not success:
                break

            matched_any_face = False  # Flag to check if any face matches

            # Detect faces
            results = self.model(frame)
            face_bboxes = face_detection(results)                   #CHANGED
            for box in face_bboxes:
                cur_date_time = datetime.now()
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]     #CHANGED
                face = frame[int(y1):int(y2), int(x1):int(x2)]

                # Extract features from the detected face
                face_embedding = extract_embeddings(face)
                if face_embedding is not None:
                    match = compare_embeddings(reference_embedding, face_embedding)
                    matched_any_face = matched_any_face or match  # Update flag if any face matches
                    if match:
                        # Draw a green bounding box around the detected face
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    else:
                        # Draw a red bounding box around the detected face
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        continue

                    unique_face_id = str(uuid.uuid4())

                    # Convert the frame to RGB before encoding
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    _, buffer = cv2.imencode('.jpg', frame_rgb)
                    
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    try:
                        frame_base64_bytes = base64.b64decode(frame_base64)
                        blob_res = bucket.blob(f"stream/results/{unique_face_id}.jpg")
                        blob_res.upload_from_string(frame_base64_bytes, content_type='image/jpg')
                        blob_res.make_public()
                        res_image_url = blob_res.public_url
                    except Exception as e:
                        yield f"data: Error while uploading to stream/results: {str(e)}\n\n"
                        return

                    detected = {
                        'frame_count': frame_count,
                        'detected': match,
                        'face_id': unique_face_id,
                        'timestamp': str(cur_date_time),
                        'video_id': self.video_id,
                        'image': res_image_url,  # Add the base64 string to metadata
                    }
                    metadata['detected'].append(detected)
                    yield f'data: {json.dumps(detected)}\n\n'  # Yield metadata for the face

        inserted_document = insert_metadata(metadata, email_id, self.mongo_client)
        if inserted_document and inserted_document.modified_count > 0:
            metadata['_id'] = str(inserted_document.upserted_id) if inserted_document.upserted_id else "existing_document"
        else:
            yield "data: Failed to insert metadata\n\n"
            insertion_bool = True
            return insertion_bool

        self.cap.release()
        yield 'data: Video processing completed\n\n'
