import cv2
import os
import base64
import numpy as np
import json
import uuid
from PIL import Image
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
from services.embeddings import extract_embeddings, compare_embeddings
from mongo_client import insert_metadata
from config import Config

class VideoProcessor:
    def __init__(self, video_path, mongo_client):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.video_id = str(uuid.uuid4())  # Unique ID for the video
        self.output_path = f'processed_{self.video_id}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        self.mongo_client = mongo_client

        # Load YOLO model for face detection
        self.model = YOLO(Config.FACE_DETECTION_MODEL_PATH)
        self.mtcnn = MTCNN()
        self.resnet = InceptionResnetV1(pretrained='casia-webface').eval()

    def process_video(self, reference_embedding, email_id, image_id, image_path):
        yield 'data: Streaming API initialized\n\n'
        frame_count = 0
        # Loading received image
        try:
            image_ref = Image.open(image_path)
            image_ref = np.array(image_ref)
            image_ref_b64 = base64.b64encode(image_ref).decode('utf-8')
        except Exception as e:
            return f"Error loading received image by user",500
        metadata = {
          'up_image_id': image_id,
          'up_image': image_ref_b64,
          'detected':[]
        }
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            matched_any_face = False  # Flag to check if any face matches

            # Detect faces
            results = self.model(frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
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

                        unique_face_id = str(uuid.uuid4())
                        timestamp = frame_count / self.cap.get(cv2.CAP_PROP_FPS)
                        
                        # Convert the frame to a base64 string
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')

                        detected = {
                            'detected': match,
                            'face_id': unique_face_id,
                            'timestamp': timestamp,
                            'video_id': self.video_id,
                            'image': frame_base64,  # Add the base64 string to metadata
                        }
                        metadata['detected'].append(detected)
                        yield f'data: {json.dumps(metadata)}\n\n'  # Yield metadata for the face
            inserted_document = insert_metadata(metadata, email_id, self.mongo_client)
            if inserted_document and inserted_document.modified_count > 0:
                metadata['_id'] = str(inserted_document.upserted_id) if inserted_document.upserted_id else "existing_document"
            else:
                print("Failed to insert metadata")
            
                        
                      

            # Write the processed frame to the video
            self.out.write(frame)
            frame_count += 1

        self.cap.release()
        self.out.release()
