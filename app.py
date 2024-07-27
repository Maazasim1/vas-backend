from flask import Flask, Response, request, jsonify
import cv2
import numpy as np
import os
import json
import uuid
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from ultralytics import YOLO
from flask_cors import CORS
import base64
import io
from pymongo import MongoClient
from mongo_client import insert_metadata

app = Flask(__name__)
CORS(app, origins="http://localhost:3000")

# Load YOLO model for face detection
model = YOLO("C:/Users/rohaa/Desktop/VAS/Models/faceDetection/face_detection.pt")

# Initialize face recognition model
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

# Paths for image and video files
received_images_path = 'ReceivedImages/'
video_path = 'C:/Users/rohaa/Desktop/VAS/test1.mp4'

# Ensure the ReceivedImages directory exists
if not os.path.exists(received_images_path):
    os.makedirs(received_images_path)

# Function to extract embeddings from an image
def extract_embeddings(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert('RGB')
    
    # Detect faces
    faces = mtcnn(img)
    if faces is not None:
        faces = faces.unsqueeze(0)
        embeddings = resnet(faces).detach()
        return embeddings
    else:
        return None

# Function to compare embeddings
def compare_embeddings(embedding1, embedding2, threshold=1.0):
    distance = torch.norm(embedding1 - embedding2).item()
    return distance < threshold

# Function to generate a unique ID
def generate_unique_id():
    return str(uuid.uuid4())

class VideoProcessor:
    def __init__(self, video_path, mongo_client):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.video_id = generate_unique_id()  # Unique ID for the video
        self.output_path = f'processed_{self.video_id}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        self.mongo_client = mongo_client

    def process_video(self, reference_embedding, email_id):
        yield 'data: Streaming API initialized\n\n'
        frame_count = 0
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            matched_any_face = False  # Flag to check if any face matches

            # Detect faces
            results = model(frame)
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

                        unique_face_id = generate_unique_id()
                        timestamp = frame_count / self.cap.get(cv2.CAP_PROP_FPS)
                        
                        # Convert the frame to a base64 string
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        metadata = {
                            'detected': match,
                            'face_id': unique_face_id,
                            'timestamp': timestamp,
                            'video_id': self.video_id,
                            'image': frame_base64,  # Add the base64 string to metadata
                        }
                        
                        # Insert metadata into MongoDB
                        inserted_document = insert_metadata(metadata, email_id, self.mongo_client)
                        if inserted_document and inserted_document.modified_count > 0:
                            metadata['_id'] = str(inserted_document.upserted_id) if inserted_document.upserted_id else "existing_document"
                        else:
                            print("Failed to insert metadata")
                        
                        yield f'data: {json.dumps(metadata)}\n\n'  # Yield metadata for the face

            # Write the processed frame to the video
            self.out.write(frame)
            frame_count += 1

        self.cap.release()
        self.out.release()

@app.route('/upload_file', methods=['POST'])
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
        image_path = os.path.join(received_images_path, f'{image_id}.jpg')
        with open(image_path, 'wb') as file:
            file.write(image_data)
    except Exception as e:
        return f"Error saving image: {e}", 500

    return "Image uploaded successfully", 200

@app.route('/stream', methods=['GET'])
def stream():
    image_id = request.args.get('image_id')
    email_id = request.args.get('email')
    if not image_id:
        return "No image ID provided", 400

    image_path = os.path.join(received_images_path, f'{image_id}.jpg')
    if not os.path.exists(image_path):
        return "Image not found", 404

    try:
        # Load the reference image
        image = Image.open(image_path)
        image = np.array(image)
    except Exception as e:
        return f"Error processing image: {e}", 500

    reference_embedding = extract_embeddings(image)
    if reference_embedding is None:
        return "No face detected in the reference image", 400

    # Path for video files
    video_directory = 'C:/Users/rohaa/Desktop/VAS/videos/'  # Update this path to your video directory

    def generate():
        for video_filename in os.listdir(video_directory):
            if video_filename.endswith(".mp4"):  # Assuming videos are in MP4 format
                video_path = os.path.join(video_directory, video_filename)
                video_processor = VideoProcessor(video_path, mongo_client)

                # Yield processed frames from the video
                yield from video_processor.process_video(reference_embedding=reference_embedding, email_id=email_id)

    # Return a streaming response with the generated data
    return Response(generate(), mimetype='text/event-stream')

@app.route('/fetch_all_metadata', methods=['GET'])
def fetch_all_metadata():
    metadata_collection = mongo_client.vas['vas-logger']
    
    try:
        documents = metadata_collection.find()
        metadata_list = []
        for doc in documents:
            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string for JSON serialization
            metadata_list.append(doc)
        return jsonify(metadata_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/get_metadata', methods=['GET'])
def get_metadata():
    email = request.args.get('email')
    if not email:
        return jsonify({"error": "Email parameter is missing"}), 400

    metadata_collection = mongo_client.vas['users']

    res_object = {}

    try:
        # Find the document with the specified email and project only the metadata_array field
        document = metadata_collection.find_one({'email': email}, {'_id': 0, 'metadata_array': 1})
        if not document or 'metadata_array' not in document:
            return jsonify({"error": "No metadata found for the given email"}), 404

        # Count the number of objects in the metadata_array
        metadata_array_length = len(document['metadata_array'])
        print(f"Number of objects in metadata_array: {metadata_array_length}")

        # Initialize an empty list to hold metadata entries
        metadata_list = []

        # Iterate through the metadata_array and process each item
        for item in document['metadata_array']:
            # Extract face_id and image from each item
            metadata_entry = {
                'face_id': item['face_id'],
                'image': item['image']  # Include image data
            }

            # Append the processed entry to the metadata_list
            metadata_list.append(metadata_entry)

        # Add the metadata_list to the response object
        res_object['metadata'] = metadata_list
        # print("Response object: ", res_object)  # Debugging: Print the response object

        # Return the response object as JSON
        return jsonify(res_object), 200

    except Exception as e:
        # Handle and log any exceptions that occur
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize MongoDB client once at startup
    mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://mazzasimq1:vFjQU2EwpPEbRwGf@cluster0.aigbuff.mongodb.net/vas")
    mongo_client = MongoClient(mongo_uri)
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
