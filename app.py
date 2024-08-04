# app.py
from flask import Flask
from flask_cors import CORS
from pymongo import MongoClient
from config import Config
from routes import default, upload, stream, metadata
from firebase.firebase_config import initialize_firebase
from services.utils import generate_unique_id
from services.embeddings import extract_embeddings, compare_embeddings
from services.video_processor import VideoProcessor


app = Flask(__name__)
CORS(app, origins=["https://vas-frontend.vercel.app","http://localhost:3000"])

    # Initialize MongoDB client once at startup
mongo_client = MongoClient(Config.MONGO_URI)
app.config['mongo_client'] = mongo_client
app.config['generate_unique_id'] = generate_unique_id
app.config['extract_embeddings'] = extract_embeddings
app.config['compare_embeddings'] = compare_embeddings
app.config['VideoProcessor'] = VideoProcessor
        # app.config['insertion_bool'] = False

        # Initialize Firebase
initialize_firebase()

        # Register blueprints
app.register_blueprint(default.bp)
app.register_blueprint(upload.bp)
app.register_blueprint(stream.bp)
app.register_blueprint(metadata.bp)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)