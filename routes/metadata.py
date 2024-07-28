# routes/metadata.py
from flask import Blueprint, jsonify, request, current_app

bp = Blueprint('metadata', __name__)

@bp.route('/fetch_all_metadata', methods=['GET'])
def fetch_all_metadata():
    mongo_client = current_app.config['mongo_client']
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

@bp.route('/get_metadata', methods=['GET'])
def get_metadata():
    email = request.args.get('email')
    if not email:
        return jsonify({"error": "Email parameter is missing"}), 400

    mongo_client = current_app.config['mongo_client']
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
