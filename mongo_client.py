from pymongo import MongoClient
import os

def insert_metadata(metadata, email_id, mongo_client):
    print("Preparing to insert data")
    db = mongo_client.vas
    metadata_collection = db['users']

    try:
        print("Metadata to be inserted: ", metadata)
        # Update the document if it exists, otherwise insert it
        result = metadata_collection.update_one(
            {'email': email_id, 'metadata_array.up_image_id': {'$ne': metadata['up_image_id']}}, 
            {"$push": {"metadata_array": metadata}}, 
            upsert=True
        )
        print("Insertion result: ", result)
        return result
    except Exception as e:
        print(f"Error inserting metadata: {e}")
        return None


