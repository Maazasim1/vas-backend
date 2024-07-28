from pymongo import MongoClient
import os

def insert_metadata(metadata, email_id, mongo_client):
    print("Preparing to insert data")
    db = mongo_client.vas
    metadata_collection = db['users']
    try:
        print("Metadata to be inserted: ", metadata)
        result = metadata_collection.update_one({'email': email_id}, {"$push": {"metadata_array": metadata}})
        print("Insertion result: ", result)
        return result
    except Exception as e:
        print(f"Error inserting metadata: {e}")
        return None

