from pymongo import MongoClient
import os

def insert_metadata(metadata, email_id):
    print("Preparing to insert data")
    mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://mazzasimq1:vFjQU2EwpPEbRwGf@cluster0.aigbuff.mongodb.net/vas")
    print("Mongo URI: ", mongo_uri)
    client = MongoClient(mongo_uri)
    db = client.vas
    metadata_collection = db['users']
    try:
        print("Metadata to be inserted: ", metadata)
        result = metadata_collection.update_one({'email': email_id}, {"$push": {"metadata_array": metadata}}, upsert=True)
        print("Insertion result: ", result)
        return result
    except Exception as e:
        print(f"Error inserting metadata: {e}")
        return None
