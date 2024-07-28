import firebase_admin
from firebase_admin import credentials, storage

def initialize_firebase():
    cred = credentials.Certificate('firebase/vas-a6205-firebase-adminsdk-6493v-f6706c0159.json')
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'vas-a6205.appspot.com'
    })
