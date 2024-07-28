import os

class Config:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://mazzasimq1:vFjQU2EwpPEbRwGf@cluster0.aigbuff.mongodb.net/vas")
    RECEIVED_IMAGES_PATH = 'ReceivedImages/'
    VIDEO_DIRECTORY = './videos/'
    FACE_DETECTION_MODEL_PATH = "./Models/faceDetection/face_detection.pt"
