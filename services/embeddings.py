import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import requests
import io

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

def extract_embeddings(image_input):
    # Check if input is a URL
    if isinstance(image_input, str):
        response = requests.get(image_input)
        response.raise_for_status()  # Raise an error for bad responses
        img = Image.open(io.BytesIO(response.content))
    # Check if input is a NumPy array
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        raise ValueError("Unsupported image input type")

    img = img.convert('RGB')
    
    # Detect faces
    faces = mtcnn(img)
    if faces is not None:
        faces = faces.unsqueeze(0)
        embeddings = resnet(faces).detach()
        return embeddings
    else:
        return None

def extract_embeddings_from_video(frame):
    faces = mtcnn(frame)
    if faces is not None:
        faces = faces.unsqueeze(0)
        embeddings = resnet(faces).detach()
        return embeddings
    else:
        return None

def compare_embeddings(embedding1, embedding2, threshold=1.0):
    distance = torch.norm(embedding1 - embedding2).item()
    return distance < threshold
