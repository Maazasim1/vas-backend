import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

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

def compare_embeddings(embedding1, embedding2, threshold=1.0):
    distance = torch.norm(embedding1 - embedding2).item()
    return distance < threshold
