# utils.py
import cv2
import numpy as np
import torch
import base64
import logging
from PIL import Image
from torchvision import transforms as T

logger = logging.getLogger(__name__)

transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image: np.ndarray, model) -> np.ndarray:
    try:
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert the NumPy array to a PIL Image
        pil_img = Image.fromarray(image)
        # Apply the transform pipeline
        image = transform(pil_img).unsqueeze(0)
        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            features = model(image)
        return features.cpu().numpy().flatten()
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return np.array([])

def encode_frame(frame: np.ndarray) -> str:
    try:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]  # Reduce quality from default 95 to 80
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    except Exception as e:
        logger.error(f"Frame encoding error: {e}")
        return ""
    
# provide code to extract initial frame from video and save to path
def extract_initial_frame(video_path: str, save_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(save_path, frame)
    cap.release()

def resize_frame(frame, width=640):
    height = int(frame.shape[0] * (width/frame.shape[1]))
    return cv2.resize(frame, (width, height))