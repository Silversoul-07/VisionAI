import os
import torch
from ultralytics import YOLO
from boxmot import BotSort
from pathlib import Path

def init_models():
    os.makedirs("models", exist_ok=True)
    
    # find lighter models for faster inference and higher accuracy
    detector = YOLO('models/yolov8l.pt')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    # experiment eith different algorithms, weights, and configurations
    tracker = BotSort(
        reid_weights=Path('models/osnet_ibn_x1_0_msmt17.pt'),
        device=device,
        half=False, # setup gpu and use half precision
    )
    
    return detector, tracker

def release_models(detector, tracker):
    del detector
    del tracker