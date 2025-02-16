import os
from ultralytics import YOLO
from boxmot import StrongSort
from pathlib import Path

def init_models():
    os.makedirs("models", exist_ok=True)
    
    # Load YOLOv8m with TensorRT acceleration
    detector = YOLO('models/yolov8l.pt')
    
    # Initialize StrongSORT for tracking
    tracker = StrongSort(
        reid_weights=Path('models/osnet_x0_25_msmt17.pt'),
        device='cpu',
        half=False,
        )
    
    return detector, tracker

def release_models(detector, tracker):
    del detector
    del tracker