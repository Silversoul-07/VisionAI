import os
import torch
from ultralytics import YOLO
from boxmot import BotSort
from pathlib import Path

def init_models():
    os.makedirs("models", exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    detector = YOLO('models/yolov8l.pt').to(device)
    
    tracker = BotSort(
        reid_weights=Path('models/osnet_ibn_x1_0_msmt17.pt'),
        device=0,
        half=True,
        track_high_thresh=0.3,  # Lower this threshold
        track_low_thresh=0.1,
        new_track_thresh=0.4,  # Lower this threshold
        track_buffer=60,
        match_thresh=0.7,  # Adjust this threshold
        proximity_thresh=0.3,
        appearance_thresh=0.25,
        frame_rate=30,
        fuse_first_associate=True,  # Try setting this to True
        with_reid=True
    )
    
    return detector, tracker

def release_models(detector, tracker):
    del detector
    del tracker