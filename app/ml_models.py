# ml_models.py
import os
import torch
import torchreid
import torchvision.transforms as T
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from .deep_sort_pytorch.deep_sort import DeepSort
from .deep_sort_pytorch.utils.parser import get_config

def init_models():
    face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0)

    os.makedirs("models", exist_ok=True)
    head_detector = YOLO('models/yolov8n.pt')
    
    cfg = get_config()
    cfg.merge_from_file("app/deep_sort_pytorch/configs/deep_sort.yaml")
    deep_sort = DeepSort(cfg.DEEPSORT.REID_CKPT, use_cuda=True)
    
    osnet = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        loss='softmax',
        pretrained=True
    )
    osnet.eval()
    osnet = osnet.cuda() if torch.cuda.is_available() else osnet
    
    return face_analyzer, head_detector, deep_sort, osnet

def release_models(face_analyzer, head_detector, deep_sort, osnet):
    del face_analyzer
    del head_detector
    del deep_sort
    del osnet