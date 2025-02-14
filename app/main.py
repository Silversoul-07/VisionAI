from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from ultralytics import YOLO
import redis
from pymilvus import connections, Collection, utility
from sqlalchemy import create_engine, Column, Integer, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import json
import torch
import torchvision.transforms as T
import torchreid
from typing import List, Dict
import base64

# Deep SORT imports remain the same
import sys
sys.path.append('deep_sort_pytorch/utils')
from .deep_sort_pytorch.deep_sort import DeepSort
from .deep_sort_pytorch.utils.parser import get_config

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
engine = create_engine('postgresql://user:pass@localhost/tracking_db')
Base = declarative_base()

class TrackingRecord(Base):
    __tablename__ = 'tracking_records'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    camera_id = Column(String)
    embedding_id = Column(String)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Initialize models
face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0)
head_detector = YOLO('yolov8n.pt')

# Initialize Deep SORT and OSNet (same as original)
cfg = get_config()
cfg.merge_from_file("app/deep_sort_pytorch/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE,
                    n_init=cfg.DEEPSORT.N_INIT,
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

osnet = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    loss='softmax',
    pretrained=True
)
osnet.eval()
if torch.cuda.is_available():
    osnet = osnet.cuda()

transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DetectionResult(BaseModel):
    bbox: List[float]
    track_id: int
    confidence: float

def extract_features(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0)
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy()

def process_frame(frame_data: np.ndarray, camera_id: str, timestamp: datetime):
    faces = face_analyzer.get(frame_data)
    heads = head_detector(frame_data, classes=[0])
    
    detections = []
    embeddings = []
    
    # Process faces
    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        detections.append(bbox)
        embeddings.append(embedding)
    
    # Process heads if no faces found
    if not faces:
        for head in heads:
            for box in head.boxes.data:
                bbox = box[:4].cpu().numpy().astype(int)
                confidence = box[4].item()
                roi = frame_data[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if roi.size > 0:
                    try:
                        embedding = extract_features(roi, osnet)
                        detections.append(bbox)
                        embeddings.append(embedding.flatten())
                    except Exception:
                        continue
    
    if detections:
        tracked_objects = deepsort.update(np.array(detections), embeddings, frame_data)
    else:
        tracked_objects = np.array([])
    
    # Store tracking data
    results = []
    with Session() as session:
        for (x1, y1, x2, y2, track_id), embedding in zip(tracked_objects, embeddings):
            embedding_id = f"{camera_id}_{track_id}"
            
            collection = Collection("embeddings")
            collection.insert([
                {"embedding_id": embedding_id, "embedding": embedding}
            ])
            
            record = TrackingRecord(
                timestamp=timestamp,
                camera_id=camera_id,
                embedding_id=embedding_id
            )
            session.add(record)
            
            results.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "track_id": int(track_id),
                "embedding_id": embedding_id
            })
        session.commit()
    
    # Draw results on frame
    result_frame = frame_data.copy()
    for det in results:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_frame, f"ID: {det['track_id']}", 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    _, buffer = cv2.imencode('.jpg', result_frame)
    result_image = base64.b64encode(buffer).decode('utf-8')
    
    return results, f"data:image/jpeg;base64,{result_image}"

@app.post("/process_image")
async def process_image(
    file: UploadFile = File(...),
    camera_id: str = "default"
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    timestamp = datetime.now()
    cache_key = f"{camera_id}_{timestamp.timestamp()}"
    
    if cached := redis_client.get(cache_key):
        return json.loads(cached)
    
    detections, result_image = process_frame(image, camera_id, timestamp)
    
    result = {
        "detections": detections,
        "timestamp": timestamp.isoformat(),
        "result_image": result_image
    }
    
    redis_client.setex(cache_key, 300, json.dumps(result))
    return result