# processor.py
from datetime import datetime
from pymilvus import Collection
from .database import Session
from .models import TrackingRecord
from .utils import extract_features, encode_frame
import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict
from .ml_models import DeepSort
from PIL import Image  # add at the top if not already imported

logger = logging.getLogger(__name__)

def process_frame(
    frame_data: np.ndarray,
    camera_id: str,
    timestamp: datetime,
    face_analyzer,
    head_detector,
    deep_sort: DeepSort,
    osnet
) -> Tuple[List[Dict], str]:
    try:
        faces = face_analyzer.get(frame_data)
        heads = head_detector(frame_data, classes=[0])
        detections, embeddings = [], []
        for head in heads:
            for box in head.boxes.data:
                bbox = box[:4].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                print(f"Head detected: {bbox}")
                
                # Clip coordinates to be within frame dimensions
                h, w = frame_data.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                roi = frame_data[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                try:
                    # Ensure ROI is a NumPy array (safeguard)
                    roi_np = np.asarray(roi)
                    # Remove conversion to PIL Image and pass roi_np directly
                    embedding = extract_features(roi_np, osnet)
                    detections.append(bbox)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Head processing error: {e}")                
        # Process detections
        detections_for_deepsort = []
        valid_embeddings = []
        for bbox, embedding in zip(detections, embeddings):
            try:
                x1, y1, x2, y2 = bbox
                detections_for_deepsort.append([x1, y1, x2-x1, y2-y1])
                valid_embeddings.append(embedding)
            except ValueError as e:
                logger.error(f"Invalid bbox: {bbox} - {e}")
                
        # Update tracking
        results = []
        if detections_for_deepsort:
            tracked_objects = deep_sort.update(
                np.array(detections_for_deepsort),
                np.ones(len(detections_for_deepsort)),
                valid_embeddings,
                frame_data
            )
            
            with Session() as session:
                for track in tracked_objects:
                    if not hasattr(track, 'to_tlbr') or not hasattr(track, 'track_id'):
                        continue
                        
                    try:
                        x1, y1, x2, y2 = map(int, track.to_tlbr())
                        track_id = track.track_id
                        embedding = np.mean(track.features, axis=0).tolist()
                        
                        embedding_id = f"{camera_id}_{track_id}"
                        Collection("embeddings").insert([{
                            "embedding_id": embedding_id,
                            "embedding": embedding
                        }])
                        
                        session.add(TrackingRecord(
                            timestamp=timestamp,
                            camera_id=camera_id,
                            embedding_id=embedding_id
                        ))
                        
                        results.append({
                            "bbox": [x1, y1, x2, y2],
                            "track_id": track_id,
                            "embedding_id": embedding_id
                        })
                    except Exception as e:
                        logger.error(f"Track processing error: {e}")
                        continue
                        
                try:
                    session.commit()
                except Exception as e:
                    logger.error(f"Commit error: {e}")
                    session.rollback()
                    
        # Draw results
        try:
            result_frame = frame_data.copy()
            for det in results:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    result_frame,
                    f"ID: {det['track_id']}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )
        except Exception as e:
            logger.error(f"Drawing error: {e}")
            
        return results, encode_frame(result_frame)
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        return [], ""