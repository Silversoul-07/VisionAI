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
        # Process both faces and heads
        faces = face_analyzer.get(frame_data)
        heads = head_detector(frame_data, classes=[0])
        detections, embeddings = [], []

        # Process faces first
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            if embedding is not None:
                detections.append(bbox)
                embeddings.append(embedding)

        # Then process heads
        for head in heads:
            for box in head.boxes.data:
                bbox = box[:4].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                
                # Clip coordinates
                h, w = frame_data.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                roi = frame_data[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                try:
                    embedding = extract_features(roi, osnet)
                    if embedding.size > 0:  # Check if embedding was successfully extracted
                        detections.append([x1, y1, x2, y2])
                        embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Head processing error: {e}")

        print(f"Detected {len(detections)} total objects (faces + heads)")
        
        # Convert detections to XYWH format for DeepSort
        detections_for_deepsort = []
        valid_embeddings = []
        for bbox, embedding in zip(detections, embeddings):
            try:
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                # DeepSort expects [x_center, y_center, width, height]
                x_center = x1 + w/2
                y_center = y1 + h/2
                detections_for_deepsort.append([x_center, y_center, w, h])
                valid_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Invalid bbox conversion: {bbox} - {e}")

        print(f"Valid detections for tracking: {len(detections_for_deepsort)}")
                
        # Update tracking with proper bbox format
        results = []
        if detections_for_deepsort:
            try:
                detections_array = np.array(detections_for_deepsort, dtype=np.float32)
                scores_array = np.ones(len(detections_for_deepsort), dtype=np.float32)
                classes_array = np.zeros(len(detections_for_deepsort), dtype=np.int32)  # Changed to int32
                
                outputs, _ = deep_sort.update(  # Capture both return values
                    detections_array,
                    scores_array,
                    classes_array,
                    frame_data
                )
                tracked_objects = outputs if outputs is not None else []
                
                print(f"Tracked objects: {len(tracked_objects)}")  # Add debug print
            except Exception as e:
                logger.error(f"DeepSort error: {str(e)}")
                tracked_objects = []

        # ... rest of your code for processing tracked objects ...
            
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
                    print(results[-1])  # Add debug print
                except Exception as e:
                    logger.error(f"Track processing error: {e}")
                    continue
                    
                try:
                    session.commit()
                except Exception as e:
                    logger.error(f"Commit error: {e}")
                    session.rollback()

        print(f"Tracking results: {results}")
                    
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