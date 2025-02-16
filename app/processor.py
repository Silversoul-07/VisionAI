from datetime import datetime
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from torch import le
from .database import Session
from .models import Person, Detection
from .utils import encode_frame
from ultralytics import YOLO
from boxmot import StrongSort

logger = logging.getLogger(__name__)

class PersonProcessor:
    def __init__(self, yolo: YOLO, tracker: StrongSort, similarity_threshold=0.5):
        self.yolo = yolo  # Ultralytics YOLOv5 detector
        self.tracker = tracker  # StrongSORT tracker
        self.tracker.max_iou_distance = 0.7    # Increase for more lenient matching
        self.tracker.max_age = 30              # Keep tracks alive longer
        self.tracker.n_init = 3                # Reduce frames needed to confirm track
        self.similarity_threshold = similarity_threshold
        self._setup_milvus_collection()

    def _setup_milvus_collection(self):
        """Create Milvus collection if it doesn't exist"""
        try:
            self.milvus_collection = Collection("embeddings")
        except Exception as e:
            fields = [
                FieldSchema(name="embedding_id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
            ]
            schema = CollectionSchema(fields=fields, description="Person embeddings collection")
            self.milvus_collection = Collection(name="embeddings", schema=schema)
            index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
            self.milvus_collection.create_index(field_name="embedding", index_params=index_params)
            self.milvus_collection.load()
            logger.info("Created new Milvus collection 'embeddings'")

    def process_frame(self, frame: np.ndarray, camera_id: str, timestamp: datetime) -> Tuple[List[Dict], str]:
        try:
            # Get YOLO detections
            results = self.yolo(frame)[0]
            if len(results.boxes) == 0:
                return [], encode_frame(frame)
    
            # Convert detections to numpy array
            dets = results.boxes.data.cpu().numpy()
            
            # Log confidence scores
            for det in dets:
                logger.info(f"YOLO detection confidence: {det[4]}")
            
            # Format detections for tracking [x1, y1, x2, y2, confidence, class]
            tracking_dets = dets[:, [0, 1, 2, 3, 4, 5]]
            logger.info(f"YOLO detections: {len(tracking_dets)}")
    
            # Get tracking results
            tracked_objects = self.tracker.update(tracking_dets, frame)
            logger.info(f"Tracked objects: {len(tracked_objects)}")
            
            # Log tracked object details
            for track in tracked_objects:
                    logger.info(f"Tracked object confidence: {track[5]}")
            
            if len(tracked_objects) == 0:
                return [], encode_frame(frame)
    
            # Format bounding boxes for feature extraction
            feature_boxes = []
            for track in tracked_objects:
                x1, y1, x2, y2 = track[:4]
                feature_boxes.append([x1, y1, x2, y2])
            feature_boxes = np.array(feature_boxes, dtype=np.float32)
            
            try:
                # Get embeddings using tracked bounding boxes
                embeddings = self.tracker.model.get_features(feature_boxes, frame)
                logger.debug(f"Embeddings shape: {embeddings.shape if embeddings is not None else 'None'}")
            except Exception as e:
                logger.error(f"Feature extraction error: {e}")
                embeddings = None
    
            # Process tracks and store in database
            results = self._process_tracks(tracked_objects, embeddings, camera_id, timestamp)
            result_frame = self._draw_results(frame, results, all_detections=dets)
            return results, encode_frame(result_frame)
        except Exception as e:
            logger.error(f"Frame processing error: {e}", exc_info=True)
            return [], encode_frame(frame)
    
    def _process_tracks(self, tracked_objects, embeddings, camera_id, timestamp):
        results = []
        filtered_count = 0
        logger.debug(f"Processing {len(tracked_objects)} tracks")
        logger.debug(f"Embeddings available: {embeddings is not None}")
        if embeddings is not None:
            logger.debug(f"Embeddings shape: {embeddings.shape}")
        
        with Session() as session:
            for track_idx, track in enumerate(tracked_objects):
                try:
                    # Extract tracking info [x1,y1,x2,y2,track_id,conf,class,idx]
                    x1, y1, x2, y2, track_id, conf, cls, idx = track
                    
                    # Skip low confidence detections
                    logger.debug(f"Processing track {track_idx} with confidence {conf}")
                    if conf < self.similarity_threshold:
                        filtered_count += 1
                        continue

                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    min_size = 30  # minimum pixel size
                    if bbox_width < min_size or bbox_height < min_size:
                        logger.debug(f"Detection too small: {bbox_width}x{bbox_height}")
                        continue
                        
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    
                    # Get embedding for this track
                    if embeddings is not None and track_idx < len(embeddings):
                        embedding = embeddings[track_idx]
                        # Ensure embedding is the right shape
                        if embedding.shape[0] != 512:
                            logger.warning(f"Wrong embedding dimension {embedding.shape[0]} for track {track_id}")
                            continue
                        
                        # Normalize embedding
                        embedding = embedding.astype(np.float32)
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                        else:
                            logger.warning(f"Zero norm embedding for track {track_id}")
                            continue
                    else:
                        logger.warning(f"No embedding available for track {track_id}")
                        continue
    
                    # Wrap person creation and detection in a single transaction
                    try:
                        # Search for matching embeddings
                        matches = self.milvus_collection.search(
                            data=[embedding.tolist()],
                            anns_field="embedding",
                            param={"metric_type": "L2", "params": {"nprobe": 10}},
                            limit=1
                        )
    
                        # Start transaction
                        session.begin_nested()
    
                        if matches and len(matches[0]) > 0 and matches[0][0].distance < self.similarity_threshold:
                            person_id = matches[0][0].id
                            # Verify person exists
                            person = session.query(Person).filter_by(person_id=person_id).first()
                            if not person:
                                # Create new person if not found
                                new_person = Person()
                                session.add(new_person)
                                session.flush()
                                person_id = new_person.person_id
                                
                                # Store embedding
                                self.milvus_collection.insert([{
                                    "embedding_id": str(person_id),
                                    "embedding": embedding.tolist()
                                }])
                        else:
                            # Create new person
                            new_person = Person()
                            session.add(new_person)
                            session.flush()
                            person_id = new_person.person_id
                            
                            # Store embedding
                            self.milvus_collection.insert([{
                                "embedding_id": str(person_id),
                                "embedding": embedding.tolist()
                            }])
    
                        # Create detection record
                        detection = Detection(
                            person_id=person_id,
                            camera_id=camera_id,
                            embedding_id=str(person_id),
                            timestamp=timestamp
                        )
                        session.add(detection)
                        
                        # Commit nested transaction
                        session.commit()
                        
                        results.append({
                            "person_id": person_id,
                            "bbox": bbox,
                            "track_id": int(track_id),
                            "confidence": float(conf)
                        })
    
                    except Exception as e:
                        logger.error(f"Database operation error for track {track_id}: {str(e)}")
                        session.rollback()
                        continue
    
                except Exception as e:
                    logger.error(f"Track processing error for track {track_idx}: {str(e)}")
                    continue
        logger.info(f"Processed {len(results)} detections, filtered out {filtered_count}")
        return results
    
    def _draw_results(self, frame, results, all_detections=None):
        result_frame = frame.copy()
        
        # Draw all YOLO detections in red
        if all_detections is not None:
            for det in all_detections:
                x1, y1, x2, y2 = map(int, det[:4])
                conf = det[4]
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red for all detections
                cv2.putText(result_frame, f"conf: {conf:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw tracked and processed detections in green
        for det in results:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for tracked
            cv2.putText(result_frame, 
                       f"ID: {det['track_id']} ({det['confidence']:.2f})", 
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return result_frame