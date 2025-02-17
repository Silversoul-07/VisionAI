from datetime import datetime
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from .database import Session
from .models import Person, Detection
from .utils import encode_frame
from ultralytics import YOLO
from boxmot import BotSort

logger = logging.getLogger(__name__)

class PersonProcessor:
    def __init__(self, yolo: YOLO, tracker: BotSort, similarity_threshold=0.5):
        self.yolo = yolo  # Ultralytics YOLOv5 detector
        self.tracker = tracker  # StrongSORT tracker
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

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        try:
            # Get YOLO detections
            results = self.yolo(frame)[0]
            if len(results.boxes) == 0:
                return []
    
            # Convert detections to numpy array
            dets = results.boxes.data.cpu().numpy()
            
            # Filter only person detections (class 0 in COCO dataset)
            person_dets = dets[dets[:, 5] == 0]
            
            if len(person_dets) == 0:
                return []
                
            # Format detections for tracking [x1, y1, x2, y2, conf, class]
            tracking_dets = np.zeros((len(person_dets), 6))
            tracking_dets[:, 0:4] = person_dets[:, 0:4]  # bbox coordinates
            tracking_dets[:, 4] = person_dets[:, 4]      # confidence scores
            tracking_dets[:, 5] = 0                       # class ID (person)
            
            logger.debug(f"Detection shape: {tracking_dets.shape}")
            logger.debug(f"Detection sample: {tracking_dets[0] if len(tracking_dets) > 0 else 'No detections'}")
            
            # Update tracker with detections
            # Returns: M x (x1, y1, x2, y2, id, conf, cls, idx)
            tracked_objects = self.tracker.update(tracking_dets, frame)
            logger.info(f"Tracked objects: {len(tracked_objects)}")
            
            # Format results
            results = []
            for track in tracked_objects:
                x1, y1, x2, y2, track_id, conf, cls, idx = track
                
                if conf < self.similarity_threshold:
                    continue
                    
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                results.append({
                    "track_id": int(track_id),
                    "bbox": bbox,
                    "confidence": float(conf),
                    "class": int(cls)
                })
    
            return results
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}", exc_info=True)
            return []
    
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

    def track_person(self, frame: np.ndarray, target_track_id: int, camera_id: str, timestamp: datetime) -> Tuple[Dict, str]:
        '''Process a single track ID in the frame, ignoring all others
        Returns: Tuple of (track_result, annotated_frame)
        '''
        try:
            # Get all tracks first
            tracked_results = self.process_frame(frame)
            
            # Find the target track
            target_track = None

            for track in tracked_results:
                if track["track_id"] == target_track_id:
                    print("found")
                    target_track = track
                    break
                    
            if target_track is None:
                logger.error("No target track found")
                return {}, encode_frame(frame)
                
            # Extract features for single track
            feature_box = np.array([target_track["bbox"]], dtype=np.float32)
            try:
                embedding = self.tracker.model.get_features(feature_box, frame)
            except Exception as e:
                logger.error(f"Feature extraction error: {e}")
                return {}, encode_frame(frame)
                
            # Process single track with database
            with Session() as session:
                try:
                    if embedding is not None and embedding.shape[1] == 512:
                        embedding = embedding[0]  # Get first (only) embedding
                        embedding = embedding.astype(np.float32)
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                            
                            # Database operations similar to _process_tracks but for single track
                            session.begin_nested()
                            
                            # Search for matching embedding
                            matches = self.milvus_collection.search(
                                data=[embedding.tolist()],
                                anns_field="embedding",
                                param={"metric_type": "L2", "params": {"nprobe": 10}},
                                limit=1
                            )

                            if matches and len(matches[0]) > 0 and matches[0][0].distance < self.similarity_threshold:
                                person_id = matches[0][0].id
                                person = session.query(Person).filter_by(person_id=person_id).first()
                                if not person:
                                    person = Person()
                                    session.add(person)
                                    session.flush()
                                    person_id = person.person_id
                            else:
                                person = Person()
                                session.add(person)
                                session.flush()
                                person_id = person.person_id
                                
                                # Store new embedding
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
                            session.commit()
                            
                            result = {
                                "person_id": person_id,
                                "track_id": target_track_id,
                                "bbox": target_track["bbox"],
                                "confidence": target_track["confidence"]
                            }
                            
                            # Draw result on frame
                            result_frame = self._draw_results(frame, [result])
                            return result, encode_frame(result_frame)
                            
                except Exception as e:
                    logger.error(f"Database operation error: {str(e)}")
                    session.rollback()
                    
            return {}, encode_frame(frame)
            
        except Exception as e:
            logger.error(f"Single track processing error: {e}")
            return {}, encode_frame(frame)

            
        