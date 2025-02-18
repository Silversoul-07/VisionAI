from dataclasses import dataclass
from datetime import datetime
import random
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from sqlalchemy.orm import Session as SQLSession
from .database import Session
from .models import Person, Detection
from .utils import encode_frame
from ultralytics import YOLO
from boxmot import BotSort

logger = logging.getLogger(__name__)

@dataclass
class Track:
    track_id: int
    bbox: List[int]
    confidence: float
    class_id: int

@dataclass
class ProcessedTrack:
    person_id: int
    track_id: int
    bbox: List[int]
    confidence: float

class PersonProcessor:
    EMBEDDING_DIM = 512
    MIN_BBOX_SIZE = 30
    PERSON_CLASS_ID = 0

    def __init__(self, yolo: YOLO, tracker: BotSort, similarity_threshold: float = 0.25):
        self.yolo = yolo
        self.tracker = tracker
        self.similarity_threshold = similarity_threshold
        self.milvus_collection = self._setup_milvus_collection()

    def _setup_milvus_collection(self) -> Collection:
        try:
            return Collection("embeddings")
        except Exception:
            fields = [
                FieldSchema(name="embedding_id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.EMBEDDING_DIM)
            ]
            schema = CollectionSchema(fields=fields, description="Person embeddings collection")
            collection = Collection(name="embeddings", schema=schema)
            collection.create_index(
                field_name="embedding",
                index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
            )
            collection.load()
            return collection

    def _normalize_embedding(self, embedding: np.ndarray) -> Optional[np.ndarray]:
        if embedding.shape[0] != self.EMBEDDING_DIM:
            logger.warning(f"Wrong embedding dimension: {embedding.shape[0]}")
            return None
            
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else None

    def _create_or_get_person(self, session: SQLSession, embedding: np.ndarray) -> Optional[int]:
        try:
            matches = self.milvus_collection.search(
                data=[embedding.tolist()],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=1
            )

            if matches and matches[0] and matches[0][0].distance < self.similarity_threshold:
                person_id = matches[0][0].id
                person = session.query(Person).filter_by(person_id=person_id).first()
                if person:
                    return person_id

            person = Person()
            session.add(person)
            session.flush()
            
            self.milvus_collection.insert([{
                "embedding_id": str(person.person_id),
                "embedding": embedding.tolist()
            }])
            
            return person.person_id
            
        except Exception as e:
            logger.error(f"Person creation error: {e}")
            return None

    def process_frame(self, frame: np.ndarray) -> List[Track]:
        try:
            results = self.yolo(frame)[0]
            if not results.boxes:
                return []
    
            # Cast to float32 for better performance
            dets = results.boxes.data.cpu().numpy().astype(np.float32)
            person_dets = dets[dets[:, 5] == self.PERSON_CLASS_ID]
            if not len(person_dets):
                return []
    
            # Ensure proper type casting for tracking input
            tracking_dets = np.column_stack((
                person_dets[:, 0:4],  # bbox coordinates
                person_dets[:, 4],    # confidence scores
                np.zeros(len(person_dets), dtype=np.float32)  # class IDs
            )).astype(np.float32)
    
            tracked_objects = self.tracker.update(tracking_dets, frame)
            
            # Cast tracking results to appropriate types
            if len(tracked_objects) > 0:
                tracked_objects = np.array(tracked_objects, dtype=np.float32)
                
            return [
                Track(
                    track_id=int(track[4]),      # track ID should be integer
                    bbox=[int(x) for x in track[0:4]],  # bbox should be integer
                    confidence=float(track[5]),   # confidence should be float
                    class_id=int(track[6])       # class ID should be integer
                )
                for track in tracked_objects
                if track[5] >= self.similarity_threshold
            ]
    
        except Exception as e:
            logger.error(f"Frame processing error: {e}", exc_info=True)
            return []

    def _is_valid_bbox(self, bbox: List[int]) -> bool:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width >= self.MIN_BBOX_SIZE and height >= self.MIN_BBOX_SIZE

    def track_person(
        self,
        frame: np.ndarray,
        target_track_id: int,
        camera_id: str,
        timestamp: datetime
    ) -> Tuple[Dict, str]:
        try:
            tracked_results = self.process_frame(frame)
            target_track = next(
                (track for track in tracked_results if track.track_id == target_track_id),
                None
            )
            
            if not target_track or not self._is_valid_bbox(target_track.bbox):
                return {}, encode_frame(frame)

            feature_box = np.array([target_track.bbox], dtype=np.float32)
            embedding = self.tracker.model.get_features(feature_box, frame)
            
            if embedding is None or embedding.shape[1] != self.EMBEDDING_DIM:
                return {}, encode_frame(frame)

            normalized_embedding = self._normalize_embedding(embedding[0])
            if normalized_embedding is None:
                return {}, encode_frame(frame)

            with Session() as session:
                session.begin_nested()
                try:
                    person_id = self._create_or_get_person(session, normalized_embedding)
                    if not person_id:
                        return {}, encode_frame(frame)

                    detection = Detection(
                        person_id=person_id,
                        camera_id=camera_id,
                        embedding_id=str(person_id),
                        timestamp=timestamp
                    )
                    session.add(detection)
                    session.commit()

                    result = ProcessedTrack(
                        person_id=person_id,
                        track_id=target_track.track_id,
                        bbox=target_track.bbox,
                        confidence=target_track.confidence
                    )
                    result_dict = result.__dict__

                    result_frame = self._draw_results(frame, [result_dict])
                    return result.__dict__, encode_frame(result_frame)

                except Exception as e:
                    logger.error(f"Database operation error: {e}")
                    session.rollback()
                    return {}, encode_frame(frame)

        except Exception as e:
            logger.error(f"Track processing error: {e}")
            return {}, encode_frame(frame)
        
    def track_all(self, frame: np.ndarray, camera_id: str, timestamp: datetime) -> Tuple[List[Dict], str]:
        tracked_results = self.process_frame(frame)
        logger.info(f"Number of initial detections: {len(tracked_results)}")  # Add this
        processed_tracks = []
        
        if not tracked_results:
            return [], encode_frame(frame)
        
        with Session() as session:
            session.begin_nested()
            for track in tracked_results:
                if not self._is_valid_bbox(track.bbox):
                    logger.debug(f"Invalid bbox: {track.bbox}")  # Add this
                    continue

                feature_box = np.array([track.bbox], dtype=np.float32)
                embedding = self.tracker.model.get_features(feature_box, frame)
                if embedding is None or embedding.shape[1] != self.EMBEDDING_DIM:
                    continue

                normalized_embedding = self._normalize_embedding(embedding[0])
                if normalized_embedding is None:
                    continue

                person_id = self._create_or_get_person(session, normalized_embedding)
                if not person_id:
                    continue

                detection = Detection(
                    person_id=person_id,
                    camera_id=camera_id,
                    embedding_id=str(person_id),
                    timestamp=timestamp
                )
                session.add(detection)

                processed_track = ProcessedTrack(
                    person_id=person_id,
                    track_id=track.track_id,
                    bbox=track.bbox,
                    confidence=track.confidence
                )
                processed_tracks.append(processed_track.__dict__)
            
            try:
                session.commit()
            except Exception as e:
                logger.error(f"Database operation error (video): {e}")
                session.rollback()
        logger.info(f"Final processed tracks: {len(processed_tracks)}")  # Add this
        result_frame = self._draw_results(frame, processed_tracks)
        return processed_tracks, encode_frame(result_frame)
    

        
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