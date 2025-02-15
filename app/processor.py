from datetime import datetime
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from .database import Session
from .models import Person, Detection
from .utils import extract_features, encode_frame
from .ml_models import DeepSort

logger = logging.getLogger(__name__)

class PersonProcessor:
    def __init__(self, face_analyzer, head_detector, deep_sort, osnet, similarity_threshold=128.0):
        self.face_analyzer = face_analyzer
        self.head_detector = head_detector
        self.deep_sort:DeepSort = deep_sort
        self.osnet = osnet
        self.similarity_threshold = similarity_threshold
        self._setup_milvus_collection()

    def _setup_milvus_collection(self):
        """Create Milvus collection if it doesn't exist"""
        try:
            self.milvus_collection = Collection("embeddings")
        except Exception as e:
            # Define the collection schema
            embedding_dim = 512  # Adjust this based on your embedding size
            fields = [
                FieldSchema(name="embedding_id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
            ]
            schema = CollectionSchema(fields=fields, description="Person embeddings collection")
            
            # Create collection
            self.milvus_collection = Collection(name="embeddings", schema=schema)
            
            # Create index for vector field
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.milvus_collection.create_index(field_name="embedding", index_params=index_params)
            self.milvus_collection.load()
            logger.info("Created new Milvus collection 'embeddings'")

    def process_frame(self, frame: np.ndarray, camera_id: str, timestamp: datetime) -> Tuple[List[Dict], str]:
        try:
            # 1. Detect and get embeddings
            detections, embeddings = self._get_detections(frame)
            if not detections:
                return [], encode_frame(frame)

            # 2. Track objects using DeepSORT
            tracked_objects = self._track_objects(frame, detections, embeddings)
            print(tracked_objects)

            # 3. Process and store results
            results = self._process_tracks(tracked_objects, camera_id, timestamp, frame)
            print(results)

            # 4. Visualize results
            result_frame = self._draw_results(frame, results)

            return results, encode_frame(result_frame)

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return [], encode_frame(frame)

    def _get_detections(self, frame):
        detections, embeddings = [], []
        
        # Try faces first
        faces = self.face_analyzer.get(frame)
        for face in faces:
            if face.embedding is not None:
                detections.append(face.bbox.astype(int))
                embeddings.append(face.embedding)

        # If no faces, try heads
        if not detections:
            heads = self.head_detector(frame, classes=[0])
            for head in heads:
                for box in head.boxes.data:
                    bbox = box[:4].cpu().numpy().astype(int)
                    roi = self._get_safe_roi(frame, bbox)
                    if roi.size > 0:
                        embedding = extract_features(roi, self.osnet)
                        if embedding.size > 0:
                            detections.append(bbox)
                            embeddings.append(embedding)

        print(detections)
        return detections, embeddings

    def _track_objects(self, frame, detections, embeddings):
        if not detections:
            return []

        detections_array = np.array([
            [x1 + (x2-x1)/2, y1 + (y2-y1)/2, x2-x1, y2-y1]
            for x1, y1, x2, y2 in detections
        ], dtype=np.float32)
        
        scores = np.ones(len(detections))
        classes = np.zeros(len(detections))

        outputs, _ = self.deep_sort.update(detections_array, scores, classes, frame)
        return outputs if outputs is not None else []

    def _process_tracks(self, tracked_objects, camera_id, timestamp, frame):
        results = []
        with Session() as session:
            for track in tracked_objects:
                try:
                    # Convert array format to bbox coordinates
                    x1, y1, x2, y2, class_id, track_id = track
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    
                    # Get ROI and extract embedding
                    roi = self._get_safe_roi(frame, bbox)
                    if roi.size > 0:
                        embedding = extract_features(roi, self.osnet)
                        if embedding.size == 0:
                            continue
                    else:
                        continue

                    # Search for matching person
                    matches = self.milvus_collection.search(
                        data=[embedding],
                        anns_field="embedding",
                        param={"metric_type": "L2", "params": {"nprobe": 10}},
                        limit=1
                    )
                    print(matches)
                    if matches and len(matches[0]) > 0 and matches[0][0].distance < self.similarity_threshold:
                        person_id = matches[0][0].id
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

                    # Log detection
                    detection = Detection(
                        person_id=person_id,
                        camera_id=camera_id,
                        embedding_id=str(person_id)
                    )
                    session.add(detection)
                    
                    results.append({
                        "person_id": person_id,
                        "bbox": bbox,
                        "track_id": int(track_id)
                    })

                    session.commit()

                except Exception as e:
                    logger.error(f"Track processing error: {e}")
                    session.rollback()

        return results

    def _draw_results(self, frame, results):
        result_frame = frame.copy()
        for det in results:
            x1, y1, x2, y2 = det["bbox"]
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
        return result_frame

    @staticmethod
    def _get_safe_roi(frame, bbox):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]