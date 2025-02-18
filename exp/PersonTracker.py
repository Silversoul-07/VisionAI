import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as T
from PIL import Image
import timm
from scipy.spatial.distance import cosine
from torch import nn
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema, connections

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super().__init__()
        # Use timm model instead of torchreid
        self.base = timm.create_model(model_name, pretrained=pretrained)
        # Remove the classification head
        self.base.reset_classifier(0)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.base(x)
        return features

class PersonTracker:
    def __init__(self, video_path, yolo_model="yolov8x.pt"):
        self.video_path = video_path
        self.detector = YOLO(yolo_model)
        
        # Initialize feature extractor
        self.embedding_model = FeatureExtractor()
        
        # Initialize tracking variables
        self.person_embeddings = {}
        self.target_id = None
        self.target_embedding = None
        self.similarity_threshold = 0.75  # Added missing threshold
        
        # Enhanced preprocessing
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup_milvus(self):
        """Setup Milvus collection for person embeddings"""
        try:
            connections.connect(host='localhost', port='19530')
            
            dim = 1280  # EfficientNet feature dimension
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
            ]
            schema = CollectionSchema(fields=fields, description="person_embeddings")
            
            # Check if collection exists before creating
            if not Collection.exists("person_embeddings"):
                self.collection = Collection(name="person_embeddings", schema=schema)
                self.collection.create_index(
                    field_name="embedding", 
                    index_params={
                        'metric_type': 'COSINE',
                        'index_type': 'IVF_FLAT',
                        'params': {'nlist': 1024}
                    }
                )
            else:
                self.collection = Collection("person_embeddings")
                self.collection.load()
                
        except Exception as e:
            print(f"Error setting up Milvus: {str(e)}")
            raise

    def extract_embedding(self, frame, bbox):
        """Extract embedding from person crop"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add boundary checks
            height, width = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            person_crop = frame[y1:y2, x1:x2]
            
            # Handle invalid crops
            if person_crop.size == 0:
                return None
                
            # Convert to PIL and apply preprocessing
            person_crop = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            img_tensor = self.transform(person_crop).unsqueeze(0)
            
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
                
            with torch.no_grad():
                embedding = self.embedding_model(img_tensor)
                
            # L2 normalization
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error extracting embedding: {str(e)}")
            return None

    def analyze_frame(self, skip_seconds=0):
        """Analyze frame to detect persons and create embeddings mapping"""
        try:
            # Clear previous mappings
            self.person_embeddings.clear()
            
            # Get frame from video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            skip_frames = int(fps * skip_seconds)
            
            # Skip frames if needed
            for _ in range(skip_frames):
                ret = cap.read()
                if not ret[0]:
                    raise ValueError("Could not skip frames")
                    
            success, frame = cap.read()
            cap.release()
            
            if not success:
                raise ValueError("Could not extract frame")
            
            # Run detection with tracking
            results = self.detector.track(frame, persist=True, classes=[0])
            
            if len(results) == 0 or not results[0].boxes:
                return {'frame': frame, 'detected_persons': []}
            
            # Create visualization frame
            display_frame = frame.copy()
            detected_persons = []
            
            for box in results[0].boxes:
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id.item())
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    # Extract embedding
                    embedding = self.extract_embedding(frame, bbox)
                    
                    if embedding is not None:
                        # Store in mapping
                        self.person_embeddings[track_id] = {
                            'embedding': embedding,
                            'bbox': bbox.tolist()
                        }
                        
                        # Draw bounding box and track ID
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"ID: {track_id}"
                        cv2.putText(display_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        detected_persons.append({
                            'track_id': track_id,
                            'bbox': bbox.tolist()
                        })
            
            return {
                'frame': display_frame,
                'detected_persons': detected_persons
            }
            
        except Exception as e:
            print(f"Error in analyze_frame: {str(e)}")
            return None

    def track_person(self, target_id, display=True):
        """Track specified person through video"""
        try:
            if target_id not in self.person_embeddings:
                raise ValueError(f"Invalid target ID: {target_id}")
            
            self.target_id = target_id
            self.target_embedding = self.person_embeddings[target_id]['embedding']
            
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                # Run detection and tracking
                results = self.detector.track(frame, persist=True, classes=[0])
                
                if len(results) == 0 or not results[0].boxes:
                    continue
                
                best_match = None
                best_similarity = -1
                
                # Find best matching person
                for box in results[0].boxes:
                    if not hasattr(box, 'id') or box.id is None:
                        continue
                        
                    bbox = box.xyxy[0].cpu().numpy()
                    current_embedding = self.extract_embedding(frame, bbox)
                    
                    if current_embedding is not None:
                        similarity = 1 - cosine(self.target_embedding, current_embedding)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = {
                                'bbox': bbox,
                                'similarity': similarity
                            }
                
                # Update and display if good match found
                if best_match and best_match['similarity'] > self.similarity_threshold:
                    if display:
                        x1, y1, x2, y2 = map(int, best_match['bbox'])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = f"ID: {self.target_id} Sim: {best_match['similarity']:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        cv2.imshow('Tracking', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    # Update target embedding with temporal smoothing
                    current_embedding = self.extract_embedding(frame, best_match['bbox'])
                    if current_embedding is not None:
                        self.target_embedding = 0.9 * self.target_embedding + 0.1 * current_embedding
                        self.target_embedding /= np.linalg.norm(self.target_embedding)
            
        except Exception as e:
            print(f"Error in track_person: {str(e)}")
            
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()