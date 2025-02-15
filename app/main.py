# main.py
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import cv2
from datetime import datetime

# app/main.py
from pymilvus import connections
from .config import CAMERA_MAPPING
from .ml_models import init_models, release_models
from .processor import PersonProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


analyser: PersonProcessor = None
# Define a lifespan manager using @asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup logic
    logger.info("Starting up...")
    connections.connect("default", host='localhost', port='19530')

    face_analyzer, head_detector, deep_sort, osnet = init_models()
    global analyser
    analyser = PersonProcessor(face_analyzer, head_detector, deep_sort, osnet)
    logger.info("Models initialized successfully.")

    yield  # This is where the application runs

    # Shutdown logic
    logger.info("Shutting down...")
    connections.disconnect("default")
    release_models(face_analyzer, head_detector, deep_sort, osnet)
    logger.info("Models released successfully.")

# Pass the lifespan manager to the FastAPI app
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static", html=True), name="static")

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

# uses websocket to stream video feed
@app.websocket("/ws/video_feed/{camera_id}")
async def video_feed(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    # reads output from camera mounted to server
    cap = cv2.VideoCapture(CAMERA_MAPPING.get(camera_id))
    # cap = cv2.VideoCapture("app/static/sample.mp4")
    
    if not cap.isOpened():
        await websocket.close(code=1008, reason="Camera initialization failed")
        return
        
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = datetime.now()
            detections, result_image = analyser.process_frame(frame, camera_id, timestamp)
            # detections has uuid convert to string
            for detection in detections:
                detection["person_id"] = str(detection["person_id"])
            if result_image:
                await websocket.send_json({
                    "detections": detections,
                    "timestamp": timestamp.isoformat(),
                    "result_image": result_image
                })
            await asyncio.sleep(0.033)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        cap.release()
        await websocket.close()


# @app.post("/search")
# async def search_person(image: UploadFile = File(...)):
#     # 1. Process query image
#     frame = cv2.imdecode(np.fromstring(await image.read(), np.uint8), cv2.IMREAD_COLOR)
#     emb = ArcFace.extract(frame)[0]
    
#     # 2. Search Milvus
#     matches = milvus.search(
#         collection_name="people",
#         vector=emb,
#         limit=5
#     )
    
#     # 3. Get last-known locations
#     results = []
#     for match in matches:
#         person_id = match.id
#         last_seen = db.execute("""
#             SELECT camera_id, timestamp 
#             FROM detections 
#             WHERE person_id = %s 
#             ORDER BY timestamp DESC 
#             LIMIT 1
#         """, (person_id,)).fetchone()
        
#         results.append({
#             "person_id": person_id,
#             "last_seen": last_seen
#         })
    
#     return results