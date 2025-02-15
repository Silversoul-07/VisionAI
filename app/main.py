# main.py
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import cv2
from datetime import datetime

from scipy.datasets import face
from .config import CAMERA_MAPPING
from .ml_models import init_models, release_models
from .processor import process_frame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

face_analyzer, head_detector, deep_sort, osnet = None, None, None, None

# Define a lifespan manager using @asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup logic
    logger.info("Starting up...")

    global face_analyzer, head_detector, deep_sort, osnet
    face_analyzer, head_detector, deep_sort, osnet = init_models()
    logger.info("Models initialized successfully.")

    yield  # This is where the application runs

    # Shutdown logic
    logger.info("Shutting down...")
    release_models(face_analyzer, head_detector, deep_sort, osnet)
    logger.info("Models released successfully.")

# Pass the lifespan manager to the FastAPI app
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static", html=True), name="static")

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

@app.websocket("/ws/video_feed/{camera_id}")
async def video_feed(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    # cap = cv2.VideoCapture(CAMERA_MAPPING.get(camera_id))
    cap = cv2.VideoCapture("app/static/sample.mp4")
    
    if not cap.isOpened():
        await websocket.close(code=1008, reason="Camera initialization failed")
        return
        
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = datetime.now()
            detections, result_image = process_frame(
                frame,
                camera_id,
                timestamp,
                face_analyzer,
                head_detector,
                deep_sort,
                osnet
            )
            
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