# main.py
from contextlib import asynccontextmanager
import random
from typing import AsyncGenerator
from unittest import result
from uuid import UUID
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import cv2
from datetime import datetime

# app/main.py
from pymilvus import connections
from .config import CAMERA_MAPPING
from .ml_models import init_models, release_models
from .processor import PersonProcessor
from .utils import extract_initial_frame
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


processor: PersonProcessor = None
# Define a lifespan manager using @asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup logic
    logger.info("Starting up...")
    connections.connect("default", host='localhost', port='19530')
    extract_initial_frame(
        "app/static/sample.mp4",
        "app/static/sample.jpg"
    )

    detector, tracker = init_models()
    global processor
    processor = PersonProcessor(detector, tracker)
    logger.info("Models initialized successfully.")

    yield  # This is where the application runs

    # Shutdown logic
    logger.info("Shutting down...")
    connections.disconnect("default")
    release_models(detector, tracker)   
    logger.info("Models released successfully.")

# Pass the lifespan manager to the FastAPI app
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static", html=True), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/test")
async def sample_ui():
    return FileResponse("app/static/index.html")

@app.get("/yolo/predict")
async def predict():
    path = "app/static/sample.jpg"
    frame = cv2.imread(path)
    return processor.process_frame(frame)

# uses websocket to stream video feed
@app.websocket("/ws/track/{track_id}")
async def video_feed(websocket: WebSocket, track_id: str):
    await websocket.accept()

    # future implementation
    # reads output from camera mounted to server stimulated by v4l2loopback
    # cap = cv2.VideoCapture(CAMERA_MAPPING.get(camera_id))

    # reads output from sample video
    cap = cv2.VideoCapture("app/static/test.mp4")

    if not cap.isOpened():
        await websocket.close(code=1008, reason="Camera initialization failed")
        return
        
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = datetime.now()
            # detection, result_image = processor.track_person(
            #     frame,
            #     int(track_id),
            #     "camera1",
            #     timestamp
            # )
            detection, result_image = processor.track_all(
                frame,
                "camera1",
                timestamp
            )
            print(len(detection))
            # detections has uuid convert to string
            # if detection and type(detection['person_id']) == UUID:
            #     logger.info("UUID detected")
            #     detection['person_id'] = str(detection['person_id'])

            for i in range(len(detection)):
                if type(detection[i]['person_id']) == UUID:
                    logger.info("UUID detected")
                    detection[i]['person_id'] = str(detection[i]['person_id'])

            if result_image:
                await websocket.send_json({
                    "detections": detection,
                    "timestamp": timestamp.isoformat(),
                    "result_image": result_image
                })
            await asyncio.sleep(0.03)   
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        cap.release()
        await websocket.close()

