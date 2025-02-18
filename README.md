# VisionAI

## **Project Description**
VisionAI is an advanced AI-powered individual tracking system. It utilizes **YOLO11x** to detect all persons in a frame, the **BoT-SORT** algorithm to track individual motion, and **Re-ID using BoT-SORT** for re-identification.

## **Demo Video**
<video width="730" height="auto" controls>
  <source src="https://silversoul-07.github.io/video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## **Recommeded OS**
- Linux

## **Prerequisites**
- CUDA Toolkit
- NVIDIA Drivers
- Docker
- Poetry

## **Setup Instructions**
The `make` command will:
- Start Docker
- Install FastAPI dependencies (for the first-time setup)
- Start Docker Compose
- Launch FastAPI

Once running, the FastAPI service will be available at:
```
http://localhost:8000
```

## **Caution**
⚠️ **Installing all required tools can consume a large amount of data and storage**

## **Cleanup**
To remove installed Docker images, Poetry environment, and dependencies, run:
```sh
make clean
```
