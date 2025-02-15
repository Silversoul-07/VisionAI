# config.py
CAMERA_MAPPING = {
    "camera1": "/dev/video10",
    "camera2": "/dev/video11",
    "camera3": "/dev/video12"
}

DB_URL = 'postgresql://user:pass@localhost/tracking_db'
REDIS_HOST = 'localhost'
REDIS_PORT = 6379