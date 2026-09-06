import os

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# File Upload Configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit

# SQLite Database Configuration
DB_PATH = os.path.join(BASE_DIR, 'detections.db')

# YOLOv8 Model Configuration
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8n.pt')
DEFAULT_CONFIDENCE = 0.5  # 50% default confidence threshold
