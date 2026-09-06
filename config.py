import os

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# File Upload Configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit

# SQLite Database Configuration
DB_PATH = os.path.join(BASE_DIR, 'detections.db')

# Haar Cascade Model Definitions
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODELS = {
    'face': {
        'name': 'Frontal Face Detection',
        'file': os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml'),
        'color': (0, 255, 0),       # Green bounding box (BGR)
        'scale_factor': 1.1,
        'min_neighbors': 5,
        'min_size': (30, 30),
        'label': 'Face'
    },
    'eye': {
        'name': 'Eye Detection',
        'file': os.path.join(MODELS_DIR, 'haarcascade_eye.xml'),
        'color': (255, 100, 0),     # Blue/Cyan in BGR
        'scale_factor': 1.1,
        'min_neighbors': 8,
        'min_size': (15, 15),
        'label': 'Eye'
    },
    'fullbody': {
        'name': 'Full Body Detection',
        'file': os.path.join(MODELS_DIR, 'haarcascade_fullbody.xml'),
        'color': (0, 140, 255),     # Orange in BGR
        'scale_factor': 1.05,
        'min_neighbors': 3,
        'min_size': (50, 100),
        'label': 'Person'
    }
}
DEFAULT_MODEL = 'face'
