import os

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# File Upload Configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit

# SQLite Database Configuration
DB_PATH = os.path.join(BASE_DIR, 'detections.db')

# YOLO Model Configuration
# High-precision YOLO-World v2 with open-vocabulary detection
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8s-worldv2.pt')
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8n.pt')
DEFAULT_CONFIDENCE = 0.40  # 40% default confidence threshold

# Curated High-Frequency Real-World Object & Living Being Vocabulary
# Optimized to detect gaming gear (mice, headphones), hands, electronics, and everyday items
DETECTABLE_CLASSES = [
    # Humans & Body Parts
    'person', 'human', 'hand', 'face', 'head', 'arm',
    
    # Tech, Gaming Gear & Computer Accessories
    'mouse', 'computer mouse', 'gaming mouse', 'mousepad',
    'headphones', 'headset', 'earphones', 'earbuds', 'airpods',
    'cell phone', 'smartphone', 'mobile phone',
    'keyboard', 'laptop', 'computer', 'monitor', 'screen', 'tv',
    'webcam', 'camera', 'microphone', 'speaker', 'bluetooth speaker',
    'charger', 'power bank', 'usb cable', 'wire', 'remote', 'controller', 'gamepad',
    
    # Wearables & Accessories
    'watch', 'smartwatch', 'glasses', 'sunglasses', 'hat', 'cap', 'jacket', 'backpack', 'bag', 'wallet', 'keys',
    
    # Office, Living & Everyday Objects
    'cup', 'coffee mug', 'mug', 'bottle', 'water bottle', 'can', 'glass',
    'chair', 'office chair', 'gaming chair', 'desk', 'table', 'couch', 'bed',
    'pen', 'pencil', 'marker', 'notebook', 'book', 'paper', 'scissors',
    
    # Living Beings & Nature
    'dog', 'cat', 'bird', 'potted plant', 'plant', 'flower',
    
    # Food & Common Essentials
    'apple', 'banana', 'snack', 'food', 'plate', 'bowl', 'fork', 'knife', 'spoon'
]
