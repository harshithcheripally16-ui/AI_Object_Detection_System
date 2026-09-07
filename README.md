# Real-Time Object Detection & Analytics System (YOLOv8 & YOLO-World Edition)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red.svg)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey.svg)](https://sqlite.org/)
[![Pytest](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](https://pytest.org/)

A production-grade, modular computer vision web application and RESTful API built with **Python, YOLO-World / YOLOv8 (Ultralytics), Flask, SQLite, and modern JavaScript/CSS3**.

The system performs real-time multi-class object detection across a rich open vocabulary of everyday items, tech accessories (gaming mice, headphones, keyboards, laptops), human body parts (hands, faces, person), and living beings. It features real-time **confidence threshold tuning** (5% to 95%), SQLite logging with full CRUD operations, live MJPEG webcam streaming, and an interactive Chart.js analytics dashboard.

---

## 📸 Real-World Live Detection Screenshots

### 1. Live Webcam Object Detection in Action
> Real-time detection across live camera feeds with dynamic HUD telemetry (**FPS**, **Model**, **Objects count**, **Latency in ms**) and active bounding boxes on everyday items:

| Live Webcam Detection 1 | Live Webcam Detection 2 |
|:---:|:---:|
| ![Live Detection 1](docs/screenshots/live_detection_webcam_1.png) | ![Live Detection 2](docs/screenshots/live_detection_webcam_2.png) |

| Live Webcam Detection 3 | Live Webcam Detection 4 |
|:---:|:---:|
| ![Live Detection 3](docs/screenshots/live_detection_webcam_3.png) | ![Live Detection 4](docs/screenshots/live_detection_webcam_4.png) |

---

## 🖼️ Static Image Detection (Real-World Internet Samples)

> Multi-class inference and bounding-box localization on high-resolution real-world test scenes:

### Gaming Gear & Desk Setup Detection
| Original Real Image | YOLO Annotated Result |
|:---:|:---:|
| ![Gaming Mouse Sample](docs/screenshots/gaming_mouse_sample.jpg) | ![Annotated Gaming Mouse](docs/screenshots/annotated_gaming_mouse_sample.jpg) |

### Modern Workspace & Laptop Detection
| Original Real Image | YOLO Annotated Result |
|:---:|:---:|
| ![Workspace Sample](docs/screenshots/workspace_sample.jpg) | ![Annotated Workspace](docs/screenshots/annotated_workspace_sample.jpg) |

---

## 🖥️ Web Application Dashboard & Telemetry

### Main Upload Interface (with Quick Confidence Presets)
> Drag-and-drop file upload zone with interactive confidence threshold slider (5% to 95%) and one-click preset buttons for high sensitivity (Razer mouse, hands) or strict precision:
![Upload Dashboard](docs/screenshots/upload_dashboard.png)

### Side-by-Side Result Inspector & Spatial Bounding Box Table
> High-resolution side-by-side inspection with coordinate metrics, confidence scores, and raw JSON export:
![Detection Results](docs/screenshots/detection_results.png)

### Real-Time Analytics & Performance Dashboard
> Real-time system telemetry with KPI cards, Top 10 detected object classes bar chart, latency trend line charts, and SQLite history table with CRUD deletion:
![Analytics Dashboard](docs/screenshots/analytics_dashboard.png)

---

## 🌟 Key Features

- **Open-Vocabulary & Multi-Class Architecture**: Powered by Ultralytics YOLO-World / YOLOv8 with preloaded real-world vocabulary covering:
  - 🖱️ **Gaming Gear & Tech**: `gaming mouse`, `computer mouse`, `headphones`, `headset`, `earbuds`, `keyboard`, `laptop`, `monitor`, `webcam`, `microphone`, `cell phone`, `usb cable`, etc.
  - 🖐️ **Humans & Body Parts**: `person`, `hand`, `face`, `head`, `arm`.
  - 👓 **Wearables & Everyday Items**: `watch`, `glasses`, `backpack`, `wallet`, `cup`, `water bottle`, `chair`, `desk`, `pen`, `book`, etc.
  - 🐶 **Living Beings**: `dog`, `cat`, `bird`, `potted plant`, etc.
- **Dynamic Confidence Threshold Tuning**: Interactive slider (5% to 95%) and preset buttons (`25% Sensitive`, `40% Recommended`, `75% High Precision`, `90% Strict`).
- **Real-Time Latency Telemetry**: Sub-millisecond profiling tracking inference latency (`processing_time_ms`) on every request.
- **Strict Input Validation & Security**:
  - Whitelist validation (`PNG`, `JPG`, `JPEG`, `WEBP`)
  - 16MB file payload limit (`MAX_CONTENT_LENGTH`)
  - OpenCV image decode validation (`cv2.imread`)
  - Standard HTTP status codes (`200`, `400`, `404`, `413`, `415`, `500`)
- **Full CRUD RESTful API**: Endpoints including `POST /detect`, `GET /api/history`, `GET /api/history/<id>`, and `DELETE /api/history/<id>`.
- **Relational Persistence**: SQLite storage with automatic schema generation, indexes, and JSON spatial telemetry.
- **100% Automated Pytest Coverage**: Unit tests for detection logic and API integration tests.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning Engine** | Ultralytics YOLO-World / YOLOv8 (`yolov8s-worldv2.pt` / `yolov8n.pt`) |
| **Backend Framework** | Flask 3.0 (Python) |
| **Computer Vision** | OpenCV (`cv2`) & NumPy |
| **Database** | SQLite 3 (Indexed relational logging) |
| **Frontend UI** | HTML5, Modern CSS3 Glassmorphism, Vanilla JavaScript |
| **Data Visualization** | Chart.js 4.x |
| **Testing Suite** | Pytest, Requests |

---

## 📂 Project Architecture & Directory Structure

```
AI_Object_Detection_System/
├── app.py                     # Flask application & REST API routes (GET, POST, DELETE)
├── detector.py                # OOP ObjectDetector class using Ultralytics YOLO
├── database.py                # SQLite module with full CRUD operations & analytics
├── config.py                  # Centralized configuration constants & YOLO vocabulary
├── requirements.txt           # Production dependencies (Flask, Ultralytics, OpenCV, NumPy)
├── requirements-dev.txt       # Development & testing dependencies (Pytest, Requests)
├── .gitignore                 # Excludes caches, temporary databases, and large model weights
├── models/
│   ├── yolov8s-worldv2.pt     # Open-vocabulary model weights
│   └── yolov8n.pt             # Fast nano weights
├── static/
│   ├── css/
│   │   └── style.css          # Glassmorphic UI styling
│   ├── js/
│   │   ├── main.js            # Upload logic, confidence slider, bounding box rendering
│   │   └── analytics.js       # Chart.js visualization & asynchronous CRUD actions
│   └── uploads/               # Saved original, annotated, and live screenshot images
├── templates/
│   ├── index.html             # Upload & real-time detection page with confidence slider
│   ├── live.html              # Real-time webcam streaming interface
│   ├── result.html            # Side-by-side inspection & per-detection table
│   └── analytics.html         # Performance metrics & historical CRUD table
├── tests/
│   ├── __init__.py
│   ├── test_detector.py       # Unit tests for YOLO ObjectDetector OOP engine
│   └── test_api.py            # API endpoint integration and validation tests
├── docs/
│   └── screenshots/           # Application and live detection screenshots
└── README.md                  # Complete project documentation
```

---

## 🚀 Installation & Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/harshithcheripally16-ui/AI_Object_Detection_System.git
cd AI_Object_Detection_System
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Production environment
pip install -r requirements.txt

# Development / Testing environment
pip install -r requirements-dev.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 🧪 Running Automated Tests

Run the complete test suite using `pytest`:

```bash
pytest -v tests/
```

---

## 📡 REST API Reference

### 1. Object Detection Endpoint
`POST /detect`

**Request Headers**: `Content-Type: multipart/form-data`  
**Parameters**:
- `image` (File, Required): Image file (`.png`, `.jpg`, `.jpeg`, `.webp`, Max: 16MB)
- `confidence` (Float, Optional): Confidence threshold from `0.05` to `0.95` (Default: `0.40`)

**Example `200 OK` Response**:
```json
{
  "id": 12,
  "result_image": "/static/uploads/result_abc123.jpg",
  "original_image": "/static/uploads/upload_abc123.jpg",
  "total_count": 3,
  "confidence_used": 0.40,
  "processing_time_ms": 94.5,
  "result_page_url": "/result/12",
  "detections": [
    {"class": "gaming mouse", "confidence": 0.88, "bbox": [140.0, 220.0, 310.0, 390.0]},
    {"class": "headphones", "confidence": 0.92, "bbox": [320.0, 110.0, 520.0, 340.0]},
    {"class": "hand", "confidence": 0.79, "bbox": [180.0, 260.0, 280.0, 370.0]}
  ]
}
```

**HTTP Status Codes**:
- `200 OK`: Detection successful.
- `400 Bad Request`: Missing file, empty filename, or corrupted image.
- `413 Payload Too Large`: Upload exceeds 16MB.
- `415 Unsupported Media Type`: File extension not allowed.
- `500 Internal Server Error`: Unexpected server or model error.

---

### 2. Analytics Telemetry Feed
`GET /api/analytics-data`

**Response**:
```json
{
  "total_images": 45,
  "total_objects": 118,
  "avg_objects_per_image": 2.62,
  "avg_processing_time_ms": 92.4,
  "top_class": "Gaming mouse",
  "class_distribution": [
    {"class": "gaming mouse", "count": 38},
    {"class": "headphones", "count": 31},
    {"class": "hand", "count": 26},
    {"class": "laptop", "count": 19},
    {"class": "person", "count": 14}
  ],
  "timeline": [ ... ]
}
```

---

### 3. Historical Detections API
- `GET /api/history`: List recent detection records.
- `GET /api/history/<id>`: Retrieve specific detection by ID.
- `DELETE /api/history/<id>`: Delete record and associated files.

---

## 📄 License
This project is licensed under the MIT License.
