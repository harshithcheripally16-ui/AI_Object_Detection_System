# Real-Time Object Detection & Analytics System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red.svg)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey.svg)](https://sqlite.org/)
[![Pytest](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](https://pytest.org/)

A production-grade, modular computer vision web application and RESTful API built with **Python, OpenCV, Flask, SQLite, and modern JavaScript/CSS3**.

The system performs multi-model object and feature detection (Frontal Face, Eye, and Full Body), tracks real-time inference latency metrics (`processing_time_ms`), persists spatial bounding box telemetry into SQLite, and provides an interactive analytics dashboard with full CRUD operations.

---

## 📸 Application Preview & Screenshots

### 1. Main Detection & Upload Interface
> Drag-and-drop file upload interface with real-time model selection dropdown (Face, Eye, Full Body) and instant bounding-box telemetry.
![Upload Dashboard](docs/screenshots/upload_dashboard.png)

---

### 2. Side-by-Side Result & Coordinate Telemetry Inspector
> High-resolution side-by-side comparison (Original Source vs Annotated Output), bounding box coordinate metrics table, centroid calculations, and raw JSON export.
![Detection Results](docs/screenshots/detection_results.png)

---

### 3. Analytics & Performance Dashboard
> Real-time system telemetry with KPI summary cards, inference latency trends over time, model distribution charts, and interactive SQLite history table with CRUD deletion.
![Analytics Dashboard](docs/screenshots/analytics_dashboard.png)

---

## 🌟 Key Features

- **Multi-Model Haar Cascade Architecture**: Easily configure and switch between multiple detection models:
  - `haarcascade_frontalface_default.xml` (Face detection)
  - `haarcascade_eye.xml` (Eye detection)
  - `haarcascade_fullbody.xml` (Full body person detection)
- **High-Performance Latency Telemetry**: Every inference run tracks precision latency in milliseconds (`processing_time_ms`), providing benchmarked performance data.
- **Strict Input Validation & Security**:
  - File extension verification against whitelist (`PNG`, `JPG`, `JPEG`, `WEBP`)
  - 16MB file payload limit (`MAX_CONTENT_LENGTH`)
  - Image decode integrity checks (`cv2.imdecode` / OpenCV decode validation)
  - Standard HTTP status codes (`200`, `400`, `404`, `413`, `415`, `500`)
- **Full CRUD RESTful API**: Complete REST endpoints including `POST /detect`, `GET /api/history`, `GET /api/history/<id>`, and `DELETE /api/history/<id>`.
- **Relational Data Persistence**: SQLite logging with automatic schema creation and indexed timestamp queries.
- **Glassmorphic Responsive UI**: Modern dark theme built with CSS3 variables, drag-and-drop zones, and Chart.js telemetry charts.
- **100% Automated Pytest Coverage**: Comprehensive unit tests for the detection engine and API integration tests.

---

## ⚠️ Notes on Dependencies & Image Formats

> [!NOTE]
> **OpenCV Package Choice (`opencv-python` vs `opencv-python-headless`)**:
> - `opencv-python` is specified in `requirements.txt` for local development and desktop environments where graphical display capabilities and standard OpenCV modules are needed.
> - For headless production environments (Docker containers, AWS/GCP servers with no GUI/display server), use `opencv-python-headless` instead to avoid missing X11/GUI library dependencies.

> [!WARNING]
> **WebP Image Format Compatibility**:
> - WebP (`.webp`) format is fully supported in validation and upload handling.
> - Please note that Haar Cascade feature detection algorithms rely heavily on edge gradients and intensity transitions; highly compressed or lossy WebP images may yield slightly variable detection confidence compared to standard uncompressed JPEG or PNG formats.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | Flask 3.0 (Python) |
| **Computer Vision Engine** | OpenCV (`cv2`) & NumPy |
| **Database** | SQLite 3 (Indexed relational logging) |
| **Frontend UI** | HTML5, Modern CSS3 Glassmorphism, Vanilla JavaScript |
| **Data Visualization** | Chart.js 4.x |
| **Testing Suite** | Pytest, Requests |

---

## 📂 Project Architecture & Directory Structure

```
AI_Object_Detection_System/
├── app.py                     # Flask application & REST API routes (GET, POST, DELETE)
├── detector.py                # OOP ObjectDetector class supporting multi-model switching
├── database.py                # SQLite database helper with full CRUD operations & analytics
├── config.py                  # Centralized configuration constants & model parameters
├── requirements.txt           # Production dependencies (Flask, OpenCV, NumPy)
├── requirements-dev.txt       # Development & testing dependencies (Pytest, Requests)
├── .gitignore                 # Excludes static/uploads, database files, and caches
├── models/
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_eye.xml
│   └── haarcascade_fullbody.xml
├── static/
│   ├── css/
│   │   └── style.css          # Glassmorphism UI styling
│   ├── js/
│   │   ├── main.js            # Upload logic, AJAX detection, bounding box rendering
│   │   └── analytics.js       # Chart.js visualization & asynchronous CRUD actions
│   └── uploads/               # Saved original and annotated result images
├── templates/
│   ├── index.html             # Upload & real-time detection page
│   ├── result.html            # Side-by-side inspection & coordinates inspector
│   └── analytics.html         # Performance metrics & historical CRUD table
├── tests/
│   ├── __init__.py
│   ├── test_detector.py       # Unit tests for ObjectDetector OOP engine
│   └── test_api.py            # API endpoint integration and validation tests
├── docs/
│   └── screenshots/           # Application screenshots for documentation
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

### Test Suite Summary:
- `tests/test_detector.py`: Tests model loading, multiscale inference across face/eye/fullbody, box annotations, and coordinate extraction.
- `tests/test_api.py`: Tests `GET /`, `GET /analytics`, `POST /detect` input validation (extension, size, empty file, invalid model), and `DELETE /api/history/<id>` CRUD lifecycle.

---

## 📡 REST API Reference

### 1. Object Detection Endpoint
`POST /detect`

**Request Headers**: `Content-Type: multipart/form-data`  
**Parameters**:
- `image` (File, Required): Image file (`.png`, `.jpg`, `.jpeg`, `.webp`, Max: 16MB)
- `model` (String, Optional): One of `'face'`, `'eye'`, `'fullbody'` (Default: `'face'`)

**Example `200 OK` Response**:
```json
{
  "id": 12,
  "count": 2,
  "model_used": "face",
  "model_name": "Frontal Face Detection",
  "processing_time_ms": 68.45,
  "original_image": "/static/uploads/upload_3f8b1a.jpg",
  "result_image": "/static/uploads/result_3f8b1a.jpg",
  "result_page_url": "/result/12",
  "stats": {
    "total_detections": 2,
    "coordinates": [
      {"x": 120, "y": 140, "w": 85, "h": 85},
      {"x": 310, "y": 155, "w": 90, "h": 90}
    ],
    "centroids": [
      {"cx": 162, "cy": 182},
      {"cx": 355, "cy": 200}
    ],
    "image_dimensions": {"width": 640, "height": 480},
    "coverage_percentage": 5.01
  }
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
  "total_detections": 118,
  "avg_detections": 2.62,
  "avg_latency_ms": 74.31,
  "models_breakdown": [
    {"model": "face", "runs": 30, "detections": 82},
    {"model": "eye", "runs": 10, "detections": 28},
    {"model": "fullbody", "runs": 5, "detections": 8}
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
