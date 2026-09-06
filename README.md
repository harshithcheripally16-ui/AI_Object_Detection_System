# Real-Time Object Detection & Analytics System (YOLOv8 Multi-Class Edition)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red.svg)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey.svg)](https://sqlite.org/)
[![Pytest](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](https://pytest.org/)

A production-grade, modular computer vision web application and RESTful API built with **Python, YOLOv8 (Ultralytics, `yolov8n.pt`), Flask, SQLite, and modern JavaScript/CSS3**.

The system performs multi-class object detection across **80+ everyday categories** (phone, chair, person, TV, mouse, bottle, dog, car, laptop, etc.), allows dynamic **confidence threshold tuning** via a frontend slider, logs per-detection spatial telemetry into SQLite with full CRUD capabilities, and features an interactive Chart.js analytics dashboard.

---

## 📸 Application Preview & Screenshots

### 1. Main Detection & Upload Interface (with Confidence Slider)
> Drag-and-drop file upload interface with real-time confidence threshold control (10% to 90%), live image preview, and multi-class YOLOv8 bounding box predictions.
![Upload Dashboard](docs/screenshots/upload_dashboard.png)

---

### 2. Live Webcam Stream & Real-Time MJPEG Telemetry
> High-framerate real-time video streaming over `multipart/x-mixed-replace` with on-the-fly YOLOv8 multi-class inference, live FPS overlay, confidence tuning, and simulated fallback feed.
![Live Webcam Stream](docs/screenshots/live_webcam_stream.png)

---

### 3. Side-by-Side Result & Coordinate Telemetry Inspector
> High-resolution side-by-side comparison (Original Source vs Annotated Output), per-object detection table (class label, confidence score, bounding box `[x1, y1, x2, y2]`), and raw JSON export.
![Detection Results](docs/screenshots/detection_results.png)

---

### 4. Analytics & Performance Dashboard
> Real-time system telemetry with KPI summary cards, Top 10 detected object classes bar chart, inference latency & count trends over time, and interactive SQLite history table with CRUD deletion.
![Analytics Dashboard](docs/screenshots/analytics_dashboard.png)

---

## 🌟 Key Features

- **80+ Category Multi-Class Detection**: Powered by Ultralytics YOLOv8 (`yolov8n.pt`) pre-trained on the MS COCO dataset (person, car, dog, bottle, chair, tv, phone, laptop, etc.).
- **Confidence Threshold Control**: Frontend interactive slider allows dynamic confidence filtering from 10% to 90% (default: 50%).
- **High-Performance Latency Telemetry**: Every inference run tracks precision inference execution time in milliseconds (`processing_time_ms`).
- **Strict Input Validation & Security**:
  - File extension verification against whitelist (`PNG`, `JPG`, `JPEG`, `WEBP`)
  - 16MB file payload limit (`MAX_CONTENT_LENGTH`)
  - Image decode integrity checks (`cv2.imread` / OpenCV validation)
  - Standard HTTP status codes (`200`, `400`, `404`, `413`, `415`, `500`)
- **Full CRUD RESTful API**: Complete REST endpoints including `POST /detect`, `GET /api/history`, `GET /api/history/<id>`, and `DELETE /api/history/<id>`.
- **Relational Data Persistence**: SQLite logging with automatic schema creation, indexing, and JSON object telemetry storage.
- **Glassmorphic Responsive UI**: Modern dark theme built with CSS3 variables, drag-and-drop zones, and Chart.js telemetry charts.
- **100% Automated Pytest Coverage**: Comprehensive unit tests for the detection engine and API integration tests.

---

## ⚠️ Notes on Dependencies & Image Formats

> [!NOTE]
> **OpenCV Package Choice (`opencv-python` vs `opencv-python-headless`)**:
> - `opencv-python` is specified in `requirements.txt` for local development and desktop environments.
> - For headless production environments (Docker containers, AWS/GCP servers with no GUI/display server), use `opencv-python-headless` instead.

> [!WARNING]
> **WebP Image Format Compatibility**:
> - WebP (`.webp`) format is supported in validation and upload handling.
> - Please note that detection results may vary depending on lossy compression levels compared to uncompressed JPEG or PNG formats.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning Model** | Ultralytics YOLOv8 Nano (`yolov8n.pt` — 6.2MB) |
| **Backend Framework** | Flask 3.0 (Python) |
| **Computer Vision Engine** | OpenCV (`cv2`) & NumPy |
| **Database** | SQLite 3 (Indexed relational logging) |
| **Frontend UI** | HTML5, Modern CSS3 Glassmorphism, Vanilla JavaScript |
| **Data Visualization** | Chart.js 4.x |
| **Testing Suite** | Pytest, Requests |

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
- `tests/test_detector.py`: Tests YOLOv8 model initialization, inference execution, blank image handling, stats structure (`total` & `objects`), confidence cutoff, and bounding box drawing.
- `tests/test_api.py`: Tests `GET /`, `GET /analytics`, `POST /detect` input validation (empty file, invalid extension, required JSON response keys), and `DELETE /api/history/<id>` CRUD lifecycle.

---

## 📄 License
This project is licensed under the MIT License.
