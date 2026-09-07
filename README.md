# 🎯 Real-Time Object Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![SQLite](https://img.shields.io/badge/SQLite-3-blue?style=flat-square&logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A production-grade, full-stack computer vision web application built with **Python**, **YOLOv8 (Ultralytics)**, **Flask**, and **SQLite**. Upload any image and instantly detect 80+ everyday objects — phones, chairs, TVs, people, cars, and more — with confidence scores, bounding box annotations, detection history logging, and a real-time analytics dashboard.

---

## 📸 Screenshots

### Upload & Detection
> *(Add screenshot of index.html with confidence slider and upload zone)*

### Results Page
> *(Add screenshot of result.html showing side-by-side original vs annotated)*

### Analytics Dashboard
> *(Add screenshot of analytics.html showing charts and history table)*

---

## ✨ Features

- 🔍 **Multi-Class Object Detection** — Detects 80+ object categories using YOLOv8n (MS COCO dataset)
- 🎚️ **Confidence Threshold Slider** — Adjust detection sensitivity from 10% to 90% in real time
- 📊 **Analytics Dashboard** — Chart.js visualizations: class distribution bar chart + processing time trend
- 🗄️ **SQLite History Logging** — Every detection run is stored with class names, confidence scores, bounding boxes, and latency
- 🔁 **Full CRUD REST API** — Create, read, and delete detection records via RESTful endpoints
- ✅ **Input Validation** — Strict file type, size, and integrity checks with proper HTTP status codes
- 🧪 **Automated Testing** — `pytest` test suite covering detector logic and all API endpoints
- 🎨 **Modern Responsive UI** — Glassmorphic design, drag-and-drop upload, mobile-friendly layout
- ⚡ **Performance Metrics** — Processing time (ms) tracked per inference run

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Object Detection | YOLOv8n (Ultralytics) |
| Computer Vision | OpenCV |
| Backend Framework | Flask |
| Database | SQLite |
| Frontend | HTML5, CSS3, JavaScript |
| Charts | Chart.js |
| Testing | pytest |
| Version Control | Git |

---

## 📁 Project Structure

```
AI_Object_Detection_System/
├── app.py                     # Flask application & REST API routes
├── detector.py                # OOP ObjectDetector class using YOLOv8
├── database.py                # SQLite module with full CRUD & analytics
├── config.py                  # Centralized configuration constants
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development & testing dependencies
├── .gitignore
├── models/
│   └── yolov8n.pt             # Auto-downloaded on first run (6MB)
├── static/
│   ├── css/style.css
│   ├── js/
│   │   ├── main.js            # Upload handling, fetch API, confidence slider
│   │   └── analytics.js       # Chart.js visualizations & CRUD interactions
│   └── uploads/               # Processed & uploaded image storage
├── templates/
│   ├── index.html             # Upload panel + confidence slider
│   ├── result.html            # Side-by-side comparison & detections table
│   └── analytics.html         # KPI cards, charts, history table
├── tests/
│   ├── __init__.py
│   ├── test_detector.py       # Unit tests for ObjectDetector class
│   └── test_api.py            # API endpoint integration tests
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip
- Git

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/harshithcheripally16-ui/AI-Object-Detection-System.git
cd AI-Object-Detection-System
```

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install production dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the application**
```bash
python app.py
```

**5. Open in browser**
```
http://127.0.0.1:5000
```

> **Note:** On first run, `yolov8n.pt` (~6MB) will auto-download from Ultralytics. An active internet connection is required for the first run only.

---

## 🧪 Running Tests

**Install development dependencies**
```bash
pip install -r requirements-dev.txt
```

**Run the full test suite**
```bash
pytest -v tests/
```

**Expected output:**
```
tests/test_detector.py::test_detector_loads_model         PASSED
tests/test_detector.py::test_detect_returns_results       PASSED
tests/test_detector.py::test_detect_blank_image           PASSED
tests/test_detector.py::test_get_stats_structure          PASSED
tests/test_detector.py::test_confidence_threshold         PASSED
tests/test_detector.py::test_draw_boxes_returns_array     PASSED
tests/test_api.py::test_index_returns_200                 PASSED
tests/test_api.py::test_detect_valid_image_returns_200    PASSED
tests/test_api.py::test_detect_no_file_returns_400        PASSED
tests/test_api.py::test_detect_invalid_extension_returns_415 PASSED
tests/test_api.py::test_detect_response_has_required_keys PASSED
tests/test_api.py::test_analytics_data_returns_json       PASSED
tests/test_api.py::test_delete_history_valid_id_returns_200 PASSED
tests/test_api.py::test_delete_history_invalid_id_returns_404 PASSED

14 passed in X.XXs
```

---

## 📡 API Reference

### `GET /`
Returns the main upload and detection interface.

**Response:** `200 OK` — HTML page

---

### `POST /detect`
Upload an image for object detection.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `image` — Image file (JPG, PNG, JPEG)
  - `confidence` — Float (0.1 – 0.9), default `0.5`

**Response — `200 OK`:**
```json
{
  "result_image": "static/uploads/result_abc123.jpg",
  "total_count": 4,
  "processing_time_ms": 187.43,
  "confidence_used": 0.5,
  "detections": [
    { "class": "person", "confidence": 0.91, "bbox": [120, 45, 380, 600] },
    { "class": "chair",  "confidence": 0.78, "bbox": [400, 200, 600, 580] },
    { "class": "tv",     "confidence": 0.85, "bbox": [10, 30, 300, 250] },
    { "class": "mouse",  "confidence": 0.72, "bbox": [500, 400, 560, 440] }
  ]
}
```

**Error Responses:**

| Status Code | Reason |
|---|---|
| `400 Bad Request` | Missing file or corrupted image |
| `413 Payload Too Large` | File exceeds 16MB limit |
| `415 Unsupported Media Type` | Invalid file extension |
| `500 Internal Server Error` | Unexpected inference error |

---

### `GET /result/<filename>`
Returns the result detail page for a specific detection run.

**Response:** `200 OK` — HTML page

---

### `GET /analytics`
Returns the analytics dashboard page.

**Response:** `200 OK` — HTML page

---

### `GET /api/analytics-data`
Returns aggregated analytics data as JSON for Chart.js.

**Response — `200 OK`:**
```json
{
  "total_images": 42,
  "total_detections": 198,
  "avg_processing_time_ms": 203.5,
  "top_class": "person",
  "class_distribution": {
    "person": 54,
    "chair": 31,
    "tv": 22,
    "mouse": 18
  },
  "latency_trend": [187, 201, 195, 220, 189]
}
```

---

### `DELETE /api/history/<int:id>`
Delete a specific detection record from history.

**Response — `200 OK`:**
```json
{ "message": "Record 7 deleted successfully." }
```

**Response — `404 Not Found`:**
```json
{ "error": "Record not found." }
```

---

## ⚙️ Configuration

All constants are managed in `config.py`:

```python
UPLOAD_FOLDER      = 'static/uploads'
DB_PATH            = 'detections.db'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024   # 16MB
YOLO_MODEL_PATH    = 'models/yolov8n.pt'
DEFAULT_CONFIDENCE = 0.5
```

---

## 🎯 Detectable Object Classes (80 COCO Categories)

```
person       bicycle      car          motorcycle   airplane
bus          train        truck        boat         traffic light
fire hydrant stop sign    bench        bird         cat
dog          horse        sheep        cow          elephant
bear         zebra        giraffe      backpack     umbrella
handbag      tie          suitcase     frisbee      skis
snowboard    sports ball  kite         baseball bat bottle
wine glass   cup          fork         knife        spoon
bowl         banana       apple        sandwich     orange
broccoli     carrot       pizza        donut        cake
chair        couch        potted plant bed          dining table
toilet       tv           laptop       mouse        remote
keyboard     cell phone   microwave    oven         toaster
sink         refrigerator book         clock        vase
scissors     teddy bear   hair drier   toothbrush   ...and more
```

---

## 📝 Git Commit History

```
feat: initial project setup and config
feat: add OOP ObjectDetector class with YOLOv8 multi-class detection
feat: add SQLite database module with CRUD operations
feat: add Flask REST API endpoints with input validation
feat: add responsive frontend UI with confidence slider and analytics dashboard
feat: add pytest unit tests for detector and API
docs: finalize README with screenshots and API reference
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to the branch (`git push origin feat/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Harshith Cheripally**
- GitHub: [@harshithcheripally16-ui](https://github.com/harshithcheripally16-ui)
- Email: harshithcheripally16@gmail.com

---

> ⭐ If you found this project useful, consider starring the repository!
