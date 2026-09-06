import os
import uuid
import time
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, abort, Response
from werkzeug.utils import secure_filename

from config import (
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    YOLO_MODEL_PATH,
    DEFAULT_CONFIDENCE
)
from detector import ObjectDetector
from database import (
    init_db,
    log_detection,
    get_all_detections,
    get_detection_by_id,
    delete_detection,
    get_analytics_summary
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize detector and database
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
init_db()
detector = ObjectDetector(YOLO_MODEL_PATH)

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- UI Web Routes -----------------

@app.route('/')
def index():
    """Renders the main multi-class detection upload dashboard."""
    return render_template('index.html', default_confidence=DEFAULT_CONFIDENCE)

@app.route('/live')
def live_view():
    """Renders real-time live webcam YOLOv8 object detection dashboard."""
    return render_template('live.html', default_confidence=DEFAULT_CONFIDENCE)

@app.route('/result/<int:record_id>')
def result_view(record_id):
    """Renders detailed side-by-side result inspector for a specific YOLOv8 run."""
    record = get_detection_by_id(record_id)
    if not record:
        return render_template('result.html', error="Detection record not found."), 404
    return render_template('result.html', record=record)

@app.route('/analytics')
def analytics_view():
    """Renders analytics dashboard page with class distribution & latency charts."""
    summary = get_analytics_summary()
    history = get_all_detections(limit=50)
    return render_template('analytics.html', summary=summary, history=history)

# ----------------- MJPEG Live Video Feed -----------------

def generate_mjpeg_frames(confidence=DEFAULT_CONFIDENCE):
    """
    Generator yielding multipart MJPEG frames with real-time YOLOv8 multi-class inference.
    Includes simulated fallback feed when no physical camera is accessible.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    camera_available = cap.isOpened()
    prev_frame_time = time.perf_counter()
    sim_tick = 0

    try:
        while True:
            if camera_available:
                success, frame = cap.read()
                if not success:
                    camera_available = False
                    continue
            else:
                sim_tick += 1
                frame = np.full((480, 640, 3), 20, dtype=np.uint8)
                # Futuristic tech background
                for x in range(0, 640, 40):
                    cv2.line(frame, (x, 0), (x, 480), (30, 30, 38), 1)
                for y in range(0, 480, 40):
                    cv2.line(frame, (0, y), (640, y), (30, 30, 38), 1)

                # Simulated person + laptop object
                center_x = int(320 + 80 * np.sin(sim_tick * 0.04))
                center_y = int(240 + 40 * np.cos(sim_tick * 0.04))

                # Person
                cv2.ellipse(frame, (center_x, center_y + 150), (120, 80), 0, 0, 360, (50, 60, 80), -1)
                cv2.ellipse(frame, (center_x, center_y), (70, 90), 0, 0, 360, (185, 205, 230), -1)
                cv2.circle(frame, (center_x - 25, center_y - 15), 10, (30, 35, 45), -1)
                cv2.circle(frame, (center_x + 25, center_y - 15), 10, (30, 35, 45), -1)

                # Laptop on desk
                lap_x, lap_y = center_x + 80, center_y + 90
                cv2.rectangle(frame, (lap_x - 50, lap_y - 35), (lap_x + 50, lap_y + 15), (70, 80, 95), -1)
                cv2.rectangle(frame, (lap_x - 60, lap_y + 15), (lap_x + 60, lap_y + 25), (120, 130, 145), -1)

            # Run YOLOv8 inference
            results, latency_ms = detector.detect(frame, confidence=confidence)
            annotated_frame = detector.draw_boxes(results)
            stats = detector.get_stats(results)

            # Calculate FPS
            curr_frame_time = time.perf_counter()
            fps = 1.0 / (curr_frame_time - prev_frame_time) if (curr_frame_time - prev_frame_time) > 0 else 30.0
            prev_frame_time = curr_frame_time

            # Top HUD bar
            hud_bg = (15, 23, 42)
            cv2.rectangle(annotated_frame, (0, 0), (640, 38), hud_bg, -1)
            cv2.line(annotated_frame, (0, 38), (640, 38), (56, 189, 248), 1)

            hud_text = f"YOLOv8n | FPS: {fps:.1f} | Conf: {int(confidence * 100)}% | Objects: {stats['total']} | {latency_ms:.1f}ms"
            cv2.putText(
                annotated_frame,
                hud_text,
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (248, 250, 252),
                1,
                cv2.LINE_AA
            )

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.03)

    finally:
        if cap and cap.isOpened():
            cap.release()

@app.route('/video-feed')
def video_feed():
    """
    GET /video-feed?conf=0.5
    Streams live multipart MJPEG video with real-time YOLOv8 object detection.
    """
    try:
        conf = float(request.args.get('conf', DEFAULT_CONFIDENCE))
        conf = max(0.1, min(0.95, conf))
    except (ValueError, TypeError):
        conf = DEFAULT_CONFIDENCE

    return Response(
        generate_mjpeg_frames(confidence=conf),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ----------------- REST API Endpoints -----------------

@app.route('/detect', methods=['POST'])
def detect():
    """
    POST /detect
    Accepts multipart/form-data:
      - 'image': file upload (PNG, JPG, JPEG, WEBP)
      - 'confidence': (optional) float between 0.1 and 0.9 (default: 0.5)
      
    Returns JSON detection stats, latency, and annotated image path.
    """
    # 1. Validate file presence
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided in request."}), 400

    file = request.files['image']
    if not file or file.filename.strip() == '':
        return jsonify({"error": "Empty filename or no file selected."}), 400

    # 2. Validate file extension
    if not is_allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file type. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 415

    # 3. Parse and validate confidence threshold
    try:
        conf_param = request.form.get('confidence', DEFAULT_CONFIDENCE)
        confidence = float(conf_param)
        confidence = max(0.05, min(0.95, confidence))
    except (ValueError, TypeError):
        confidence = DEFAULT_CONFIDENCE

    try:
        # 4. Generate unique filenames & save original
        original_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_token = uuid.uuid4().hex[:10]
        orig_filename = f"upload_{unique_token}.{original_ext}"
        result_filename = f"result_{unique_token}.{original_ext}"

        orig_filepath = os.path.join(app.config['UPLOAD_FOLDER'], orig_filename)
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

        file.save(orig_filepath)

        # 5. Read and validate image decode with OpenCV
        img = cv2.imread(orig_filepath)
        if img is None:
            if os.path.exists(orig_filepath):
                os.remove(orig_filepath)
            return jsonify({"error": "Corrupted or unreadable image file."}), 400

        # 6. Execute YOLOv8 inference & latency tracking
        results, processing_time_ms = detector.detect(img, confidence=confidence)
        annotated_img = detector.draw_boxes(results)
        stats = detector.get_stats(results)

        # 7. Save annotated result image
        cv2.imwrite(result_filepath, annotated_img)

        # 8. Log run in SQLite
        record_id = log_detection(
            filename=orig_filename,
            result_filename=result_filename,
            confidence_threshold=confidence,
            total_count=stats["total"],
            processing_time_ms=processing_time_ms,
            detections_data=stats["objects"]
        )

        # 9. Return structured response
        return jsonify({
            "id": record_id,
            "total_count": stats["total"],
            "count": stats["total"],
            "confidence_used": confidence,
            "processing_time_ms": processing_time_ms,
            "original_image": f"/static/uploads/{orig_filename}",
            "result_image": f"/static/uploads/{result_filename}",
            "result_page_url": f"/result/{record_id}",
            "detections": stats["objects"],
            "stats": stats
        }), 200

    except Exception as e:
        return jsonify({"error": f"Internal inference error: {str(e)}"}), 500

@app.route('/api/history', methods=['GET'])
def api_history():
    """Returns historical YOLOv8 detection runs as JSON."""
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    history = get_all_detections(limit=limit, offset=offset)
    return jsonify(history), 200

@app.route('/api/history/<int:record_id>', methods=['GET'])
def api_get_record(record_id):
    """Returns single detection record by ID."""
    record = get_detection_by_id(record_id)
    if not record:
        return jsonify({"error": f"Record #{record_id} not found."}), 404
    return jsonify(record), 200

@app.route('/api/history/<int:record_id>', methods=['DELETE'])
def api_delete_record(record_id):
    """
    DELETE /api/history/<id>
    Deletes a specific detection record from SQLite and cleans up associated files.
    """
    record = get_detection_by_id(record_id)
    if not record:
        return jsonify({"error": f"Record #{record_id} not found."}), 404

    for fname in [record.get("filename"), record.get("result_filename")]:
        if fname:
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass

    deleted = delete_detection(record_id)
    if deleted:
        return jsonify({
            "success": True,
            "message": f"Record #{record_id} successfully deleted."
        }), 200
    else:
        return jsonify({"error": "Failed to delete record."}), 500

@app.route('/api/analytics-data', methods=['GET'])
def api_analytics_data():
    """Returns real-time analytics aggregation JSON for frontend charts."""
    summary = get_analytics_summary()
    return jsonify(summary), 200

# ----------------- Error Handlers -----------------

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File size exceeds 16MB maximum limit."}), 413

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({"error": "Endpoint or resource not found."}), 404
    return render_template('index.html', error="Page not found.", default_confidence=DEFAULT_CONFIDENCE), 404

@app.errorhandler(500)
def server_error(error):
    if request.path.startswith('/api/'):
        return jsonify({"error": "Internal server error."}), 500
    return render_template('index.html', error="An internal error occurred.", default_confidence=DEFAULT_CONFIDENCE), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
