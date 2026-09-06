import os
import uuid
import cv2
from flask import Flask, render_template, request, jsonify, redirect, url_for, abort
from werkzeug.utils import secure_filename

from config import (
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    MODELS,
    DEFAULT_MODEL
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
detector = ObjectDetector(MODELS)

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- UI Web Routes -----------------

@app.route('/')
def index():
    """Renders the main detection dashboard."""
    return render_template('index.html', models=MODELS, default_model=DEFAULT_MODEL)

@app.route('/result/<int:record_id>')
def result_view(record_id):
    """Renders detailed side-by-side result inspector for a specific detection run."""
    record = get_detection_by_id(record_id)
    if not record:
        return render_template('result.html', error="Detection record not found."), 404
    return render_template('result.html', record=record)

@app.route('/analytics')
def analytics_view():
    """Renders analytics dashboard page."""
    summary = get_analytics_summary()
    history = get_all_detections(limit=50)
    return render_template('analytics.html', summary=summary, history=history)

# ----------------- REST API Endpoints -----------------

@app.route('/detect', methods=['POST'])
def detect():
    """
    POST /detect
    Accepts multipart/form-data:
      - 'image': file upload
      - 'model': (optional) 'face', 'eye', 'fullbody' (default: 'face')
      
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

    # 3. Validate selected model
    model_type = request.form.get('model', DEFAULT_MODEL).strip().lower()
    if model_type not in MODELS:
        return jsonify({
            "error": f"Invalid model '{model_type}'. Available: {list(MODELS.keys())}"
        }), 400

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

        # 6. Execute object detection & latency tracking
        img, detections, processing_time_ms = detector.detect(img, model_type=model_type)
        annotated_img = detector.draw_boxes(img, detections, model_type=model_type)
        stats = detector.get_stats(detections, img.shape)

        # 7. Save annotated result image
        cv2.imwrite(result_filepath, annotated_img)

        # 8. Log run in SQLite
        record_id = log_detection(
            filename=orig_filename,
            result_filename=result_filename,
            model_used=model_type,
            count=stats["total_detections"],
            processing_time_ms=processing_time_ms,
            coordinates_data=stats["coordinates"]
        )

        # 9. Return structured response
        return jsonify({
            "id": record_id,
            "count": stats["total_detections"],
            "model_used": model_type,
            "model_name": MODELS[model_type]["name"],
            "processing_time_ms": processing_time_ms,
            "original_image": f"/static/uploads/{orig_filename}",
            "result_image": f"/static/uploads/{result_filename}",
            "result_page_url": f"/result/{record_id}",
            "stats": stats
        }), 200

    except Exception as e:
        return jsonify({"error": f"Internal inference error: {str(e)}"}), 500

@app.route('/api/history', methods=['GET'])
def api_history():
    """Returns historical detection runs as JSON."""
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

    # Remove files if present
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
    return render_template('index.html', error="Page not found.", models=MODELS), 404

@app.errorhandler(500)
def server_error(error):
    if request.path.startswith('/api/'):
        return jsonify({"error": "Internal server error."}), 500
    return render_template('index.html', error="An internal error occurred.", models=MODELS), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
