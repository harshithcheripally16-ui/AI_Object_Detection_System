import io
import pytest
import numpy as np
import cv2

from app import app
from database import init_db

@pytest.fixture
def client():
    app.config['TESTING'] = True
    init_db()
    with app.test_client() as client:
        yield client

def create_synthetic_image_bytes():
    """Encodes a synthetic 100x100 BGR test image as JPEG bytes."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', img)
    return io.BytesIO(encoded.tobytes())

def test_index_route(client):
    """Test GET / returns 200 and loads HTML template."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'VisionTrack AI' in response.data
    assert b'Detector' in response.data

def test_analytics_route(client):
    """Test GET /analytics returns 200."""
    response = client.get('/analytics')
    assert response.status_code == 200
    assert b'System Telemetry' in response.data

def test_api_analytics_data(client):
    """Test GET /api/analytics-data returns expected JSON structure."""
    response = client.get('/api/analytics-data')
    assert response.status_code == 200
    data = response.get_json()
    assert 'total_images' in data
    assert 'total_detections' in data
    assert 'avg_latency_ms' in data
    assert 'models_breakdown' in data

def test_detect_no_file(client):
    """Test POST /detect without file returns 400."""
    response = client.post('/detect', data={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_detect_invalid_extension(client):
    """Test POST /detect with forbidden extension returns 415."""
    data = {
        'image': (io.BytesIO(b"fake text content"), 'test.txt'),
        'model': 'face'
    }
    response = client.post('/detect', data=data, content_type='multipart/form-data')
    assert response.status_code == 415
    json_data = response.get_json()
    assert "Invalid file type" in json_data["error"]

def test_detect_invalid_model(client):
    """Test POST /detect with unsupported model returns 400."""
    data = {
        'image': (create_synthetic_image_bytes(), 'sample.jpg'),
        'model': 'invalid_model_abc'
    }
    response = client.post('/detect', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Invalid model" in json_data["error"]

def test_detect_valid_flow_and_crud(client):
    """
    Test full end-to-end detection pipeline:
    1. Upload valid image via POST /detect
    2. Retrieve single record via GET /api/history/<id>
    3. View result page via GET /result/<id>
    4. Delete record via DELETE /api/history/<id>
    """
    # 1. Upload & Detect
    img_bytes = create_synthetic_image_bytes()
    data = {
        'image': (img_bytes, 'sample_test.jpg'),
        'model': 'face'
    }
    response = client.post('/detect', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    json_data = response.get_json()

    assert "id" in json_data
    assert "processing_time_ms" in json_data
    assert "result_image" in json_data
    assert "stats" in json_data
    record_id = json_data["id"]

    # 2. Query History endpoint
    history_res = client.get('/api/history')
    assert history_res.status_code == 200
    history_list = history_res.get_json()
    assert any(rec["id"] == record_id for rec in history_list)

    # 3. Query Single Record
    single_res = client.get(f'/api/history/{record_id}')
    assert single_res.status_code == 200
    record = single_res.get_json()
    assert record["id"] == record_id
    assert record["model_used"] == "face"

    # 4. Query Result HTML page
    result_page = client.get(f'/result/{record_id}')
    assert result_page.status_code == 200
    assert f'Detection Record #{record_id}'.encode() in result_page.data

    # 5. Delete Record via DELETE /api/history/<id>
    delete_res = client.delete(f'/api/history/{record_id}')
    assert delete_res.status_code == 200
    delete_data = delete_res.get_json()
    assert delete_data["success"] is True

    # 6. Verify Deletion
    deleted_check = client.get(f'/api/history/{record_id}')
    assert deleted_check.status_code == 404
