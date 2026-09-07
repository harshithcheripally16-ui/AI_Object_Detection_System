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

def test_index_returns_200(client):
    """Test GET / returns 200 and loads HTML template."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'VisionTrack AI' in response.data
    assert b'YOLOv8' in response.data

def test_analytics_returns_200(client):
    """Test GET /analytics returns 200."""
    response = client.get('/analytics')
    assert response.status_code == 200
    assert b'YOLOv8 Analytics' in response.data

def test_detect_no_file_returns_400(client):
    """Test POST /detect without file returns 400."""
    response = client.post('/detect', data={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_detect_invalid_extension_returns_415(client):
    """Test POST /detect with invalid extension returns 415."""
    data = {
        'image': (io.BytesIO(b"fake text content"), 'test.txt'),
        'confidence': 0.5
    }
    response = client.post('/detect', data=data, content_type='multipart/form-data')
    assert response.status_code == 415
    json_data = response.get_json()
    assert "Invalid file type" in json_data["error"]

def test_detect_valid_image_returns_200_and_json(client):
    """Test POST /detect with valid image returns 200 and JSON response."""
    data = {
        'image': (create_synthetic_image_bytes(), 'test.jpg'),
        'confidence': 0.5
    }
    response = client.post('/detect', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    json_data = response.get_json()
    assert "result_image" in json_data
    assert "total_count" in json_data
    assert "processing_time_ms" in json_data
    assert "confidence_used" in json_data
    assert "detections" in json_data

def test_detect_response_has_required_keys(client):
    """Test POST /detect response has all required fields."""
    data = {
        'image': (create_synthetic_image_bytes(), 'sample.jpg'),
        'confidence': 0.6
    }
    response = client.post('/detect', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    res = response.get_json()
    for key in ["result_image", "total_count", "processing_time_ms", "confidence_used", "detections"]:
        assert key in res

def test_analytics_data_returns_json(client):
    """Test GET /api/analytics-data returns expected JSON aggregation."""
    response = client.get('/api/analytics-data')
    assert response.status_code == 200
    data = response.get_json()
    assert 'total_images' in data
    assert 'total_objects' in data
    assert 'avg_processing_time_ms' in data
    assert 'class_distribution' in data
    assert 'top_class' in data

def test_delete_history_valid_id_returns_200(client):
    """Test DELETE /api/history/<id> deletes record successfully."""
    # First create a detection
    data = {
        'image': (create_synthetic_image_bytes(), 'test_del.jpg'),
        'confidence': 0.5
    }
    detect_res = client.post('/detect', data=data, content_type='multipart/form-data')
    rec_id = detect_res.get_json()["id"]

    # Delete record
    delete_res = client.delete(f'/api/history/{rec_id}')
    assert delete_res.status_code == 200
    del_json = delete_res.get_json()
    assert del_json["success"] is True

def test_delete_history_invalid_id_returns_404(client):
    """Test DELETE /api/history/<id> with non-existent ID returns 404."""
    response = client.delete('/api/history/999999')
    assert response.status_code == 404

def test_clear_all_history(client):
    """Test DELETE /api/history/clear-all clears table and resets autoincrement sequence."""
    # Create detection
    data = {
        'image': (create_synthetic_image_bytes(), 'sample_clear.jpg'),
        'confidence': 0.5
    }
    client.post('/detect', data=data, content_type='multipart/form-data')

    # Clear all
    clear_res = client.delete('/api/history/clear-all')
    assert clear_res.status_code == 200
    assert clear_res.get_json()["success"] is True

    # Check history is empty
    hist_res = client.get('/api/history')
    assert hist_res.status_code == 200
    assert len(hist_res.get_json()) == 0
