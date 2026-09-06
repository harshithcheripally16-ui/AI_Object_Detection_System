import pytest
import numpy as np
from detector import ObjectDetector
from config import YOLO_MODEL_PATH

@pytest.fixture
def detector():
    return ObjectDetector(YOLO_MODEL_PATH)

@pytest.fixture
def sample_image():
    """Generates a synthetic 300x300 BGR test image."""
    return np.zeros((300, 300, 3), dtype=np.uint8)

def test_detector_loads_model(detector):
    """Test YOLO model initializes without error."""
    assert detector.model is not None
    assert hasattr(detector.model, 'predict')

def test_detect_returns_results(detector, sample_image):
    """Test valid image returns results and processing time in ms."""
    results, ms = detector.detect(sample_image, confidence=0.5)
    assert results is not None
    assert isinstance(ms, (int, float))
    assert ms >= 0

def test_detect_blank_image(detector, sample_image):
    """Test blank image returns 0 detections."""
    results, ms = detector.detect(sample_image, confidence=0.5)
    stats = detector.get_stats(results)
    assert stats["total"] == 0
    assert stats["objects"] == []

def test_get_stats_structure(detector, sample_image):
    """Test stats dict has 'total' and 'objects' keys."""
    results, _ = detector.detect(sample_image, confidence=0.5)
    stats = detector.get_stats(results)
    assert "total" in stats
    assert "objects" in stats
    assert isinstance(stats["objects"], list)

def test_confidence_threshold(detector, sample_image):
    """Test high confidence threshold (0.95) runs without errors."""
    results_high, _ = detector.detect(sample_image, confidence=0.95)
    stats_high = detector.get_stats(results_high)
    assert stats_high["total"] == 0

def test_draw_boxes_returns_array(detector, sample_image):
    """Test draw_boxes returns numpy array matching image dimensions."""
    results, _ = detector.detect(sample_image, confidence=0.5)
    annotated = detector.draw_boxes(results)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == sample_image.shape
