import os
import pytest
import numpy as np
import cv2

from detector import ObjectDetector
from config import MODELS

@pytest.fixture
def detector():
    return ObjectDetector(MODELS)

@pytest.fixture
def sample_image():
    """Generates a synthetic 300x300 BGR test image."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 50, (255, 255, 255), -1)
    return img

def test_detector_initialization(detector):
    """Test that all models are properly loaded into the detector."""
    assert 'face' in detector.classifiers
    assert 'eye' in detector.classifiers
    assert 'fullbody' in detector.classifiers
    for model_name, classifier in detector.classifiers.items():
        assert not classifier.empty(), f"Model {model_name} classifier failed to load."

def test_detect_models(detector, sample_image):
    """Test detection across all supported model types."""
    for model_type in ['face', 'eye', 'fullbody']:
        img, detections, ms = detector.detect(sample_image, model_type=model_type)
        assert isinstance(img, np.ndarray)
        assert isinstance(ms, (int, float))
        assert ms >= 0
        assert isinstance(detections, (list, np.ndarray))

def test_draw_boxes(detector, sample_image):
    """Test that bounding box rendering preserves dimensions and format."""
    mock_detections = [(50, 50, 60, 60)]
    annotated = detector.draw_boxes(sample_image, mock_detections, model_type='face')
    assert annotated.shape == sample_image.shape
    assert isinstance(annotated, np.ndarray)

def test_get_stats_empty(detector):
    """Test metrics extraction when no detections are found."""
    stats = detector.get_stats([], img_shape=(300, 300, 3))
    assert stats["total_detections"] == 0
    assert stats["coordinates"] == []
    assert stats["centroids"] == []
    assert stats["coverage_percentage"] == 0.0

def test_get_stats_with_detections(detector):
    """Test metrics extraction with valid detections."""
    mock_detections = [(10, 20, 30, 40)]
    stats = detector.get_stats(mock_detections, img_shape=(200, 200, 3))
    assert stats["total_detections"] == 1
    assert stats["coordinates"] == [{"x": 10, "y": 20, "w": 30, "h": 40}]
    assert stats["centroids"] == [{"cx": 25, "cy": 40}]
    assert stats["coverage_percentage"] == round((30 * 40) / (200 * 200) * 100, 2)

def test_invalid_model_type(detector, sample_image):
    """Test error raising when requesting an unsupported model."""
    with pytest.raises(ValueError):
        detector.detect(sample_image, model_type='non_existent_model')

def test_invalid_image_path(detector):
    """Test error raising when image path is invalid."""
    with pytest.raises(FileNotFoundError):
        detector.detect("non_existent_path_12345.jpg")
