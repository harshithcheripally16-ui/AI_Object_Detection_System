import os
import time
import cv2
import numpy as np
from config import MODELS, DEFAULT_MODEL

class ObjectDetector:
    """
    ObjectDetector handles multi-model object and feature detection using OpenCV Haar Cascades.
    Supports model switching (Face, Eye, Full Body), latency measurement, bounding box styling,
    and extraction of spatial metrics.
    """
    def __init__(self, models_dict=None):
        self.models_config = models_dict or MODELS
        self.classifiers = {}
        self._load_models()

    def _load_models(self):
        """Preloads and verifies Haar Cascade XML models into memory."""
        for key, conf in self.models_config.items():
            model_file = conf['file']
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            classifier = cv2.CascadeClassifier(model_file)
            if classifier.empty():
                raise ValueError(f"Failed to load cascade classifier from {model_file}")
            
            self.classifiers[key] = classifier

    def detect(self, image_input, model_type=DEFAULT_MODEL, scale_factor=None, min_neighbors=None, min_size=None):
        """
        Runs detection on an image path or numpy BGR image array.
        
        Args:
            image_input: File path (str) or numpy ndarray (BGR image)
            model_type: One of 'face', 'eye', 'fullbody'
            scale_factor: Optional override for multiscale scaleFactor
            min_neighbors: Optional override for minNeighbors
            min_size: Optional override for minSize tuple (w, h)
            
        Returns:
            tuple: (img_bgr, detections_list, processing_time_ms)
        """
        if model_type not in self.classifiers:
            raise ValueError(f"Unknown model_type '{model_type}'. Available: {list(self.classifiers.keys())}")

        # Load image if file path is given
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not decode image at {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            raise TypeError("image_input must be a file path (str) or numpy array")

        conf = self.models_config[model_type]
        sf = scale_factor or conf.get('scale_factor', 1.1)
        mn = min_neighbors or conf.get('min_neighbors', 4)
        ms = min_size or conf.get('min_size', (30, 30))

        # Convert to grayscale for Haar cascades
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Inference & timing
        start_time = time.perf_counter()
        detections = self.classifiers[model_type].detectMultiScale(
            gray,
            scaleFactor=sf,
            minNeighbors=mn,
            minSize=ms,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        end_time = time.perf_counter()
        
        processing_time_ms = round((end_time - start_time) * 1000, 2)
        
        # Ensure detections is an iterable list/array
        if len(detections) == 0:
            detections = np.array([])
        
        return img, detections, processing_time_ms

    def draw_boxes(self, img, detections, model_type=DEFAULT_MODEL):
        """
        Draws annotated bounding boxes and labels onto the image.
        
        Args:
            img: BGR numpy image
            detections: List/array of (x, y, w, h) bounding boxes
            model_type: Model identifier to use corresponding color and label
            
        Returns:
            annotated_image (numpy ndarray)
        """
        annotated = img.copy()
        conf = self.models_config.get(model_type, {})
        color = conf.get('color', (0, 255, 0))
        label_prefix = conf.get('label', 'Object')

        for idx, (x, y, w, h) in enumerate(detections, start=1):
            # Draw outer rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label text
            tag = f"{label_prefix} #{idx} ({w}x{h})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(tag, font, font_scale, thickness)

            # Draw label background badge
            badge_y1 = max(0, y - text_h - 8)
            badge_y2 = y
            badge_x2 = min(annotated.shape[1], x + text_w + 10)
            cv2.rectangle(annotated, (x, badge_y1), (badge_x2, badge_y2), color, -1)

            # Draw white text inside badge
            cv2.putText(
                annotated,
                tag,
                (x + 5, y - 5 if y - 5 > 5 else y + text_h + 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        return annotated

    def get_stats(self, detections, img_shape=None):
        """
        Computes structured metadata regarding detections.
        
        Args:
            detections: List or array of (x, y, w, h)
            img_shape: Tuple (height, width, channels) of the source image
            
        Returns:
            dict containing total detections, box coordinates, centroids, and area statistics.
        """
        total = len(detections) if len(detections) > 0 else 0
        coords = []
        centroids = []
        total_detection_area = 0

        if total > 0:
            for item in detections:
                x, y, w, h = int(item[0]), int(item[1]), int(item[2]), int(item[3])
                coords.append({"x": x, "y": y, "w": w, "h": h})
                centroids.append({"cx": x + (w // 2), "cy": y + (h // 2)})
                total_detection_area += (w * h)

        stats = {
            "total_detections": total,
            "coordinates": coords,
            "centroids": centroids
        }

        if img_shape:
            img_h, img_w = img_shape[0], img_shape[1]
            img_area = img_h * img_w
            coverage_pct = round((total_detection_area / img_area) * 100, 2) if img_area > 0 else 0.0
            stats["image_dimensions"] = {"width": img_w, "height": img_h}
            stats["coverage_percentage"] = coverage_pct

        return stats
