import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, DEFAULT_CONFIDENCE

class ObjectDetector:
    """
    ObjectDetector handles multi-class object detection across 80+ everyday categories
    using Ultralytics YOLOv8 architecture (yolov8n.pt).
    """
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model_path = model_path
        self.model = YOLO(model_path)

    def detect(self, image_input, confidence=DEFAULT_CONFIDENCE):
        """
        Runs YOLOv8 multi-class inference on an image path or numpy BGR array.
        
        Args:
            image_input: File path (str) or numpy ndarray (BGR image)
            confidence (float): Confidence threshold (0.1 to 0.9)
            
        Returns:
            tuple: (results, processing_time_ms)
        """
        # Ensure image is valid
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not decode image at {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise TypeError("image_input must be a file path (str) or numpy ndarray")

        start_time = time.perf_counter()
        results = self.model.predict(img, conf=confidence, verbose=False)
        end_time = time.perf_counter()

        processing_time_ms = round((end_time - start_time) * 1000, 2)
        return results, processing_time_ms

    def draw_boxes(self, results):
        """
        Uses Ultralytics native plot annotator to draw bounding boxes, class names,
        and confidence percentages.
        
        Args:
            results: YOLO inference result list
            
        Returns:
            numpy.ndarray: Annotated BGR image
        """
        if not results or len(results) == 0:
            raise ValueError("No results provided to draw_boxes")
        return results[0].plot()

    def get_stats(self, results):
        """
        Extracts structured per-detection stats: class name, confidence, and bounding box.
        
        Args:
            results: YOLO inference results list
            
        Returns:
            dict: {"total": int, "objects": [{"class": str, "confidence": float, "bbox": [x1, y1, x2, y2]}]}
        """
        if not results or len(results) == 0:
            return {"total": 0, "objects": []}

        detections = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                conf_val = float(box.conf[0].item()) if hasattr(box.conf[0], 'item') else float(box.conf[0])
                xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]

                class_name = result.names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "class": class_name,
                    "confidence": round(conf_val, 2),
                    "bbox": xyxy
                })

        return {
            "total": len(detections),
            "objects": detections
        }
