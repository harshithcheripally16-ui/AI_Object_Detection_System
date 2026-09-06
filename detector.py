import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, FALLBACK_MODEL_PATH, DETECTABLE_CLASSES, DEFAULT_CONFIDENCE

class ObjectDetector:
    """
    ObjectDetector handles high-accuracy multi-class object detection across a rich
    vocabulary of everyday objects, gaming gear (Razer mice, headphones), hands,
    tech equipment, and living beings using Ultralytics YOLO-World / YOLOv8 architecture.
    """
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Loads model weights and configures the open-vocabulary classes."""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
            elif os.path.exists(FALLBACK_MODEL_PATH):
                self.model = YOLO(FALLBACK_MODEL_PATH)
            else:
                self.model = YOLO(self.model_path)  # Auto-downloads if needed
            
            # Configure custom open-vocabulary classes if YOLO-World architecture
            if hasattr(self.model, 'set_classes') and DETECTABLE_CLASSES:
                try:
                    self.model.set_classes(DETECTABLE_CLASSES)
                except Exception as e:
                    print(f"Warning: set_classes failed ({e}), using default model classes.")
        except Exception as e:
            # Fallback to standard YOLOv8n
            print(f"Error loading {self.model_path}: {e}. Falling back to standard YOLOv8n.")
            self.model = YOLO('models/yolov8n.pt')

    def detect(self, image_input, confidence=DEFAULT_CONFIDENCE):
        """
        Runs YOLO inference on an image with a strict confidence threshold.
        
        Args:
            image_input: File path (str) or numpy ndarray (BGR image)
            confidence (float): Confidence threshold (0.05 to 0.95)
            
        Returns:
            tuple: (results, processing_time_ms)
        """
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

        # Clamp confidence threshold between 0.05 and 0.95
        conf_val = max(0.05, min(0.95, float(confidence)))

        start_time = time.perf_counter()
        results = self.model.predict(img, conf=conf_val, iou=0.45, verbose=False)
        end_time = time.perf_counter()

        processing_time_ms = round((end_time - start_time) * 1000, 2)
        return results, processing_time_ms

    def draw_boxes(self, results):
        """
        Uses Ultralytics plot annotator to draw crisp bounding boxes, class names,
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
        Extracts structured per-detection stats: class name, confidence score, and bbox.
        
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
