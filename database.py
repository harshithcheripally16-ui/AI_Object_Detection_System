import sqlite3
import json
from collections import Counter
from config import DB_PATH

def get_db_connection():
    """Returns a SQLite connection with Row row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema for YOLOv8 multi-class detection records."""
    with get_db_connection() as conn:
        c = conn.cursor()
        
        # Check if table exists with old schema, recreate if columns differ
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detections'")
        table_exists = c.fetchone()
        
        if table_exists:
            c.execute("PRAGMA table_info(detections)")
            cols = [col["name"] for col in c.fetchall()]
            if 'confidence_threshold' not in cols or 'detections_json' not in cols:
                c.execute("DROP TABLE detections")

        c.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                result_filename TEXT NOT NULL,
                confidence_threshold REAL NOT NULL,
                total_count INTEGER NOT NULL,
                processing_time_ms REAL NOT NULL,
                detections_json TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_count ON detections(total_count)')
        conn.commit()

def log_detection(filename, result_filename, confidence_threshold, total_count, processing_time_ms, detections_data):
    """
    Logs a YOLOv8 detection run into SQLite.
    
    Args:
        filename (str): Uploaded original filename
        result_filename (str): Output annotated image filename
        confidence_threshold (float): Confidence cutoff used
        total_count (int): Number of detected objects
        processing_time_ms (float): Inference execution time in ms
        detections_data (list or dict): Per-object detections JSON data
        
    Returns:
        int: Inserted row ID
    """
    det_json = json.dumps(detections_data) if not isinstance(detections_data, str) else detections_data
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO detections (filename, result_filename, confidence_threshold, total_count, processing_time_ms, detections_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        ''', (filename, result_filename, confidence_threshold, total_count, processing_time_ms, det_json))
        conn.commit()
        return c.lastrowid

def get_all_detections(limit=100, offset=0):
    """Fetches list of detection records ordered by ID ascending."""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''
            SELECT id, filename, result_filename, confidence_threshold, total_count, processing_time_ms, detections_json, timestamp
            FROM detections
            ORDER BY id ASC
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        rows = c.fetchall()
        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "filename": r["filename"],
                "result_filename": r["result_filename"],
                "confidence_threshold": r["confidence_threshold"],
                "total_count": r["total_count"],
                "processing_time_ms": r["processing_time_ms"],
                "detections": json.loads(r["detections_json"]) if r["detections_json"] else [],
                "timestamp": r["timestamp"]
            })
        return results

def get_detection_by_id(record_id):
    """Fetches a single detection record by primary key."""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM detections WHERE id = ?', (record_id,))
        row = c.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "filename": row["filename"],
            "result_filename": row["result_filename"],
            "confidence_threshold": row["confidence_threshold"],
            "total_count": row["total_count"],
            "processing_time_ms": row["processing_time_ms"],
            "detections": json.loads(row["detections_json"]) if row["detections_json"] else [],
            "timestamp": row["timestamp"]
        }

def delete_detection(record_id):
    """
    Deletes a detection log by primary key. If table becomes empty, resets autoincrement sequence back to 1.
    
    Returns:
        bool: True if deleted, False if record was not found.
    """
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('DELETE FROM detections WHERE id = ?', (record_id,))
        deleted = c.rowcount > 0
        if deleted:
            c.execute('SELECT COUNT(*) as cnt FROM detections')
            if c.fetchone()['cnt'] == 0:
                c.execute("DELETE FROM sqlite_sequence WHERE name='detections'")
        conn.commit()
        return deleted

def clear_all_detections():
    """
    Deletes all detection records from SQLite and resets the autoincrement ID counter back to 1.
    
    Returns:
        bool: True
    """
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('DELETE FROM detections')
        c.execute("DELETE FROM sqlite_sequence WHERE name='detections'")
        conn.commit()
        return True

def get_analytics_summary():
    """
    Aggregates metrics across all historical YOLOv8 detection runs.
    
    Returns:
        dict: total_images, total_objects, avg_processing_time_ms, top_class,
              class_distribution, and timeline.
    """
    with get_db_connection() as conn:
        c = conn.cursor()
        
        # Overall totals
        c.execute('''
            SELECT 
                COUNT(*) as total_images,
                COALESCE(SUM(total_count), 0) as total_objects,
                COALESCE(AVG(total_count), 0) as avg_objects_per_image,
                COALESCE(AVG(processing_time_ms), 0) as avg_latency_ms
            FROM detections
        ''')
        overview = c.fetchone()

        # Fetch detections_json for class distribution aggregation
        c.execute('SELECT detections_json FROM detections')
        all_json_rows = c.fetchall()

        class_counter = Counter()
        for r in all_json_rows:
            if r["detections_json"]:
                try:
                    items = json.loads(r["detections_json"])
                    for item in items:
                        cls_name = item.get("class", "unknown")
                        class_counter[cls_name] += 1
                except Exception:
                    pass

        top_classes = [
            {"class": cls_name, "count": count}
            for cls_name, count in class_counter.most_common(10)
        ]
        most_detected_class = top_classes[0]["class"].capitalize() if top_classes else "None"

        # Recent timeline data (last 20 detections for line chart)
        c.execute('''
            SELECT id, timestamp, total_count, processing_time_ms, confidence_threshold
            FROM detections
            ORDER BY id ASC
            LIMIT 20
        ''')
        timeline = [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "total_count": row["total_count"],
                "latency_ms": row["processing_time_ms"],
                "confidence": row["confidence_threshold"]
            }
            for row in c.fetchall()
        ]

        return {
            "total_images": overview["total_images"] if overview else 0,
            "total_objects": overview["total_objects"] if overview else 0,
            "avg_objects_per_image": round(overview["avg_objects_per_image"], 2) if overview else 0.0,
            "avg_processing_time_ms": round(overview["avg_latency_ms"], 2) if overview else 0.0,
            "top_class": most_detected_class,
            "class_distribution": top_classes,
            "timeline": timeline
        }
