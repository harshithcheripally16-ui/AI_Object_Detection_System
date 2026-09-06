import sqlite3
import json
from datetime import datetime
from config import DB_PATH

def get_db_connection():
    """Returns a SQLite connection with Row row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema and indexes."""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                result_filename TEXT NOT NULL,
                model_used TEXT NOT NULL,
                count INTEGER NOT NULL,
                processing_time_ms REAL NOT NULL,
                coordinates_json TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_model ON detections(model_used)')
        conn.commit()

def log_detection(filename, result_filename, model_used, count, processing_time_ms, coordinates_data):
    """
    Logs a detection run into SQLite.
    
    Args:
        filename (str): Uploaded original filename
        result_filename (str): Output annotated image filename
        model_used (str): Name or key of the model used
        count (int): Number of detected objects
        processing_time_ms (float): Execution time in milliseconds
        coordinates_data (list or dict): Coordinates metadata to serialize as JSON
        
    Returns:
        int: Inserted row ID
    """
    coords_json = json.dumps(coordinates_data) if not isinstance(coordinates_data, str) else coordinates_data
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO detections (filename, result_filename, model_used, count, processing_time_ms, coordinates_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        ''', (filename, result_filename, model_used, count, processing_time_ms, coords_json))
        conn.commit()
        return c.lastrowid

def get_all_detections(limit=100, offset=0):
    """Fetches list of detection records ordered by timestamp descending."""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''
            SELECT id, filename, result_filename, model_used, count, processing_time_ms, coordinates_json, timestamp
            FROM detections
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        rows = c.fetchall()
        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "filename": r["filename"],
                "result_filename": r["result_filename"],
                "model_used": r["model_used"],
                "count": r["count"],
                "processing_time_ms": r["processing_time_ms"],
                "coordinates": json.loads(r["coordinates_json"]) if r["coordinates_json"] else [],
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
            "model_used": row["model_used"],
            "count": row["count"],
            "processing_time_ms": row["processing_time_ms"],
            "coordinates": json.loads(row["coordinates_json"]) if row["coordinates_json"] else [],
            "timestamp": row["timestamp"]
        }

def delete_detection(record_id):
    """
    Deletes a detection log by primary key.
    
    Returns:
        bool: True if deleted, False if record was not found.
    """
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('DELETE FROM detections WHERE id = ?', (record_id,))
        conn.commit()
        return c.rowcount > 0

def get_analytics_summary():
    """
    Aggregates metrics across all historical detection runs.
    
    Returns:
        dict with total_images, total_detections, avg_detections, avg_latency_ms,
        model_distribution, and recent_activity.
    """
    with get_db_connection() as conn:
        c = conn.cursor()
        
        # Overall totals
        c.execute('''
            SELECT 
                COUNT(*) as total_images,
                COALESCE(SUM(count), 0) as total_detections,
                COALESCE(AVG(count), 0) as avg_detections,
                COALESCE(AVG(processing_time_ms), 0) as avg_latency_ms
            FROM detections
        ''')
        overview = c.fetchone()
        
        # Model breakdown
        c.execute('''
            SELECT model_used, COUNT(*) as usage_count, COALESCE(SUM(count), 0) as detections_count
            FROM detections
            GROUP BY model_used
        ''')
        model_stats = [
            {"model": row["model_used"], "runs": row["usage_count"], "detections": row["detections_count"]}
            for row in c.fetchall()
        ]
        
        # Recent timeline data (last 10 detections for charts)
        c.execute('''
            SELECT id, timestamp, count, processing_time_ms, model_used
            FROM detections
            ORDER BY id ASC
            LIMIT 30
        ''')
        timeline = [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "count": row["count"],
                "latency_ms": row["processing_time_ms"],
                "model": row["model_used"]
            }
            for row in c.fetchall()
        ]

        return {
            "total_images": overview["total_images"] if overview else 0,
            "total_detections": overview["total_detections"] if overview else 0,
            "avg_detections": round(overview["avg_detections"], 2) if overview else 0.0,
            "avg_latency_ms": round(overview["avg_latency_ms"], 2) if overview else 0.0,
            "models_breakdown": model_stats,
            "timeline": timeline
        }
