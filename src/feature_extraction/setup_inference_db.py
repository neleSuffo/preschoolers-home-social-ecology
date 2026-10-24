"""
src/database/setup_inference_db.py

Initializes the schema for the multimodal inference database (inference.db).
Creates tables for video sessions, models, YOLO face/person/book detections,
audio diarization records, and critical composite performance indexes.
Populates metadata from metadata.csv and calculates video frame counts.
"""

import argparse
import json
import logging
from pathlib import Path
import sqlite3
from typing import List, Optional

import cv2
import pandas as pd

from src.config import DetectionConfig
from src.constants import AnalysisPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_audio_table_schema(
    audio_classes: Optional[List[str]] = None,
) -> str:
    """
    Construct dynamic CREATE TABLE DDL for AudioClassifications 
    based on the configured audio classes.
    """
    if audio_classes is None:
        audio_classes = DetectionConfig.AUDIO_CLASSES

    column_defs = [
        "classification_id INTEGER PRIMARY KEY AUTOINCREMENT",
        "video_id INTEGER NOT NULL",
        "frame_number INTEGER NOT NULL",
        "model_id INTEGER NOT NULL",
    ]

    for token in audio_classes:
        # Standardize name (e.g., 'KCDS' -> 'cds', 'OHS' -> 'ohs')
        col_name = token.lower().replace("kcds", "cds")
        column_defs.append(f"has_{col_name} INTEGER CHECK(has_{col_name} IN (0, 1))")
        column_defs.append(f"{col_name}_confidence_score REAL")

    column_defs.extend([
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "FOREIGN KEY (video_id) REFERENCES Videos(video_id)",
        "FOREIGN KEY (model_id) REFERENCES Models(model_id)",
    ])

    formatted_cols = ",\n                ".join(column_defs)
    return f"""
        CREATE TABLE IF NOT EXISTS AudioClassifications (
            {formatted_cols}
        )
    """


def get_max_frame_count(video_path: Path) -> Optional[int]:
    """
    Retrieve the total number of frames in a video container using OpenCV.

    Parameters
    ----------
    video_path : Path
        Path to the target video file.

    Returns
    -------
    Optional[int]
        Total frame count, or None if the video cannot be opened.
    """
    if not video_path.exists():
        logger.warning("Video file not found: %s", video_path)
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Failed to open video container: %s", video_path)
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def setup_inference_schema(db_path: Path) -> None:
    """
    Create database tables and composite lookup indexes if they do not exist.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # 1. Video Sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Videos (
                video_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT UNIQUE NOT NULL,
                child_id INTEGER,
                recording_date DATE,
                age_at_recording REAL,
                age_group INTEGER,
                max_frame INTEGER
            )
        """)

        # 2. Registered Model Metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Models (
                model_id INTEGER PRIMARY KEY,
                model_name TEXT UNIQUE NOT NULL,
                description TEXT,
                output_variables TEXT
            )
        """)

        # 3. YOLO Face Detections & Proximity
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS FaceDetections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                model_id INTEGER NOT NULL,
                confidence_score REAL NOT NULL,
                x_min REAL NOT NULL,
                y_min REAL NOT NULL,
                x_max REAL NOT NULL,
                y_max REAL NOT NULL,
                age_class INTEGER CHECK(age_class IN (0, 1)),
                proximity REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES Videos(video_id),
                FOREIGN KEY (model_id) REFERENCES Models(model_id)
            )
        """)

        # 4. YOLO Person Detections
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS PersonDetections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                model_id INTEGER NOT NULL,
                confidence_score REAL NOT NULL,
                age_class INTEGER DEFAULT 0,
                x_min REAL NOT NULL,
                y_min REAL NOT NULL,
                x_max REAL NOT NULL,
                y_max REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES Videos(video_id),
                FOREIGN KEY (model_id) REFERENCES Models(model_id)
            )
        """)

        # 5. COCO Object / Book Detections
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS BookDetections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                model_id INTEGER NOT NULL,
                confidence_score REAL NOT NULL,
                x_min REAL NOT NULL,
                y_min REAL NOT NULL,
                x_max REAL NOT NULL,
                y_max REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES Videos(video_id),
                FOREIGN KEY (model_id) REFERENCES Models(model_id)
            )
        """)

        # 6. Dynamic Audio Diarization Classifications
        audio_table_sql = build_audio_table_schema(DetectionConfig.AUDIO_CLASSES)
        cursor.execute(audio_table_sql)

        # -------------------------------------------------------------------
        # B-Tree Indexes (Essential for Fast Multi-Modal Frame Alignment)
        # -------------------------------------------------------------------
        cursor.executescript("""
            CREATE INDEX IF NOT EXISTS idx_face_lookup ON FaceDetections(video_id, frame_number);
            CREATE INDEX IF NOT EXISTS idx_person_lookup ON PersonDetections(video_id, frame_number);
            CREATE INDEX IF NOT EXISTS idx_book_lookup ON BookDetections(video_id, frame_number);
            CREATE INDEX IF NOT EXISTS idx_audio_lookup ON AudioClassifications(video_id, frame_number);
        """)

        # Build dynamic output_variables mapping for registered audio classes
        audio_outputs = {
            cls: DetectionConfig.AUDIO_CLASS_DESCRIPTIONS.get(cls, cls.lower())
            for cls in DetectionConfig.AUDIO_CLASSES
        }

        # Populate Model Registry
        models_data = [
            (
                DetectionConfig.FACE_MODEL_ID,
                "yolo12_face",
                "Fine-tuned YOLOv12-l model for face detection and demographic age estimation.",
                '{"age_class": {"0": "child_face", "1": "adult_face"}, "proximity": "continuous_proximity_0_to_1"}',
            ),
            (
                DetectionConfig.PERSON_MODEL_ID,
                "yolo12_person",
                "Fine-tuned YOLOv12-l model for external social partner detection.",
                '{"person": "detected_bounding_box"}',
            ),
            (
                DetectionConfig.AUDIO_MODEL_ID,
                "vtc_speech_classifier",
                "Voice Type Classifier (VTC) for egocentric speech diarization.",
                json.dumps(audio_outputs),
            ),
            (
                DetectionConfig.BOOK_MODEL_ID,
                "yolo_book_detector",
                "Pretrained YOLO detector filtering illustrated storybook faces.",
                '{"book": "bounding_box"}',
            ),
        ]

        cursor.executemany(
            """
            INSERT INTO Models (model_id, model_name, description, output_variables)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                model_name=excluded.model_name,
                description=excluded.description,
                output_variables=excluded.output_variables
            """,
            models_data,
        )

        conn.commit()

    logger.info("Database schema and indexes initialized at: %s", db_path)


def populate_video_metadata(
    metadata_csv: Path,
    db_path: Path,
    raw_videos_dir: Optional[Path] = None,
) -> None:
    """
    Parse recording session metadata and populate the Videos table.

    Parameters
    ----------
    metadata_csv : Path
        Path to metadata.csv (delimited with semicolon).
    db_path : Path
        Target SQLite database path.
    raw_videos_dir : Optional[Path], default=None
        Directory containing raw MP4 video containers to compute max_frame.
    """
    if not metadata_csv.exists():
        logger.warning("Metadata CSV not found at %s. Skipping video population.", metadata_csv)
        return

    try:
        df = pd.read_csv(metadata_csv, sep=";", encoding="utf-8")
        # Handle comma or dot decimal notations in age
        df["age"] = (
            df["age"].astype(str).str.replace(",", ".", regex=False).astype(float)
        )
    except Exception as e:
        logger.error("Failed to read metadata CSV: %s", e)
        return

    records = []
    for _, row in df.iterrows():
        video_name = str(row["video_name"]).strip()
        child_id = int(row["child_id"]) if pd.notna(row["child_id"]) else None
        age_at_recording = float(row["age"]) if pd.notna(row["age"]) else None
        age_group = int(row["age_group"]) if pd.notna(row["age_group"]) else None

        # Parse date formatted as dd.mm.yyyy
        rec_date_str = None
        if pd.notna(row.get("recording_date")):
            try:
                rec_date_str = pd.to_datetime(row["recording_date"], format="%d.%m.%Y").strftime("%Y-%m-%d")
            except Exception:
                rec_date_str = str(row["recording_date"])

        # Locate raw video file if present to determine container frame count
        max_frame = None
        if raw_videos_dir and raw_videos_dir.exists():
            matched = list(raw_videos_dir.glob(f"{video_name}.*"))
            if matched:
                video_file = matched[0]
                if video_file.suffix.lower() == ".mp4":
                    max_frame = get_max_frame_count(video_file)

        records.append((video_name, child_id, rec_date_str, age_at_recording, age_group, max_frame))

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO Videos (video_name, child_id, recording_date, age_at_recording, age_group, max_frame)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_name) DO UPDATE SET
                child_id=excluded.child_id,
                recording_date=excluded.recording_date,
                age_at_recording=excluded.age_at_recording,
                age_group=excluded.age_group,
                max_frame=COALESCE(excluded.max_frame, Videos.max_frame)
            """,
            records,
        )
        conn.commit()

    logger.info("Populated metadata for %d video recordings into Videos table.", len(records))


def main() -> None:
    """CLI entrypoint to set up schema and populate video metadata."""
    parser = argparse.ArgumentParser(description="Initialize and populate inference.db")
    parser.add_argument(
        "--db_path",
        type=str,
        default=str(AnalysisPaths.INFERENCE_DB),
        help="Path to destination SQLite database",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=str(AnalysisPaths.METADATA_CSV),
        help="Path to recording metadata CSV",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default=str(AnalysisPaths.VIDEOS_DIR),
        help="Directory containing raw MP4 video files",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    metadata_csv = Path(args.metadata_csv)
    videos_dir = Path(args.videos_dir) if args.videos_dir else None

    setup_inference_schema(db_path)
    populate_video_metadata(metadata_csv, db_path, raw_videos_dir=videos_dir)


if __name__ == "__main__":
    main()