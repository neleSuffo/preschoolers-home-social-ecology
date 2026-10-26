"""
src/feature_extraction/run_video_detection.py

Corpus batch extraction engine for YOLO face, person, and book detectors.
Iterates over extracted frames, executes model inference, performs proximity
estimation (faces) or class filtering (books), and persists detections into SQLite.
"""

import argparse
import logging
from pathlib import Path
import shutil
import sqlite3
from typing import Callable, List, Optional, Tuple

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from src.config import DetectionConfig, VideoSpecs
from src.constants import AnalysisPaths, ModelAssets, PipelineLogs
from src.feature_extraction.download_model_weights import download_weights
from src.feature_extraction.utils import (
    extract_frame_number,
    get_frame_paths,
    get_video_id,
    load_processed_videos,
    save_processed_video,
)
from src.models.proximity import calculate_proximity

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task-Specific Frame Processors
# ---------------------------------------------------------------------------
def process_face_frame(
    frame_path: Path,
    video_id: int,
    frame_number: int,
    model: YOLO,
    cursor: sqlite3.Cursor,
) -> int:
    """
    Execute YOLO face detection on a frame and persist records into FaceDetections.

    Parameters
    ----------
    frame_path : Path
        Filesystem path to the image frame.
    video_id : int
        Foreign key identifier of the parent video in the database.
    frame_number : int
        Chronological index of the frame.
    model : YOLO
        Loaded Ultralytics face detection model.
    cursor : sqlite3.Cursor
        Active database cursor.

    Returns
    -------
    int
        Number of valid face detections recorded.
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        logger.warning("Failed to read image frame: %s", frame_path)
        return 0

    results = model.predict(
        frame,
        conf=DetectionConfig.FACE_CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    detections_batch = []
    face_count = 0

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            proximity = calculate_proximity([x1, y1, x2, y2], class_id=class_id)
            if proximity is None:
                continue

            detections_batch.append((
                video_id,
                frame_number,
                DetectionConfig.FACE_MODEL_ID,
                confidence,
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                class_id,
                proximity,
            ))
            face_count += 1

    if detections_batch:
        cursor.executemany(
            """
            INSERT INTO FaceDetections (
                video_id, frame_number, model_id, confidence_score,
                x_min, y_min, x_max, y_max, age_class, proximity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            detections_batch,
        )

    return face_count


def process_person_frame(
    frame_path: Path,
    video_id: int,
    frame_number: int,
    model: YOLO,
    cursor: sqlite3.Cursor,
) -> int:
    """
    Execute YOLO person detection on a frame and persist records into PersonDetections.

    Parameters
    ----------
    frame_path : Path
        Filesystem path to the image frame.
    video_id : int
        Foreign key identifier of the parent video in the database.
    frame_number : int
        Chronological index of the frame.
    model : YOLO
        Loaded Ultralytics person detection model.
    cursor : sqlite3.Cursor
        Active database cursor.

    Returns
    -------
    int
        Number of valid person detections recorded.
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        logger.warning("Failed to read image frame: %s", frame_path)
        return 0

    results = model.predict(
        frame,
        conf=DetectionConfig.PERSON_CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    detections_batch = []
    person_count = 0

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())


            detections_batch.append((
                video_id,
                frame_number,
                DetectionConfig.PERSON_MODEL_ID,
                confidence,
                class_id,
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ))
            person_count += 1

    if detections_batch:
        cursor.executemany(
            """
            INSERT INTO PersonDetections (
                video_id, frame_number, model_id, confidence_score,
                age_class, x_min, y_min, x_max, y_max
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            detections_batch,
        )

    return person_count


def process_book_frame(
    frame_path: Path,
    video_id: int,
    frame_number: int,
    model: YOLO,
    cursor: sqlite3.Cursor,
) -> int:
    """
    Execute COCO object detection on a frame and persist book detections into BookDetections.

    Parameters
    ----------
    frame_path : Path
        Filesystem path to the image frame.
    video_id : int
        Foreign key identifier of the parent video in the database.
    frame_number : int
        Chronological index of the frame.
    model : YOLO
        Loaded Ultralytics model
    cursor : sqlite3.Cursor
        Active database cursor.

    Returns
    -------
    int
        Number of valid book detections recorded.
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        logger.warning("Failed to read image frame: %s", frame_path)
        return 0

    results = model.predict(
        frame,
        verbose=False,
    )

    detections_batch = []
    book_count = 0

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].cpu().numpy())

            # Filter strictly for COCO 'book' category (class ID 73)
            if class_id in DetectionConfig.BOOK_CATEGORY_IDS:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())

                detections_batch.append((
                    video_id,
                    frame_number,
                    DetectionConfig.BOOK_MODEL_ID,
                    confidence,
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ))
                book_count += 1

    if detections_batch:
        cursor.executemany(
            """
            INSERT INTO BookDetections (
                video_id, frame_number, model_id, confidence_score,
                x_min, y_min, x_max, y_max
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            detections_batch,
        )

    return book_count


# ---------------------------------------------------------------------------
# Core Stream & Extraction Logic
# ---------------------------------------------------------------------------
def process_video_stream(
    video_name: str,
    frames_dir_root: Path,
    frame_step: int,
    model: YOLO,
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    process_fn: Callable[[Path, int, int, YOLO, sqlite3.Cursor], int],
) -> None:
    """
    Iterate over sampled video frames, execute inference, and commit batches to SQLite.

    Parameters
    ----------
    video_name : str
        Base filename of the video session.
    frames_dir_root : Path
        Directory housing the extracted frame folders.
    frame_step : int
        Sub-sampling step interval (e.g. 10 for 3 Hz at 30 fps).
    model : YOLO
        Loaded detection model.
    cursor : sqlite3.Cursor
        Active database cursor.
    conn : sqlite3.Connection
        Active database connection.
    process_fn : Callable
        Task-specific frame callback handler.
    """
    logger.info("Processing video: %s", video_name)
    video_id = get_video_id(video_name, cursor)
    if video_id is None:
        raise ValueError(f"Video {video_name} not registered in Videos table.")

    video_frames_dir = frames_dir_root / video_name
    if not video_frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {video_frames_dir}")

    frame_files = get_frame_paths(video_frames_dir)
    if not frame_files:
        raise ValueError(f"No extracted frames found in {video_frames_dir}")

    # Sub-sample frames matching the analysis frequency
    filtered_frames: List[Tuple[Path, int]] = []
    for fp in frame_files:
        try:
            num = extract_frame_number(fp.name)
            if frame_step > 1 and (num % frame_step != 0):
                continue
            filtered_frames.append((fp, num))
        except Exception:
            continue

    total_detections = 0
    with tqdm(filtered_frames, desc=f"{video_name}", unit="frames") as pbar:
        for idx, (frame_file, frame_number) in enumerate(pbar, 1):
            count = process_fn(frame_file, video_id, frame_number, model, cursor)
            total_detections += count

            pbar.set_postfix({"detections": total_detections, "frame": frame_number})

            if idx % 100 == 0:
                conn.commit()

    conn.commit()


def run_pipeline(
    task: str,
    video_list: List[str],
    frame_step: int = VideoSpecs.FRAME_STEP_INTERVAL,
    force: bool = False,
    frames_dir: Optional[Path] = None,
) -> None:
    """
    Orchestrate full video corpus detection runs for face, person, or book tasks.

    Parameters
    ----------
    task : str
        Target detection domain: 'face', 'person', or 'book'.
    video_list : List[str]
        List of video names to process.
    frame_step : int, default=10
        Sub-sampling step interval.
    force : bool, default=False
        If True, reprocesses videos even if logged as complete.
    frames_dir : Optional[Path], default=None
        Directory containing extracted frame folders.
    """
    # Map tasks to logging paths, model weights, and frame handlers
    if task == "face":
        log_path = PipelineLogs.FACE_LOG
        weights_path = ModelAssets.FACE_MODEL_PATH
        process_fn = process_face_frame
    elif task == "person":
        log_path = PipelineLogs.PERSON_LOG
        weights_path = ModelAssets.PERSON_MODEL_PATH
        process_fn = process_person_frame
    elif task == "book":
        log_path = PipelineLogs.BOOK_LOG
        weights_path = Path(DetectionConfig.BOOK_MODEL_NAME)
        process_fn = process_book_frame
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'face', 'person', or 'book'.")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    if force and log_path.exists():
        logger.info("Force flag set: archiving previous log to .backup")
        shutil.copy2(log_path, log_path.with_suffix(".txt.backup"))
        log_path.unlink()

    processed = load_processed_videos(log_path)
    pending = [v for v in video_list if v not in processed]

    if not pending:
        logger.info("All %d requested videos have already been processed for %s.", len(video_list), task)
        return

    # Ensure custom fine-tuned weights exist; automatically fetch from Hugging Face if missing
    if task in ("face", "person") and not weights_path.exists():
        logger.warning(
            "Model checkpoint missing at %s. Automatically downloading from Hugging Face...",
            weights_path,
        )
        try:
            download_weights(weights_list=[weights_path.name])
        except Exception as e:
            logger.error("Automated weight download failed for '%s': %s", weights_path.name, e)
            return

    logger.info("Loading model weights for %s from: %s", task, weights_path)
    model = YOLO(str(weights_path))

    # Uses CLI --frames_dir if provided, otherwise falls back to the constant
    root_frames_dir = frames_dir or AnalysisPaths.FRAMES_DIR
    
    with sqlite3.connect(AnalysisPaths.INFERENCE_DB) as conn:
        cursor = conn.cursor()
        for video_name in pending:
            try:
                process_video_stream(
                    video_name=video_name,
                    frames_dir_root=root_frames_dir,
                    frame_step=frame_step,
                    model=model,
                    cursor=cursor,
                    conn=conn,
                    process_fn=process_fn,
                )
                save_processed_video(log_path, video_name)
            except Exception as e:
                logger.error("Error encountered while processing %s: %s", video_name, e)
                continue

    logger.info("Batch %s detection processing completed.", task)


def main() -> None:
    """CLI entrypoint for batch extraction."""
    parser = argparse.ArgumentParser(description="Corpus Batch Detection Engine")
    parser.add_argument(
        "--task",
        choices=["face", "person", "book"],
        required=True,
        help="Detection task: 'face', 'person', or 'book'",
    )
    parser.add_argument(
        "--video_list",
        nargs="+",
        required=True,
        help="Space-separated video IDs",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=VideoSpecs.FRAME_STEP_INTERVAL,
        help="Frame sample interval (default: 10)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess already logged videos",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=None,
        help="Root folder of extracted frames",
    )
    args = parser.parse_args()

    custom_frames = Path(args.frames_dir) if args.frames_dir else None
    run_pipeline(
        task=args.task,
        video_list=args.video_list,
        frame_step=args.frame_step,
        force=args.force,
        frames_dir=custom_frames,
    )


if __name__ == "__main__":
    main()