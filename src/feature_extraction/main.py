"""
src/feature_extraction/main.py

Top-level CLI orchestrator for feature extraction.
Sets up the relational inference database, ensures model checkpoints are present,
and dispatches batch extraction for face proximity, person detection, book filtering,
and speech diarization.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from constants import AnalysisPaths
from src.config import DetectionConfig, VideoSpecs
from src.feature_extraction.setup_inference_db import (
    populate_video_metadata,
    setup_inference_schema,
)
from src.feature_extraction.download_model_weights import download_weights
from src.feature_extraction.run_audio_ingestion import run_audio_pipeline
from src.feature_extraction.run_video_detection import run_pipeline as run_detection_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_video_list(video_input_path: Path) -> List[str]:
    """
    Parse a video path, directory, or text file into a list of clean video stems.

    Parameters
    ----------
    video_input_path : Path
        Path to a single video file, a directory of videos, or a .txt file.

    Returns
    -------
    List[str]
        Unique list of video identifiers (filenames without extension).
    """
    selected_videos: List[str] = []
    video_ext = VideoSpecs.VIDEO_FILE_EXTENSION.lower()

    if not video_input_path.exists():
        logger.error("Provided video path does not exist: %s", video_input_path)
        return []

    # Case 1: Text file containing list of video stems
    if video_input_path.is_file() and video_input_path.suffix.lower() == ".txt":
        logger.info("Reading video identifiers from text file: %s", video_input_path.name)
        with open(video_input_path, "r", encoding="utf-8") as f:
            selected_videos = [line.strip() for line in f if line.strip()]

    # Case 2: Single video file
    elif video_input_path.is_file() and video_input_path.suffix.lower() == video_ext:
        logger.info("Targeting single video: %s", video_input_path.name)
        selected_videos = [video_input_path.stem]

    # Case 3: Directory containing video files
    elif video_input_path.is_dir():
        logger.info("Scanning directory for %s videos: %s", video_ext, video_input_path)
        video_files = [
            f for f in video_input_path.iterdir()
            if f.is_file() and f.suffix.lower() == video_ext
        ]
        selected_videos = [f.stem for f in video_files]

    else:
        logger.error(
            "Unsupported file or path format for: %s (expected .txt, %s, or directory)",
            video_input_path,
            video_ext,
        )
        return []

    # Deduplicate while preserving order
    return list(dict.fromkeys(selected_videos))


def run_inference_pipeline(
    video_path: Path,
    db_path: Path = AnalysisPaths.INFERENCE_DB,
    metadata_csv: Path = AnalysisPaths.METADATA_CSV,
    frame_step: int = VideoSpecs.FRAME_STEP_INTERVAL,
    models: Optional[List[str]] = None,
    force: bool = False,
    auto_download_weights: bool = True,
) -> bool:
    """
    Execute full multi-modal detection and diarization ingestion pipeline.

    Parameters
    ----------
    video_path : Path
        Path to a video file, folder, or text file listing videos.
    db_path : Path, default=AnalysisPaths.INFERENCE_DB
        SQLite database file for persisting inference outputs.
    metadata_csv : Path, default=AnalysisPaths.METADATA_CSV
        Metadata CSV containing recording age and child IDs.
    frame_step : int, default=VideoSpecs.FRAME_STEP_INTERVAL
        Frame sampling step (e.g., 10 for 3 Hz).
    models : Optional[List[str]], default=None
        Models to execute: subset of {'person', 'face', 'book', 'speech'} or ['all'].
    force : bool, default=False
        If True, reprocesses videos even if logged in progress logs.
    auto_download_weights : bool, default=True
        Automatically downloads missing Hugging Face model checkpoints.

    Returns
    -------
    bool
        True if all stages completed without fatal error, False otherwise.
    """
    try:
        # 1. Ensure Model Weights Exist
        if auto_download_weights:
            download_weights()

        # 2. Initialize Database Schema and Metadata
        setup_inference_schema(db_path=db_path)
        populate_video_metadata(
            metadata_csv=metadata_csv,
            db_path=db_path,
            raw_videos_dir=AnalysisPaths.VIDEOS_DIR,
        )

        # 3. Resolve Target Video List
        selected_videos = resolve_video_list(video_path)
        if not selected_videos:
            logger.error("No valid video recordings identified. Halting execution.")
            return False

        logger.info("Found %d video recording(s) to process.", len(selected_videos))

        # 4. Resolve Model Execution Selection
        if not models or "all" in models:
            models_to_run = ["person", "face", "book", "speech"]
        else:
            models_to_run = [m for m in models]

        # 5. Execute Extraction Pipelines
        if "person" in models_to_run:
            logger.info("Executing Person Detection (Confidence Floor: %.3f)...", DetectionConfig.PERSON_CONFIDENCE_THRESHOLD)
            run_detection_pipeline(
                task="person",
                video_list=selected_videos,
                frame_step=frame_step,
                force=force,
            )

        if "face" in models_to_run:
            logger.info("Executing Face Detection & Proximity (Confidence Floor: %.3f)...", DetectionConfig.FACE_CONFIDENCE_THRESHOLD)
            run_detection_pipeline(
                task="face",
                video_list=selected_videos,
                frame_step=frame_step,
                force=force,
            )

        if "book" in models_to_run:
            logger.info("Executing Book / Object Filter Detection...")
            run_detection_pipeline(
                task="book",
                video_list=selected_videos,
                frame_step=frame_step,
                force=force,
            )

        if "speech" in models_to_run:
            logger.info("Executing Voice Type Classifier (VTC) Audio Ingestion...")
            run_audio_pipeline(
                video_list=selected_videos,
                force=force,
            )

        logger.info("Feature extraction pipeline completed successfully.")
        return True

    except Exception as e:
        logger.exception("Inference pipeline failed: %s", e)
        return False


def main() -> None:
    """CLI entrypoint for batch extraction orchestration."""
    parser = argparse.ArgumentParser(
        description="Unified Feature Extraction Pipeline Orchestrator"
    )
    parser.add_argument(
        "--video_path",
        type=Path,
        default=AnalysisPaths.VIDEOS_LIST,
        help="Path to a video file, folder of videos, or text list of video IDs",
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default=AnalysisPaths.INFERENCE_DB,
        help="Target SQLite database path",
    )
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=AnalysisPaths.METADATA_CSV,
        help="Path to metadata.csv",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=VideoSpecs.FRAME_STEP_INTERVAL,
        help="Frame step size for processing (default: 10)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["person", "face", "book", "speech", "all"],
        default=["all"],
        help="Select which extractors to execute",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already completed video sessions",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip automatic model weight verification and download",
    )
    args = parser.parse_args()

    success = run_inference_pipeline(
        video_path=args.video_path,
        db_path=args.db_path,
        metadata_csv=args.metadata_csv,
        frame_step=args.frame_step,
        models=args.models,
        force=args.force,
        auto_download_weights=not args.skip_download,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()