"""
src/feature_extraction/run_audio_rttm_ingestion.py

Parses Voice Type Classifier (VTC) RTTM diarization files and maps continuous
speech intervals to frame-level timestamps, dynamically writing records for all
configured AUDIO_CLASSES into the SQLite inference database.
"""

import argparse
import logging
from pathlib import Path
import shutil
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import DetectionConfig, VideoSpecs
from src.constants import AnalysisPaths, ModelAssets, PipelineLogs
from src.feature_extraction.utils import (
    get_video_id,
    load_processed_videos,
    save_processed_video,
)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_rttm_file(rttm_path: Path) -> Dict[str, List[Dict[str, float]]]:
    """
    Parse a standard NIST RTTM diarization file into per-recording speech intervals.

    Parameters
    ----------
    rttm_path : Path
        Path to the VTC output file.

    Returns
    -------
    Dict[str, List[Dict[str, float]]]
        Dictionary mapping recording file IDs (stems) to a list of segments:
        [{'start_time': float, 'duration': float, 'label': str}, ...]
    """
    snippets_by_file: Dict[str, List[Dict[str, float]]] = {}

    if not rttm_path.exists():
        logger.error("RTTM diarization file not found at: %s", rttm_path)
        return snippets_by_file

    valid_classes = set(DetectionConfig.AUDIO_CLASSES)

    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                raw_file_id = parts[1]
                file_stem = Path(raw_file_id).stem

                try:
                    start = float(parts[3])
                    duration = float(parts[4])
                except ValueError:
                    logger.warning("Skipping malformed line: %s", line.strip())
                    continue

                label = parts[7]
                if label not in valid_classes:
                    continue

                if duration <= 0.0:
                    continue

                snippets_by_file.setdefault(file_stem, []).append({
                    "start_time": start,
                    "duration": duration,
                    "label": label,
                })

    logger.info("Parsed %d recording session(s) from RTTM.", len(snippets_by_file))
    return snippets_by_file


def map_snippets_to_frames(
    snippets: List[Dict[str, float]],
    video_id: int,
    fps: int = VideoSpecs.FPS,
    frame_step: int = VideoSpecs.FRAME_STEP_INTERVAL,
    model_id: int = DetectionConfig.AUDIO_MODEL_ID,
    audio_classes: Optional[List[str]] = None,
) -> List[Tuple]:
    """
    Discretize continuous time intervals into frame-indexed database rows.

    Parameters
    ----------
    snippets : List[Dict[str, float]]
        List of speech intervals with 'start_time', 'duration', and 'label'.
    video_id : int
        Database foreign key identifier for the recording session.
    fps : int, default=30
        Video frames per second.
    frame_step : int, default=10
        Downsampling rate (every 10th frame = 3 Hz).
    model_id : int, default=DetectionConfig.AUDIO_MODEL_ID
        Classifier identifier in the database.
    audio_classes : Optional[List[str]], default=None
        Classes to extract. Defaults to DetectionConfig.AUDIO_CLASSES.

    Returns
    -------
    List[Tuple]
        Batch of tuples ready for executemany insertion.
    """
    if audio_classes is None:
        audio_classes = DetectionConfig.AUDIO_CLASSES

    records_batch = []

    for snippet in snippets:
        start_sec = snippet["start_time"]
        duration = snippet["duration"]
        active_label = snippet["label"]
        end_sec = start_sec + duration

        # Compute (has_<class>, <class>_confidence) pairs dynamically
        class_values: List[float] = []
        for cls in audio_classes:
            is_active = 1 if cls == active_label else 0
            conf = 1.0 if is_active else 0.0
            class_values.extend([is_active, conf])

        # Convert continuous seconds to frame numbers
        start_frame = int(np.floor(start_sec * fps))
        end_frame = int(np.ceil(end_sec * fps))

        # Align to nearest frame_step multiple
        first_frame = ((start_frame + frame_step - 1) // frame_step) * frame_step

        for frame_number in range(first_frame, end_frame, frame_step):
            row = (video_id, frame_number, model_id, *class_values)
            records_batch.append(row)

    return records_batch


def insert_audio_records(
    cursor: sqlite3.Cursor,
    records: List[Tuple],
    audio_classes: Optional[List[str]] = None,
) -> None:
    """
    Dynamically batch insert audio classification records into SQLite.

    Parameters
    ----------
    cursor : sqlite3.Cursor
        Active database cursor.
    records : List[Tuple]
        List of pre-formatted frame-level records.
    audio_classes : Optional[List[str]], default=None
        List of classes determining column names. Defaults to DetectionConfig.AUDIO_CLASSES.
    """
    if not records:
        return

    if audio_classes is None:
        audio_classes = DetectionConfig.AUDIO_CLASSES

    # Construct column names dynamically: has_<cls>, <cls>_confidence_score
    columns = ["video_id", "frame_number", "model_id"]
    for cls in audio_classes:
        clean_name = cls.lower().replace("kcds", "cds")  # Maps KCDS to cds if using standard DB schema
        columns.append(f"has_{clean_name}")
        columns.append(f"{clean_name}_confidence_score")

    col_names = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    query = f"INSERT INTO AudioClassifications ({col_names}) VALUES ({placeholders})"

    cursor.executemany(query, records)


def run_audio_pipeline(
    video_list: List[str],
    rttm_file: Optional[Path] = None,
    force: bool = False,
) -> None:
    """
    Orchestrate full RTTM diarization ingestion across specified videos.

    Parameters
    ----------
    video_list : List[str]
        List of target recording stems to process.
    rttm_file : Optional[Path], default=None
        Path to input RTTM file. Defaults to ModelAssets.VTC_RTTM_FILE.
    force : bool, default=False
        If True, reprocesses videos already listed in the log file.
    """
    rttm_path = rttm_file or ModelAssets.VTC_RTTM_FILE
    log_path = PipelineLogs.SPEECH_LOG

    log_path.parent.mkdir(parents=True, exist_ok=True)

    if force and log_path.exists():
        logger.info("Force flag enabled: backing up previous log.")
        shutil.copy2(log_path, log_path.with_suffix(".txt.backup"))
        log_path.unlink()

    processed = load_processed_videos(log_path)
    pending_videos = [v for v in video_list if v not in processed]

    if not pending_videos:
        logger.info("All %d requested videos have already been processed for audio.", len(video_list))
        return

    logger.info("Loading RTTM annotations from: %s", rttm_path)
    all_snippets = parse_rttm_file(rttm_path)

    if not all_snippets:
        logger.warning("No speech snippets extracted. Aborting audio ingestion.")
        return

    selected_stems = set(pending_videos)
    processed_count = 0

    with sqlite3.connect(AnalysisPaths.INFERENCE_DB) as conn:
        cursor = conn.cursor()

        for base_stem in selected_stems:
            if base_stem not in all_snippets:
                logger.warning("No RTTM entries found for video stem '%s'. Skipping.", base_stem)
                continue

            video_id = get_video_id(base_stem, cursor)
            if video_id is None:
                logger.warning("Video stem '%s' not registered in Videos table. Skipping.", base_stem)
                continue

            snippets = all_snippets[base_stem]
            records = map_snippets_to_frames(snippets=snippets, video_id=video_id)

            try:
                insert_audio_records(cursor, records)
                conn.commit()
                save_processed_video(log_path, base_stem)
                logger.info(
                    "Inserted %d frame records across %d snippets for: %s (id=%d)",
                    len(records), len(snippets), base_stem, video_id,
                )
                processed_count += 1
            except Exception as e:
                logger.error("Error inserting audio records for %s: %s", base_stem, e)
                conn.rollback()

    logger.info("Audio ingestion completed successfully for %d recordings.", processed_count)


def main() -> None:
    """CLI entrypoint for batch audio ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest VTC RTTM audio diarization into inference.db"
    )
    parser.add_argument(
        "--video_list",
        nargs="+",
        required=True,
        help="Space-separated list of video stems to process",
    )
    parser.add_argument(
        "--rttm_path",
        type=str,
        default=None,
        help="Path to the all.rttm file (defaults to data/diarization/all.rttm)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already logged videos",
    )
    args = parser.parse_args()

    custom_rttm = Path(args.rttm_path) if args.rttm_path else None
    run_audio_pipeline(
        video_list=args.video_list,
        rttm_file=custom_rttm,
        force=args.force,
    )


if __name__ == "__main__":
    main()