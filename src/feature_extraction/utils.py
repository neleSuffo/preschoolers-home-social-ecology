"""
src/feature_extraction/utils.py

Shared utility functions for video metadata extraction, frame retrieval,
frame number parsing, and processing log management across feature pipelines.
"""

import logging
from pathlib import Path
import re
import sqlite3
from typing import List, Optional, Set

from src.config import VideoSpecs

logger = logging.getLogger(__name__)


def get_video_id(video_name: str, cursor: sqlite3.Cursor) -> Optional[int]:
    """
    Retrieve the unique database primary key for a given video stem.

    Parameters
    ----------
    video_name : str
        Base identifier of the recording session (without file extension).
    cursor : sqlite3.Cursor
        Active database cursor connected to the SQLite inference database.

    Returns
    -------
    Optional[int]
        The integer video_id if present in the Videos table, otherwise None.
    """
    cursor.execute("SELECT video_id FROM Videos WHERE video_name = ?", (video_name,))
    result = cursor.fetchone()
    if result:
        return int(result[0])

    logger.error("Video '%s' not registered in Videos table.", video_name)
    return None


def extract_frame_number(filename: str) -> int:
    """
    Extract the trailing frame number from an image filename using regular expressions.

    Examples:
        - 'frame_000120.jpg' -> 120
        - 'quantex_at_home_id255_0042.png' -> 42
        - '0015.PNG' -> 15

    Parameters
    ----------
    filename : str
        Filename or path stem to parse.

    Returns
    -------
    int
        Extracted integer frame number, or 0 if no digits are present.
    """
    numbers = re.findall(r"\d+", filename)
    if numbers:
        return int(numbers[-1])
    return 0


def get_frame_paths(video_frame_dir: Path) -> List[Path]:
    """
    Retrieve and sort all valid frame image paths within a video directory.

    Frames are sorted chronologically based on their extracted frame numbers.

    Parameters
    ----------
    video_frame_dir : Path
        Filesystem directory containing extracted video stills.

    Returns
    -------
    List[Path]
        Chronologically sorted list of image paths.
    """
    if not video_frame_dir.exists():
        logger.warning("Frame directory does not exist: %s", video_frame_dir)
        return []

    frame_paths = [
        p for p in video_frame_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VideoSpecs.VALID_IMG_EXTENSIONS
    ]

    # Sort robustly by extracted frame number
    frame_paths.sort(key=lambda p: extract_frame_number(p.name))
    return frame_paths


def load_processed_videos(log_file_path: Path) -> Set[str]:
    """
    Read the progress log text file and return the set of completed video stems.

    Parameters
    ----------
    log_file_path : Path
        Filesystem path to the processing checkpoint log.

    Returns
    -------
    Set[str]
        Set of video stem strings that have already been processed.
    """
    if not log_file_path.exists():
        return set()

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        logger.warning("Could not load processing log from %s: %s", log_file_path, e)
        return set()


def save_processed_video(log_file_path: Path, video_name: str) -> None:
    """
    Append a completed video stem to the processing checkpoint log.

    Parameters
    ----------
    log_file_path : Path
        Filesystem path to the processing checkpoint log.
    video_name : str
        Video stem name to record.
    """
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"{video_name}\n")
    except Exception as e:
        logger.error("Failed to append '%s' to log file %s: %s", video_name, log_file_path, e)