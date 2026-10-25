"""
src/research_questions/utils.py

Shared utilities for behavioral research question extraction:
- NIST RTTM speech diarization parsing.
- Inter-speaker interval merging and vocalization duration aggregation.
- Dense second-level array label reconstruction.
- Participant Child ID extraction and cumulative timeline fold boundary calculation.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.constants import ModelAssets


def time_to_seconds(time_str: Union[str, float, int]) -> Optional[float]:
    """
    Convert MM:SS, HH:MM:SS, or numeric strings to float seconds.

    Parameters
    ----------
    time_str : Union[str, float, int]
        Time representation (e.g., '01:30', '01:15:30', or 90.0).

    Returns
    -------
    Optional[float]
        Time in seconds, or None if conversion fails.
    """
    if pd.isna(time_str):
        return None

    try:
        parts = str(time_str).strip().split(":")
        if len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60.0 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600.0 + minutes * 60.0 + seconds
        else:
            return float(time_str)
    except (ValueError, TypeError):
        return None


def extract_child_id(video_name: str) -> Optional[str]:
    """
    Extract the 6-digit child ID from a recording filename or video stem.

    Examples:
        - 'id123456_session01.mp4' -> '123456'
        - 'quantex_ID654321_02' -> '654321'
        - '123456_01.MP4' -> '123456'

    Parameters
    ----------
    video_name : str
        The filename or stem of the video recording.

    Returns
    -------
    Optional[str]
        Extracted 6-digit string ID, or None if not found.
    """
    if pd.isna(video_name):
        return None

    s = str(video_name)
    # 1. Match 'id123456' or 'ID123456'
    match = re.search(r"id(\d{6})", s, re.IGNORECASE)
    if match:
        return match.group(1)

    # 2. Match leading or isolated 6 digits
    match_digits = re.search(r"(?:^|[_/-])(\d{6})(?:$|[_/-])", s)
    if match_digits:
        return match_digits.group(1)

    return None


def create_second_level_labels(
    segments_df: pd.DataFrame,
    video_duration_seconds: int,
) -> np.ndarray:
    """
    Create a second-by-second label array for a video based on discrete segment intervals.

    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame containing segment onset/offset timestamps and state assignments.
    video_duration_seconds : int
        Total recording span in seconds.

    Returns
    -------
    np.ndarray
        Array of shape (video_duration_seconds,) containing string labels or None.
    """
    labels = np.full(video_duration_seconds, None, dtype=object)

    if segments_df.empty:
        return labels

    df_work = segments_df.copy()

    # Ensure time columns exist in seconds
    if "start_time_sec" not in df_work.columns or "end_time_sec" not in df_work.columns:
        if "start_time_min" in df_work.columns and "end_time_min" in df_work.columns:
            df_work["start_time_sec"] = df_work["start_time_min"].apply(time_to_seconds)
            df_work["end_time_sec"] = df_work["end_time_min"].apply(time_to_seconds)
        else:
            return labels

    state_col = "social_state" if "social_state" in df_work.columns else "interaction_type"

    for _, segment in df_work.iterrows():
        try:
            start_sec = int(np.round(float(segment["start_time_sec"])))
        except (ValueError, TypeError):
            start_sec = 0

        try:
            end_sec = int(np.round(float(segment["end_time_sec"])))
            end_sec = min(end_sec, video_duration_seconds - 1)
        except (ValueError, TypeError):
            end_sec = start_sec

        interaction_type = str(segment[state_col]).lower().strip()
        start_sec = max(0, start_sec)

        if start_sec <= end_sec:
            labels[start_sec : end_sec + 1] = interaction_type

    return labels


def parse_rttm(
    file_path: Path = ModelAssets.VTC_RTTM_FILE,
    target_speech_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Parse a NIST RTTM speech diarization file into a structured vocalization table.

    Parameters
    ----------
    file_path : Path, default=ModelAssets.VTC_RTTM_FILE
        Filesystem path to the VTC RTTM output file.
    target_speech_types : Optional[List[str]], default=None
        Speech classification tokens to filter for (e.g. ['KCHI', 'KCDS', 'OHS']).
        If None, all speaker types are included.

    Returns
    -------
    pd.DataFrame
        Table with ['video_name', 'start_time_seconds', 'end_time_seconds',
        'duration_seconds', 'speech_type'].
    """
    data = []
    if not file_path.exists():
        return pd.DataFrame()

    filter_types = set(target_speech_types) if target_speech_types else None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                speech_type = parts[7]
                if filter_types and speech_type not in filter_types:
                    continue

                raw_video_name = parts[1]
                clean_stem = Path(raw_video_name).stem

                try:
                    start_sec = float(parts[3])
                    dur_sec = float(parts[4])
                except (ValueError, IndexError):
                    continue

                if dur_sec <= 0.0:
                    continue

                data.append({
                    "video_name": clean_stem,
                    "start_time_seconds": start_sec,
                    "end_time_seconds": start_sec + dur_sec,
                    "duration_seconds": dur_sec,
                    "speech_type": speech_type,
                })

    return pd.DataFrame(data)


def merge_overlapping_intervals(
    intervals: List[Tuple[float, float]],
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Merge overlapping or adjacent continuous temporal intervals and calculate total duration.

    Parameters
    ----------
    intervals : List[Tuple[float, float]]
        List of (start_sec, end_sec) tuples.

    Returns
    -------
    Tuple[List[Tuple[float, float]], float]
        (merged_intervals, total_duration_seconds)
    """
    if not intervals:
        return [], 0.0

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]

    for current_start, current_end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    total_duration = sum(end - start for start, end in merged)
    return merged, total_duration


def merge_overlapping_vocalizations(vocs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge overlapping vocalizations from the same speaker category within a video.

    Parameters
    ----------
    vocs_df : pd.DataFrame
        DataFrame containing ['video_name', 'speech_type', 'start_time_seconds', 'end_time_seconds'].

    Returns
    -------
    pd.DataFrame
        DataFrame with fused vocalization episodes.
    """
    if vocs_df.empty:
        return vocs_df

    merged_rows = []
    group_cols = ["video_name", "speech_type"]

    for (video_name, speech_type), group in vocs_df.groupby(group_cols):
        sorted_group = group.sort_values("start_time_seconds").to_dict("records")
        current_voc = sorted_group[0].copy()

        for next_voc in sorted_group[1:]:
            if next_voc["start_time_seconds"] <= current_voc["end_time_seconds"]:
                current_voc["end_time_seconds"] = max(
                    current_voc["end_time_seconds"],
                    next_voc["end_time_seconds"],
                )
                current_voc["duration_seconds"] = (
                    current_voc["end_time_seconds"] - current_voc["start_time_seconds"]
                )
            else:
                merged_rows.append(current_voc)
                current_voc = next_voc.copy()

        merged_rows.append(current_voc)

    return pd.DataFrame(merged_rows).reset_index(drop=True)


def get_child_fold_boundaries(
    segments_df: pd.DataFrame,
    num_folds: int = 5,
) -> Dict[Any, List[Tuple[float, float]]]:
    """
    Calculate equal-duration continuous timeline boundaries for each child across cross-validation folds.

    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame containing ['child_id', 'global_start', 'global_end'] or ['child_id', 'duration_sec'].
    num_folds : int, default=5
        Number of equal folds to construct per participant.

    Returns
    -------
    Dict[Any, List[Tuple[float, float]]]
        Dictionary mapping child_id to a list of (start_sec, end_sec) fold boundary intervals.
    """
    boundaries: Dict[Any, List[Tuple[float, float]]] = {}

    for child_id, group in segments_df.groupby("child_id"):
        if "global_start" in group.columns and "global_end" in group.columns:
            total_dur = group["global_end"].max() - group["global_start"].min()
            start_offset = group["global_start"].min()
        else:
            total_dur = group["duration_sec"].sum()
            start_offset = 0.0

        fold_size = total_dur / num_folds
        points = [start_offset + (i * fold_size) for i in range(num_folds + 1)]
        boundaries[child_id] = [(points[i], points[i + 1]) for i in range(num_folds)]

    return boundaries