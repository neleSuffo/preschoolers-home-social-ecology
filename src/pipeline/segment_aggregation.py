"""
src/pipeline/segment_aggregation.py

Segment-Level Social State Aggregation and Timeline Refinement.

This module aggregates discrete, frame-level provisional classifications into
continuous, robust social interaction segments. The aggregation proceeds in
two sequential stages:

1. Change Point Detection (CPD) & Segment-Level Assignment:
    Partitions the 1D discrete time series of visual presence scores
    (y_t = S^{presence}_{v,t}) into candidate segment slices using the Pruned
    Exact Linear Time (PELT) algorithm with an L2 cost function. Each slice is
    then assigned a consensus state based on frame densities (d_int, d_avl).

2. Three-Stage Temporal Post-Processing:
    - Initial Segment Merging: Combines identical adjacent states separated by small gaps.
    - Gap Stretching and Default Gap Filling: Extends boundaries for small gaps
        or inserts baseline labels ('Alone' / 'Not Interacting') for large gaps.
    - Secondary Segment Merging: Re-consolidates adjacent segments that came into
        contact during boundary stretching.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import ruptures as rpt

from src.config import AnalysisConfig, VideoSpecs
from src.constants import AnalysisPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FPS: int = VideoSpecs.FPS


def apply_cpd_smoothing(
    frame_data: pd.DataFrame,
    social_state_mode: str,
) -> pd.DataFrame:
    """
    Partition frame timelines via PELT and assign segment-level consensus states.

    Applies the Pruned Exact Linear Time (PELT) algorithm with an L2 cost
    function to partition the rolling visual presence signal into homogeneous
    candidate slices. Within each slice, the consensus social state is assigned
    using hierarchical frame-density voting:
    - Interacting: d_int >= CPD_INTERACTING_THRESHOLD
    - Available:   (d_int + d_avl) >= CPD_TOTAL_PRESENCE_FLOOR
    - Alone:       Otherwise (or Not Interacting in binary mode)

    Parameters
    ----------
    frame_data : pd.DataFrame
        DataFrame containing frame-level multimodal flags, rolling presence
        scores ('presence_score'), and provisional states ('social_state').
    social_state_mode : str
        Classification granularity: 'tertiary' (3 classes) or 'binary' (2 classes).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'social_state' updated to consensus states.
    """
    smoothed_results = []

    for _, video_df in frame_data.groupby("video_id"):
        video_df = video_df.sort_values("frame_number").copy()

        # Extract 1D continuous visual presence signal: y = S^{presence}_{v,t}
        signal = video_df["presence_score"].values.reshape(-1, 1)

        # Fit PELT with L2 cost function
        algo = rpt.Pelt(model="l2").fit(signal)
        breakpoints = algo.predict(pen=AnalysisConfig.CPD_PENALTY)

        start_idx = 0
        for end_idx in breakpoints:
            segment_slice = video_df.iloc[start_idx:end_idx]
            if not segment_slice.empty:
                counts = segment_slice["social_state"].value_counts(normalize=True)
                d_int = counts.get(1, 0.0)  # Density voting for Interacting
                d_avl = counts.get(2, 0.0)  # Density voting for Available

                if d_int >= AnalysisConfig.CPD_INTERACTING_THRESHOLD:
                    state = 1  # Interacting
                elif (d_int + d_avl) >= AnalysisConfig.CPD_TOTAL_PRESENCE_FLOOR:
                    state = 2  # Available (or Not Interacting in binary mode)
                else:
                    state = 3 if social_state_mode == "tertiary" else 2  # Alone / Baseline

                video_df.iloc[start_idx:end_idx, video_df.columns.get_loc("social_state")] = state

            start_idx = end_idx

        smoothed_results.append(video_df)

    return pd.concat(smoothed_results, ignore_index=True)


def create_segments_for_video(video_id: int, video_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert contiguous runs of identical frame states into timestamped segments.
    
    Parameters
    ----------
    video_id : int
        Unique identifier for the video.
    video_df : pd.DataFrame
        DataFrame containing frame-level social states for a single video.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries representing the created segments.
    """
    video_df = video_df.sort_values("frame_number").reset_index(drop=True)
    if len(video_df) == 0:
        return []

    video_name = video_df["video_name"].iloc[0]
    states = video_df["social_state"].values
    frame_nums = video_df["frame_number"].values
    segments = []

    curr_state = states[0]
    start_fr = frame_nums[0]

    for i in range(1, len(states)):
        if states[i] != curr_state:
            end_fr = frame_nums[i - 1]
            dur = (end_fr - start_fr) / FPS
            segments.append({
                "video_id": video_id,
                "video_name": video_name,
                "social_state": curr_state,
                "segment_start": start_fr,
                "segment_end": end_fr,
                "start_time_sec": start_fr / FPS,
                "end_time_sec": end_fr / FPS,
                "duration_sec": dur,
            })
            start_fr = frame_nums[i]
            curr_state = states[i]

    end_fr = frame_nums[-1]
    dur = (end_fr - start_fr) / FPS
    segments.append({
        "video_id": video_id,
        "video_name": video_name,
        "social_state": curr_state,
        "segment_start": start_fr,
        "segment_end": end_fr,
        "start_time_sec": start_fr / FPS,
        "end_time_sec": end_fr / FPS,
        "duration_sec": dur,
    })

    return segments


def merge_same_segments(segments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge adjacent segments sharing identical social states separated by brief gaps.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame containing segment-level social states with start/end timestamps.

    Returns
    -------
    pd.DataFrame
        DataFrame with merged segments.
    """
    merged_segments = []

    for _, video_segments in segments_df.groupby("video_id"):
        video_segments = video_segments.sort_values("start_time_sec").reset_index(drop=True)
        if len(video_segments) == 0:
            continue

        current_segment = video_segments.iloc[0].copy()
        for i in range(1, len(video_segments)):
            next_segment = video_segments.iloc[i]
            gap_duration = next_segment["start_time_sec"] - current_segment["end_time_sec"]

            if (
                current_segment["social_state"] == next_segment["social_state"]
                and gap_duration <= AnalysisConfig.SAME_SEGMENT_MERGE_THRESHOLD
            ):
                current_segment["segment_end"] = next_segment["segment_end"]
                current_segment["end_time_sec"] = next_segment["end_time_sec"]
                current_segment["duration_sec"] = (
                    current_segment["end_time_sec"] - current_segment["start_time_sec"]
                )
            else:
                merged_segments.append(current_segment.to_dict())
                current_segment = next_segment.copy()

        merged_segments.append(current_segment.to_dict())

    return pd.DataFrame(merged_segments) if merged_segments else segments_df


def fill_gaps_with_default(segments_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Reconcile timeline gaps via boundary stretching or baseline state insertion.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame containing segment-level social states with start/end timestamps. 
    
    Returns
    -------
    pd.DataFrame
        DataFrame with gaps filled.
    Dict[str, Any]
        Statistics by video.
    """
    default_type = AnalysisConfig.GAP_DEFAULT_LABEL
    filled_segments = []
    stats_by_video = {}
    total_default_segments = 0
    total_default_seconds = 0.0

    for video_id, video_df in segments_df.groupby("video_id"):
        v_segs = video_df.sort_values("start_time_sec").to_dict("records")
        video_name = v_segs[0]["video_name"] if v_segs else "unknown"
        video_default_segments = 0
        video_default_seconds = 0.0

        for i in range(len(v_segs)):
            filled_segments.append(v_segs[i])
            if i < len(v_segs) - 1:
                gap = v_segs[i + 1]["start_time_sec"] - v_segs[i]["end_time_sec"]

                # Case 1: Short gap -> stretch previous boundary forward
                if 0 < gap <= AnalysisConfig.GAP_STRETCH_THRESHOLD:
                    filled_segments[-1]["end_time_sec"] = v_segs[i + 1]["start_time_sec"]
                    filled_segments[-1]["segment_end"] = v_segs[i + 1]["segment_start"] - 1
                    filled_segments[-1]["duration_sec"] = (
                        filled_segments[-1]["end_time_sec"] - filled_segments[-1]["start_time_sec"]
                    )

                # Case 2: Long gap -> insert baseline default state segment
                elif gap > AnalysisConfig.GAP_STRETCH_THRESHOLD:
                    default_duration = v_segs[i + 1]["start_time_sec"] - v_segs[i]["end_time_sec"]
                    filled_segments.append({
                        "video_id": video_id,
                        "video_name": v_segs[i]["video_name"],
                        "social_state": default_type,
                        "start_time_sec": v_segs[i]["end_time_sec"],
                        "end_time_sec": v_segs[i + 1]["start_time_sec"],
                        "segment_start": v_segs[i]["segment_end"] + 1,
                        "segment_end": v_segs[i + 1]["segment_start"] - 1,
                        "duration_sec": default_duration,
                    })
                    video_default_segments += 1
                    video_default_seconds += default_duration

        stats_by_video[video_id] = {
            "video_name": video_name,
            "default_segments_inserted": video_default_segments,
            "default_seconds_filled": video_default_seconds,
            "default_minutes_filled": video_default_seconds / 60.0,
        }
        total_default_segments += video_default_segments
        total_default_seconds += video_default_seconds

    fill_stats = {
        "default_label": default_type,
        "total_default_segments_inserted": total_default_segments,
        "total_default_seconds_filled": total_default_seconds,
        "total_default_minutes_filled": total_default_seconds / 60.0,
        "per_video": stats_by_video,
    }

    return pd.DataFrame(filled_segments), fill_stats


def print_default_fill_summary(fill_stats: Dict[str, Any]) -> None:
    """
    Log diagnostic summary of baseline gap filling operations.
    
    Parameters
    ----------
    fill_stats : Dict[str, Any]
        Dictionary containing statistics about gap filling operations.
    """
    logger.info(
        "Gap-fill summary: Label='%s' | Segments inserted=%d | Duration=%.2fs (%.2f min)",
        fill_stats["default_label"],
        fill_stats["total_default_segments_inserted"],
        fill_stats["total_default_seconds_filled"],
        fill_stats["total_default_minutes_filled"],
    )


def print_segment_summary(segments_df: pd.DataFrame, social_state_mode: str) -> None:
    """
    Log summary metrics (counts, durations, proportions) by social state.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame containing segment-level social states with start/end timestamps.
    social_state_mode : str
        Classification granularity: 'tertiary' (3 classes) or 'binary' (2 classes).
    """
    if len(segments_df) > 0:
        total_min = round(segments_df["duration_sec"].sum() / 60.0, 2)
        logger.info("Final segment summary: %.2f minutes total analyzed.", total_min)
        target_classes = (
            ["Interacting", "Not_Interacting"]
            if social_state_mode == "binary"
            else ["Interacting", "Available", "Alone"]
        )
        for itype in target_classes:
            df_sub = segments_df[segments_df["social_state"] == itype]
            mins = round(df_sub["duration_sec"].sum() / 60.0, 2)
            perc = (mins / total_min * 100.0) if total_min > 0 else 0.0
            logger.info("   %-15s: %4d segments (%6.2f min - %5.1f%%)", itype, len(df_sub), mins, perc)
    else:
        logger.warning("No segments found in DataFrame.")


def main(
    output_file_path: Path,
    frame_data_path: Path,
    social_state_mode: str = "tertiary",
    hyperparameter_tuning: bool = False,
    metadata_csv_path: Path = AnalysisPaths.METADATA_CSV,
) -> Path:
    """
    Execute segment-level aggregation, CPD smoothing, and boundary post-processing.

    Parameters
    ----------
    output_file_path : Path
        Destination path where interaction_segments.csv will be saved.
    frame_data_path : Path
        Source path to frame_level_social_interactions.csv.
    social_state_mode : str, default='tertiary'
        Granularity: 'tertiary' (3 classes) or 'binary' (2 classes).
    hyperparameter_tuning : bool, default=False
        If True, preserves active tuning configuration in memory.
    metadata_csv_path : Path, default=AnalysisPaths.METADATA_CSV
        Path to metadata CSV for participant age and ID enrichment.

    Returns
    -------
    Path
        Path to the saved interaction segments CSV.
    """
    if not hyperparameter_tuning:
        AnalysisConfig.apply_mode(social_state_mode)

    if not frame_data_path.exists():
        raise FileNotFoundError(f"Frame data file not found at: {frame_data_path}")

    # 1. Load intermediate frame-level classifications
    frame_data = pd.read_csv(frame_data_path)

    # 2. Stage 1: PELT Change Point Detection and Segment State Assignment
    frame_data = apply_cpd_smoothing(frame_data, social_state_mode)

    # 3. Map numeric state IDs (1, 2, 3) to canonical string labels
    frame_data["social_state"] = frame_data["social_state"].map(AnalysisConfig.SOCIAL_STATE_MAPPING)

    # 4. Extract contiguous state blocks into segment records
    all_segments = []
    for video_id, video_df in frame_data.groupby("video_id"):
        all_segments.extend(create_segments_for_video(video_id, video_df))

    if not all_segments:
        logger.warning("No segments generated from frame data. Aborting.")
        return output_file_path

    segments_df = pd.DataFrame(all_segments).sort_values(["video_id", "start_time_sec"]).reset_index(drop=True)

    # 5. Stage 2: Three-Step Temporal Post-Processing
    segments_df = merge_same_segments(segments_df)
    segments_df, fill_stats = fill_gaps_with_default(segments_df)
    segments_df = merge_same_segments(segments_df)

    print_default_fill_summary(fill_stats)

    final_df = segments_df[segments_df["duration_sec"] > 0].copy()
    print_segment_summary(final_df, social_state_mode)

    # 6. Merge participant metadata (Child ID, Age at Recording)
    if metadata_csv_path.exists():
        try:
            meta_df = pd.read_csv(metadata_csv_path, sep=";", encoding="utf-8")
            # Normalize age column naming
            if "age" in meta_df.columns and "age_at_recording" not in meta_df.columns:
                meta_df["age_at_recording"] = meta_df["age"].astype(str).str.replace(",", ".", regex=False).astype(float)
            elif "age_at_recording" in meta_df.columns:
                meta_df["age_at_recording"] = meta_df["age_at_recording"].astype(str).str.replace(",", ".", regex=False).astype(float)

            keep_cols = [c for c in ["video_name", "age_at_recording", "child_id"] if c in meta_df.columns]
            final_df = final_df.merge(meta_df[keep_cols], on="video_name", how="left")
        except Exception as e:
            logger.warning("Could not merge metadata CSV: %s", e)
    else:
        logger.warning("Metadata file not found at %s. Skipping metadata merge.", metadata_csv_path)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file_path, index=False)
    logger.info("Segment-level analysis successfully saved to: %s", output_file_path)
    return output_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment-Level Aggregation and Temporal Smoothing Pipeline"
    )
    parser.add_argument(
        "--output_folder_path",
        type=Path,
        default=None,
        help="Directory containing frame_level_social_interactions.csv and where segments will be saved",
    )
    parser.add_argument(
        "--frame_data_path",
        type=Path,
        default=None,
        help="Explicit path to frame_level_social_interactions.csv",
    )
    parser.add_argument(
        "--output_file_path",
        type=Path,
        default=None,
        help="Explicit path to write interaction_segments.csv",
    )
    parser.add_argument(
        "--social_state_mode",
        choices=["binary", "tertiary"],
        default="tertiary",
        help="Classification mode: 'binary' or 'tertiary'",
    )
    parser.add_argument(
        "--hyperparameter_tuning",
        action="store_true",
        help="Flag indicating execution within automated hyperparameter search",
    )
    args = parser.parse_args()

    # Resolve paths whether given a folder or explicit file targets
    if args.output_folder_path:
        f_in = args.frame_data_path or (args.output_folder_path / AnalysisPaths.FRAME_INTERACTIONS_CSV.name)
        f_out = args.output_file_path or (args.output_folder_path / AnalysisPaths.INTERACTION_SEGMENTS_CSV.name)
    else:
        f_in = args.frame_data_path or AnalysisPaths.FRAME_INTERACTIONS_CSV
        f_out = args.output_file_path or AnalysisPaths.INTERACTION_SEGMENTS_CSV

    main(
        output_file_path=f_out,
        frame_data_path=f_in,
        social_state_mode=args.social_state_mode,
        hyperparameter_tuning=args.hyperparameter_tuning,
    )