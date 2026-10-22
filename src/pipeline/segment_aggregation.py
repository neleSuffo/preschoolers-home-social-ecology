"""
Segment-Level Social State Aggregation and Timeline Refinement (segma)

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
import sys
import logging
import pandas as pd
import ruptures as rpt
from pathlib import Path
from typing import List, Dict, Tuple

# Path configuration for internal framework imports
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

<<<<<<< Updated upstream
from constants import Analysis, DataPaths
from config import AnalysisConfig, DataConfig
=======
from src.constants import Analysis, DataPaths
from src.config import AnalysisConfig, DataConfig
>>>>>>> Stashed changes

# Configure logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Video recording frame rate (used for timestamp and second conversions)
FPS: int = DataConfig.FPS


def apply_cpd_smoothing(frame_data: pd.DataFrame, 
                        social_state_mode: str) -> pd.DataFrame:
    """Partitions frame timelines via PELT and assigns segment-level consensus states.

    Applies the Pruned Exact Linear Time (PELT) algorithm with an L2 cost
    function to partition the rolling visual presence signal (S^{presence}_{v,t})
    into homogeneous candidate slices. Within each slice, the consensus social
    state is assigned using the hierarchical frame-density criteria:
    - Interacting: d_int >= CPD_INTERACTING_THRESHOLD
    - Available:   (d_int + d_avl) >= CPD_TOTAL_PRESENCE_FLOOR
    - Alone:       Otherwise (or Not Interacting in binary mode)

    Parameters
    ----------
    frame_data : pd.DataFrame
        DataFrame containing frame-level multimodal flags, rolling presence
        scores ('presence_score'), and provisional states ('interaction_type').
    social_state_mode : str
        Classification granularity mode:
        - "tertiary": 3 classes (1: Interacting, 2: Available, 3: Alone).
        - "binary": 2 classes (1: Interacting, 2: Not Interacting).

    Returns
    -------
    pd.DataFrame
        DataFrame with the 'interaction_type' column updated to the consensus
        state across all frames within each PELT candidate slice.
    """
    smoothed_results = []

    # Process each recording session independently
    for video_id, video_df in frame_data.groupby('video_id'):
        video_df = video_df.sort_values('frame_number').copy()
        
        # Extract 1D continuous visual presence signal: y = S^{presence}_{v,t}
        signal = video_df['presence_score'].values.reshape(-1, 1)
        
        # Fit PELT with L2 cost function to find optimal changepoint boundaries
        algo = rpt.Pelt(model="l2").fit(signal)
        breakpoints = algo.predict(pen=AnalysisConfig.CPD_PENALTY)

        start_idx = 0
        for end_idx in breakpoints:
            segment_slice = video_df.iloc[start_idx:end_idx]
            if not segment_slice.empty:
                # Calculate normalized frame densities: d_s = (1 / N_slice) * sum(I(Y_{v,t} == s))
                counts = segment_slice['interaction_type'].value_counts(normalize=True)
                d_int = counts.get(1, 0.0)  # Frame density voting for Interacting
                d_avl = counts.get(2, 0.0)  # Frame density voting for Available

                # Hierarchical priority decision logic (Interacting > Available > Alone)
                if d_int >= AnalysisConfig.CPD_INTERACTING_THRESHOLD:
                    state = 1  # Interacting
                elif (d_int + d_avl) >= AnalysisConfig.CPD_TOTAL_PRESENCE_FLOOR:
                    state = 2  # Available (or Not Interacting in binary mode)
                else:
                    state = 3 if social_state_mode == "tertiary" else 2  # Alone / Baseline

                # Overwrite frame-level provisional states with slice consensus state
                video_df.iloc[start_idx:end_idx, video_df.columns.get_loc('interaction_type')] = state

            start_idx = end_idx

        smoothed_results.append(video_df)
        
    return pd.concat(smoothed_results, ignore_index=True)


def create_segments_for_video(video_id: int, 
                              video_df: pd.DataFrame) -> List[Dict]:
    """Converts contiguous runs of identical frame states into timestamped segments.

    Scans the frame sequence chronologically and generates discrete segment
    records whenever the assigned state changes.

    Parameters
    ----------
    video_id : int
        Unique integer identifier for the video recording session.
    video_df : pd.DataFrame
        Frame-level data for a single video, sorted by frame number.

    Returns
    -------
    list of dict
        List of candidate segment dictionaries containing:
        - 'video_id': Recording session ID.
        - 'video_name': Video filename string.
        - 'interaction_type': Assigned social state label.
        - 'segment_start': Start frame index.
        - 'segment_end': End frame index.
        - 'start_time_sec': Segment onset time in seconds.
        - 'end_time_sec': Segment offset time in seconds.
        - 'duration_sec': Segment duration in seconds.
    """
    video_df = video_df.sort_values('frame_number').reset_index(drop=True)
    if len(video_df) == 0:
        return []
    
    video_name = video_df['video_name'].iloc[0] 
    states = video_df['interaction_type'].values
    frame_nums = video_df['frame_number'].values
    segments = []
    
    # Initialize the first continuous episode
    curr_state = states[0]
    start_fr = frame_nums[0]
    
    # Iterate through frames to detect state transitions
    for i in range(1, len(states)):
        if states[i] != curr_state:
            end_fr = frame_nums[i - 1]
            dur = (end_fr - start_fr) / FPS
            segments.append({
                'video_id': video_id, 
                'video_name': video_name,
                'interaction_type': curr_state, 
                'segment_start': start_fr,
                'segment_end': end_fr, 
                'start_time_sec': start_fr / FPS, 
                'end_time_sec': end_fr / FPS, 
                'duration_sec': dur
            })
            start_fr = frame_nums[i]
            curr_state = states[i]
            
    # Finalize the terminal segment
    end_fr = frame_nums[-1]
    dur = (end_fr - start_fr) / FPS
    segments.append({
        'video_id': video_id, 
        'video_name': video_name,
        'interaction_type': curr_state, 
        'segment_start': start_fr,
        'segment_end': end_fr, 
        'start_time_sec': start_fr / FPS, 
        'end_time_sec': end_fr / FPS, 
        'duration_sec': dur
    })
    
    return segments


def merge_same_segments(segments_df: pd.DataFrame) -> pd.DataFrame:
    """Merges adjacent segments sharing the identical social state separated by small gaps.

    Prevents over-fragmentation by consolidating identical states separated by
    brief interruptions (gap <= SAME_SEGMENT_MERGE_THRESHOLD seconds).

    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame of timestamped segments.

    Returns
    -------
    pd.DataFrame
        DataFrame containing merged, consolidated continuous segments.
    """
    merged_segments = []

    for _, video_segments in segments_df.groupby('video_id'):
        video_segments = video_segments.sort_values('start_time_sec').reset_index(drop=True)
        if len(video_segments) == 0:
            continue
        
        current_segment = video_segments.iloc[0].copy()
        for i in range(1, len(video_segments)):
            next_segment = video_segments.iloc[i]
            gap_duration = next_segment['start_time_sec'] - current_segment['end_time_sec']
            
            # Merge condition: identical state AND gap within temporal ceiling
            if (current_segment['interaction_type'] == next_segment['interaction_type'] and 
                gap_duration <= AnalysisConfig.SAME_SEGMENT_MERGE_THRESHOLD):
                # Extend current segment offset to subsume gap and next segment
                current_segment['segment_end'] = next_segment['segment_end']
                current_segment['end_time_sec'] = next_segment['end_time_sec']
                current_segment['duration_sec'] = current_segment['end_time_sec'] - current_segment['start_time_sec']
            else:
                merged_segments.append(current_segment.to_dict())
                current_segment = next_segment.copy()

        merged_segments.append(current_segment.to_dict())
        
    return pd.DataFrame(merged_segments) if merged_segments else segments_df


def fill_gaps_with_default(segments_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Reconciles timeline gaps via boundary stretching or baseline state insertion.

    Handles unassigned intervals between divergent segments:
    - Short gaps (0 < gap <= GAP_STRETCH_THRESHOLD): Resolved by extending
      the boundary of the preceding segment forward.
    - Long gaps (gap > GAP_STRETCH_THRESHOLD): Populated with a new segment
      assigned to the baseline state (e.g., 'Alone' or 'Not Interacting').

    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame of timestamped interaction segments.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        - pd.DataFrame: Continuous segment timeline without unassigned gaps.
        - dict: Summary statistics detailing inserted default baseline segments.
    """
    default_type = AnalysisConfig.GAP_DEFAULT_LABEL
    filled_segments = []
    stats_by_video = {}
    total_default_segments = 0
    total_default_seconds = 0.0

    for video_id, video_df in segments_df.groupby('video_id'):
        v_segs = video_df.sort_values('start_time_sec').to_dict('records')
        video_name = v_segs[0]['video_name'] if v_segs else 'unknown'
        video_default_segments = 0
        video_default_seconds = 0.0

        for i in range(len(v_segs)):
            filled_segments.append(v_segs[i])
            if i < len(v_segs) - 1:
                gap = v_segs[i + 1]['start_time_sec'] - v_segs[i]['end_time_sec']

                # Case 1: Short gap -> stretch previous segment boundary forward
                if 0 < gap <= AnalysisConfig.GAP_STRETCH_THRESHOLD:
                    filled_segments[-1]['end_time_sec'] = v_segs[i + 1]['start_time_sec']
                    filled_segments[-1]['segment_end'] = v_segs[i + 1]['segment_start'] - 1
                    filled_segments[-1]['duration_sec'] = (
                        filled_segments[-1]['end_time_sec'] - filled_segments[-1]['start_time_sec']
                    )

                # Case 2: Long gap -> insert default baseline state segment
                elif gap > AnalysisConfig.GAP_STRETCH_THRESHOLD:
                    default_duration = v_segs[i + 1]['start_time_sec'] - v_segs[i]['end_time_sec']
                    filled_segments.append({
                        'video_id': video_id,
                        'video_name': v_segs[i]['video_name'],
                        'interaction_type': default_type,
                        'start_time_sec': v_segs[i]['end_time_sec'],
                        'end_time_sec': v_segs[i + 1]['start_time_sec'], 
                        'segment_start': v_segs[i]['segment_end'] + 1,
                        'segment_end': v_segs[i + 1]['segment_start'] - 1,
                        'duration_sec': default_duration
                    })

                    video_default_segments += 1
                    video_default_seconds += default_duration

        stats_by_video[video_id] = {
            'video_name': video_name,
            'default_segments_inserted': video_default_segments,
            'default_seconds_filled': video_default_seconds,
            'default_minutes_filled': video_default_seconds / 60.0
        }
        total_default_segments += video_default_segments
        total_default_seconds += video_default_seconds

    fill_stats = {
        'default_label': default_type,
        'total_default_segments_inserted': total_default_segments,
        'total_default_seconds_filled': total_default_seconds,
        'total_default_minutes_filled': total_default_seconds / 60.0,
        'per_video': stats_by_video
    }

    return pd.DataFrame(filled_segments), fill_stats


def print_default_fill_summary(fill_stats: Dict) -> None:
    """Prints diagnostic summary of baseline gap filling operations across recordings.

    Parameters
    ----------
    fill_stats : dict
        Summary dictionary generated by `fill_gaps_with_default`.
    """
    logging.info(
        f"Gap-fill summary: Label='{fill_stats['default_label']}' | "
        f"Segments inserted={fill_stats['total_default_segments_inserted']} | "
        f"Duration={fill_stats['total_default_seconds_filled']:.2f}s "
        f"({fill_stats['total_default_minutes_filled']:.2f} min)"
    )

    for video_id, stats in fill_stats['per_video'].items():
        if stats['default_segments_inserted'] > 0:
            logging.debug(
                f"   - Video {video_id} ({stats['video_name']}): "
                f"segments={stats['default_segments_inserted']}, "
                f"duration={stats['default_seconds_filled']:.2f}s"
            )


def print_segment_summary(segments_df: pd.DataFrame, social_state_mode: str) -> None:
    """Logs aggregated summary metrics (counts, durations, proportions) by social state.

    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame of final segmented episodes.
    social_state_mode : str
        Classification mode ('binary' or 'tertiary').
    """
    if len(segments_df) > 0:
        total_min = round(segments_df['duration_sec'].sum() / 60.0, 2)
        logging.info(f"Final segment summary: {total_min:.2f} minutes total analyzed.")
        target_classes = (
            ['Interacting', 'Not_Interacting'] 
            if social_state_mode == "binary" 
            else ['Interacting', 'Available', 'Alone']
        )
        for itype in target_classes:
            df_sub = segments_df[segments_df['interaction_type'] == itype]
            mins = round(df_sub['duration_sec'].sum() / 60.0, 2)
            perc = (mins / total_min * 100.0) if total_min > 0 else 0.0
            logging.info(f"   {itype:<15}: {len(df_sub):>4} segments ({mins:>6.2f} min - {perc:>5.1f}%)")
    else:
        logging.warning("No segments found in DataFrame.")


def main(output_file_path: Path, 
         frame_data_path: Path, 
         social_state_mode: str = "tertiary",
         hyperparameter_tuning: bool = False) -> None:
    """Executes the segment-level aggregation, CPD smoothing, and boundary post-processing.

    Parameters
    ----------
    output_file_path : Path
        Destination path where the final segment CSV will be written.
    frame_data_path : Path
        Source path to the frame-level classifications CSV.
    social_state_mode : str, default="tertiary"
        Classification granularity: "tertiary" (3 classes) or "binary" (2 classes).
    hyperparameter_tuning : bool, default=False
        If True, preserves tuning configurations passed from the parent cross-validation
        controller and avoids resetting hyperparameters via `apply_mode`.
    """
    if not hyperparameter_tuning:
        AnalysisConfig.apply_mode(social_state_mode)
    
    # 1. Load intermediate frame-level classifications
    frame_data = pd.read_csv(frame_data_path)
    
    # 2. Stage 1: PELT Change Point Detection and Segment State Assignment
    frame_data = apply_cpd_smoothing(frame_data, social_state_mode)
    
    # 3. Map numeric state IDs (1, 2, 3) to canonical string labels
    frame_data['interaction_type'] = frame_data['interaction_type'].map(AnalysisConfig.SOCIAL_STATE_MAPPING)

    # 4. Extract contiguous state blocks into segment records
    all_segments = []
    for video_id, video_df in frame_data.groupby('video_id'):
        all_segments.extend(create_segments_for_video(video_id, video_df))
    
    if not all_segments:
        logging.warning("No segments generated from frame data. Aborting.")
        return

    segments_df = pd.DataFrame(all_segments).sort_values(['video_id', 'start_time_sec']).reset_index(drop=True)

    # 5. Stage 2: Three-Step Temporal Post-Processing
    # Step 1: Initial Segment Merging (fuse small pauses between identical states)
    segments_df = merge_same_segments(segments_df)
    # Step 2: Gap Stretching & Baseline Default Filling
    segments_df, fill_stats = fill_gaps_with_default(segments_df)
    # Step 3: Secondary Segment Merging (fuse stretched segments of the same state)
    segments_df = merge_same_segments(segments_df)

    print_default_fill_summary(fill_stats)

    # Filter out empty or zero-duration segments
    final_df = segments_df[segments_df['duration_sec'] > 0].copy()
    print_segment_summary(final_df, social_state_mode)
    
    # 6. Merge participant metadata (Child ID, Age at Recording)
    if Path(DataPaths.SUBJECTS_CSV_PATH).exists():
        age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH, sep=";", decimal=",")[["video_name", "age_at_recording", "child_id"]]
        final_df = final_df.merge(age_df, on="video_name", how="left")
    else:
        logging.warning(f"Subjects metadata file not found at {DataPaths.SUBJECTS_CSV_PATH}. Skipping metadata merge.")
    
    # Save final structured segment output
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file_path, index=False)
    logging.info(f"✅ Segment-level analysis successfully saved to: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment-Level Aggregation and Temporal Smoothing Pipeline"
    )
    parser.add_argument(
        '--output_folder_path', 
        type=str, 
        required=True,
        help="Path to folder containing the frame-level CSV and where segment results will be saved."
    )
    parser.add_argument(
        '--social_state_mode', 
        type=str, 
        choices=['binary', 'tertiary'], 
        default='tertiary',
        help="Classification mode: 'binary' (Interacting vs Not Interacting) or 'tertiary' (Interacting, Available, Alone)."
    )
    parser.add_argument(
        '--hyperparameter_tuning', 
        action='store_true',
        help="Flag indicating execution within automated hyperparameter search or cross-validation."
    )

    args = parser.parse_args()
    
    output_folder = Path(args.output_folder_path)
    frame_data_path = output_folder / Analysis.FRAME_LEVEL_INTERACTIONS_CSV.name
    output_file_path = output_folder / Analysis.INTERACTION_SEGMENTS_CSV.name

    main(
        output_file_path=output_file_path, 
        frame_data_path=frame_data_path, 
        social_state_mode=args.social_state_mode, 
        hyperparameter_tuning=args.hyperparameter_tuning
    )