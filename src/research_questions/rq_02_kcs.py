"""
src/research_questions/rq_02_kcs.py

Research Question 2: Child Language Production Analysis.

Quantifies Key Child Speech (KCS / KCHI) vocalization rates across reconstructed
social states (Interacting, Available, Alone) by cross-referencing temporal segment
boundaries with NIST RTTM speech diarization timestamps.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.constants import AnalysisPaths, BasePaths
from src.research_questions.utils import (
    get_child_fold_boundaries,
    merge_overlapping_intervals,
    parse_rttm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_kcs_analysis(
    output_folder: Optional[Path] = None,
    use_folds: bool = False,
    use_ground_truth: bool = False,
) -> None:
    """
    Generate segment-level and child-level summaries of Key Child Speech (KCS).

    Parameters
    ----------
    output_folder : Optional[Path], default=None
        Target folder for exporting tabular CSV results.
    use_folds : bool, default=False
        If True, computes 5-fold cross-validation metrics per child.
    use_ground_truth : bool, default=False
        If True, matches vocalizations against ground_truth_segments.csv.
    """
    mode_str = "per-fold metrics" if use_folds else "overall recording metrics"
    benchmark_str = "Ground Truth Validation" if use_ground_truth else "Standard Pipeline Predictions"
    logger.info("Executing RQ 02: Child Language Production (KCS) [%s | %s]", mode_str, benchmark_str)

    # 1. Load Segments
    if use_ground_truth:
        segments_path = AnalysisPaths.GT_SEGMENTS_CSV
    else:
        segments_path = (output_folder / AnalysisPaths.INTERACTION_SEGMENTS_CSV.name) if output_folder else AnalysisPaths.INTERACTION_SEGMENTS_CSV

    if not segments_path.exists():
        logger.error("Segments file not found: %s", segments_path)
        return

    segments_df = pd.read_csv(segments_path)
    
    # 2. Extract Key Child (KCHI) vocalizations from RTTM
    kchi_vocalizations = parse_rttm(target_speech_types=["KCHI"])
    if kchi_vocalizations.empty:
        logger.warning("No KCHI vocalizations identified in RTTM diarization files.")

    # 3. Create Timeline Offsets across multiple videos per child
    video_stats = (
        segments_df.groupby(["child_id", "video_name"])["end_time_sec"]
        .max()
        .reset_index(name="video_span_sec")
    )
    video_stats = video_stats.sort_values(["child_id", "video_name"])
    video_stats["offset_raw"] = video_stats.groupby("child_id")["video_span_sec"].shift(1).fillna(0)
    video_stats["offset"] = video_stats.groupby("child_id")["offset_raw"].transform("cumsum")

    segments_df = segments_df.merge(
        video_stats[["child_id", "video_name", "offset"]],
        on=["child_id", "video_name"],
    )

    segments_df["global_start"] = segments_df["start_time_sec"] + segments_df["offset"]
    segments_df["global_end"] = segments_df["end_time_sec"] + segments_df["offset"]

    # 4. Partition by Fold
    if use_folds:
        fold_map = get_child_fold_boundaries(segments_df)
    else:
        fold_map = {}
        for child_id in segments_df["child_id"].unique():
            child_data = segments_df[segments_df["child_id"] == child_id]
            fold_map[child_id] = [(child_data["global_start"].min(), child_data["global_end"].max())]

    final_rows = []
    for child_id, folds in fold_map.items():
        child_segs = segments_df[segments_df["child_id"] == child_id]

        for fold_idx, (f_start, f_end) in enumerate(folds, 1):
            for _, seg in child_segs.iterrows():
                overlap_start = max(seg["global_start"], f_start)
                overlap_end = min(seg["global_end"], f_end)

                if overlap_start < overlap_end:
                    current_duration = overlap_end - overlap_start
                    local_start = overlap_start - seg["offset"]
                    local_end = overlap_end - seg["offset"]

                    group = kchi_vocalizations[
                        (kchi_vocalizations["video_name"] == seg["video_name"])
                        & (kchi_vocalizations["start_time_seconds"] < local_end)
                        & (kchi_vocalizations["end_time_seconds"] > local_start)
                    ].copy()

                    if not group.empty:
                        group["clipped_start"] = np.maximum(group["start_time_seconds"], local_start)
                        group["clipped_end"] = np.minimum(group["end_time_seconds"], local_end)
                        intervals = list(zip(group["clipped_start"], group["clipped_end"]))
                        _, total_speech_seconds = merge_overlapping_intervals(intervals)
                    else:
                        total_speech_seconds = 0.0

                    final_rows.append({
                        "child_id": seg["child_id"],
                        "fold": fold_idx,
                        "video_name": seg["video_name"],
                        "age_at_recording": seg.get("age_at_recording", np.nan),
                        "social_state": seg["social_state"],
                        "segment_start_time": seg["start_time_sec"],
                        "segment_end_time": seg["end_time_sec"],
                        "total_speech_seconds": total_speech_seconds,
                        "total_segment_duration": current_duration,
                        "segment_duration_minutes": current_duration / 60.0,
                    })

    final_df = pd.DataFrame(final_rows)
    if final_df.empty:
        logger.warning("No segment records generated for KCS.")
        return

    final_df["speech_activity_percent"] = (
        final_df["total_speech_seconds"] / final_df["total_segment_duration"]
    ).fillna(0.0)
    final_df["kchi_speech_minutes"] = final_df["total_speech_seconds"] / 60.0

    dest_folder = output_folder or AnalysisPaths.OUTPUT_FOLDER
    dest_folder.mkdir(parents=True, exist_ok=True)
    suffix = "_gt" if use_ground_truth else ""

    out_kcs = dest_folder / f"{AnalysisPaths.KCS_SUMMARY_CSV.stem}{suffix}.csv"
    final_df.to_csv(out_kcs, index=False)
    logger.info("Saved segment KCS summary to: %s", out_kcs)

    # 5. Child-Level Aggregation
    group_cols = ["child_id", "fold"] if use_folds else ["child_id"]
    child_summary = final_df.groupby(group_cols).agg({
        "total_speech_seconds": "sum",
        "total_segment_duration": "sum",
        "age_at_recording": "min",
    }).reset_index()

    child_summary["speech_activity_percent"] = (
        child_summary["total_speech_seconds"] / child_summary["total_segment_duration"]
    ).fillna(0.0)
    child_summary["total_speech_minutes"] = child_summary["total_speech_seconds"] / 60.0
    child_summary["total_recording_minutes"] = child_summary["total_segment_duration"] / 60.0

    out_gkcs = dest_folder / f"{AnalysisPaths.GLOBAL_KCS_SUMMARY_CSV.stem}{suffix}.csv"
    child_summary.to_csv(out_gkcs, index=False)
    logger.info("Saved child-level KCS summary to: %s", out_gkcs)


def main() -> None:
    """CLI entrypoint for RQ 02 Child Language Production Analysis."""
    parser = argparse.ArgumentParser(description="Generate KCS (Key Child Speech) analysis summaries")
    parser.add_argument("--output_folder", type=Path, default=None, help="Target folder for output CSVs")
    parser.add_argument("--use_folds", action="store_true", default=False, help="Compute per-fold CV metrics")
    parser.add_argument("--use_gt", action="store_true", default=False, help="Evaluate on ground truth segments")
    args = parser.parse_args()

    run_kcs_analysis(
        output_folder=args.output_folder,
        use_folds=args.use_folds,
        use_ground_truth=args.use_gt,
    )


if __name__ == "__main__":
    main()