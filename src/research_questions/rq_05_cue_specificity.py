"""
src/research_questions/rq_05_interaction_composition.py

Research Question 5: Multimodal Interaction Composition & Feature Synthesis.

Synthesizes frame-level detection features and active rule triggers with continuous
segment-level social ecology states (Interacting, Available, Alone).
Constructs dense indicator flags for subsequent Bayesian multilevel modeling.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import VideoSpecs
from src.constants import AnalysisPaths, BasePaths
from src.research_questions.utils import extract_child_id, get_child_fold_boundaries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def add_interaction_columns(
    frames_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    use_ground_truth: bool = False,
    social_state_mode: str = "tertiary",
) -> pd.DataFrame:
    """
    Map segment-level social states onto sampled video frames as boolean indicator columns.

    Parameters
    ----------
    frames_df : pd.DataFrame
        Frame data containing ['video_name', 'frame_number', 'proximity', ...].
    segments_df : pd.DataFrame
        Segment data containing ['video_name', 'segment_start', 'segment_end', 'social_state'].
    use_ground_truth : bool, default=False
        If True, calculates frame ranges using start_time_sec * FPS.
    social_state_mode : str, default='tertiary'
        Granularity: 'tertiary' (Interacting, Available, Alone) or 'binary'.

    Returns
    -------
    pd.DataFrame
        Frame dataframe augmented with boolean state indicator columns.
    """
    target_cols = ["is_interaction"]
    type_to_col = {"Interacting": "is_interaction"}

    if social_state_mode == "binary":
        target_cols.append("is_not_interaction")
        type_to_col["Not_Interacting"] = "is_not_interaction"
        type_to_col["Not interacting"] = "is_not_interaction"
    else:
        target_cols.extend(["is_alone", "is_available"])
        type_to_col.update({
            "Alone": "is_alone",
            "Available": "is_available",
        })

    for col in target_cols:
        frames_df[col] = False

    video_groups = dict(list(frames_df.groupby("video_name")))
    processed_fragments = []

    for video_name, segments in segments_df.groupby("video_name"):
        if video_name not in video_groups:
            continue

        v_frames = video_groups[video_name].copy()

        for _, seg in segments.iterrows():
            raw_state = str(seg["social_state"]).capitalize()
            col = type_to_col.get(raw_state)
            if col:
                if use_ground_truth:
                    start_frame = int(round(seg["start_time_sec"] * VideoSpecs.FPS))
                    end_frame = int(round(seg["end_time_sec"] * VideoSpecs.FPS))
                else:
                    start_frame = int(seg.get("segment_start", round(seg["start_time_sec"] * VideoSpecs.FPS)))
                    end_frame = int(seg.get("segment_end", round(seg["end_time_sec"] * VideoSpecs.FPS)))

                mask = (v_frames["frame_number"] >= start_frame) & (v_frames["frame_number"] <= end_frame)
                v_frames.loc[mask, col] = True

        processed_fragments.append(v_frames)
        del video_groups[video_name]

    if use_ground_truth:
        final_df = pd.concat(processed_fragments).sort_index()
    else:
        remaining_frames = list(video_groups.values())
        final_df = pd.concat(processed_fragments + remaining_frames).sort_index()

    return final_df


def run_composition_analysis(
    social_state_mode: str = "tertiary",
    output_folder: Optional[Path] = None,
    use_folds: bool = False,
    use_ground_truth: bool = False,
) -> None:
    """
    Generate frame-level interaction composition by merging segment states with frame cues.

    Parameters
    ----------
    social_state_mode : str, default='tertiary'
        Granularity: 'tertiary' or 'binary'.
    output_folder : Optional[Path], default=None
        Target folder for exporting tabular CSV results.
    use_folds : bool, default=False
        If True, assigns cross-validation fold IDs (1 to 5).
    use_ground_truth : bool, default=False
        If True, evaluates on manual ground-truth segments.
    """
    mode_str = "per-fold metrics" if use_folds else "overall recording metrics"
    benchmark_str = "Ground Truth Validation" if use_ground_truth else "Standard Pipeline Predictions"
    logger.info("Executing RQ 05: Interaction Composition [%s | %s]", mode_str, benchmark_str)

    # 1. Resolve Frame and Segment CSV Paths
    if output_folder:
        frames_path = output_folder / AnalysisPaths.FRAME_INTERACTIONS_CSV.name
        segments_path = output_folder / AnalysisPaths.INTERACTION_SEGMENTS_CSV.name
    else:
        frames_path = AnalysisPaths.FRAME_INTERACTIONS_CSV
        segments_path = AnalysisPaths.INTERACTION_SEGMENTS_CSV

    if not frames_path.exists():
        logger.error("Frame interactions CSV not found: %s", frames_path)
        return

    frames_df = pd.read_csv(frames_path)

    if use_ground_truth:
        segments_path = AnalysisPaths.GT_SEGMENTS_CSV
        if not segments_path.exists():
            logger.error("Ground truth file not found: %s", segments_path)
            return
        segments_df = pd.read_csv(segments_path)
        gt_videos = segments_df["video_name"].unique()
        frames_df = frames_df[frames_df["video_name"].isin(gt_videos)].copy()
    else:
        if not segments_path.exists():
            logger.error("Segments file not found: %s", segments_path)
            return
        segments_df = pd.read_csv(segments_path)

    # Standardize numeric types and child IDs
    segments_df["start_time_sec"] = pd.to_numeric(segments_df["start_time_sec"], errors="coerce")
    segments_df["end_time_sec"] = pd.to_numeric(segments_df["end_time_sec"], errors="coerce")
    segments_df["duration_sec"] = pd.to_numeric(segments_df["duration_sec"], errors="coerce")

    if "child_id" not in segments_df.columns:
        segments_df["child_id"] = segments_df["video_name"].apply(extract_child_id).astype(str)
    else:
        segments_df["child_id"] = segments_df["child_id"].astype(str)

    frames_df["child_id"] = frames_df["video_name"].apply(extract_child_id).astype(str)

    # 2. Global Timeline Offsets across multiple recordings per child
    video_stats = (
        segments_df.groupby(["child_id", "video_name"])["end_time_sec"]
        .max()
        .reset_index(name="video_span_sec")
    )
    video_stats = video_stats.sort_values(["child_id", "video_name"])
    video_stats["offset_raw"] = video_stats.groupby("child_id")["video_span_sec"].shift(1).fillna(0)
    video_stats["offset_sec"] = video_stats.groupby("child_id")["offset_raw"].transform("cumsum")

    frames_df = frames_df.merge(video_stats[["child_id", "video_name", "offset_sec"]], on=["child_id", "video_name"])
    segments_df = segments_df.merge(video_stats[["child_id", "video_name", "offset_sec"]], on=["child_id", "video_name"])

    segments_df["global_start"] = segments_df["start_time_sec"] + segments_df["offset_sec"]
    segments_df["global_end"] = segments_df["end_time_sec"] + segments_df["offset_sec"]

    frames_df["global_sec_pos"] = (frames_df["frame_number"] / VideoSpecs.FPS) + frames_df["offset_sec"]

    # 3. Assign Folds
    if use_folds:
        fold_map = get_child_fold_boundaries(segments_df)
        frames_df["fold"] = 0

        for child_id, folds in fold_map.items():
            child_mask = frames_df["child_id"] == child_id
            for fold_idx, (f_start, f_end) in enumerate(folds):
                fold_mask = (frames_df["global_sec_pos"] >= f_start) & (frames_df["global_sec_pos"] < f_end)
                if fold_idx == len(folds) - 1:
                    fold_mask = (frames_df["global_sec_pos"] >= f_start) & (frames_df["global_sec_pos"] <= f_end + 0.1)

                frames_df.loc[child_mask & fold_mask, "fold"] = fold_idx + 1
    else:
        frames_df["fold"] = 1

    # 4. Map Social Interaction Columns
    frames_df = add_interaction_columns(
        frames_df=frames_df,
        segments_df=segments_df,
        use_ground_truth=use_ground_truth,
        social_state_mode=social_state_mode,
    )

    # 5. Metadata Sync
    logger.info("Syncing participant metadata from segments...")
    meta_cols = [c for c in ["video_name", "age_at_recording", "child_id"] if c in segments_df.columns]
    metadata_map = segments_df[meta_cols].drop_duplicates()

    if "age_at_recording" in metadata_map.columns:
        metadata_map["age_at_recording"] = pd.to_numeric(
            metadata_map["age_at_recording"]
            .astype(str)
            .str.replace('"', "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip(),
            errors="coerce",
        )
        frames_df = frames_df.merge(metadata_map, on=["video_name", "child_id"], how="left")

    if "proximity" in frames_df.columns:
        frames_df["proximity_filled"] = frames_df["proximity"].fillna(-1.0)

    # 6. Save Formatted Output
    dest_folder = output_folder or AnalysisPaths.OUTPUT_FOLDER
    dest_folder.mkdir(parents=True, exist_ok=True)
    suffix = "_gt" if use_ground_truth else ""

    output_path = dest_folder / f"{AnalysisPaths.CUE_SPECIFICITY_CSV.stem}{suffix}.csv"
    frames_df.to_csv(output_path, index=False)

    if social_state_mode == "binary":
        logger.info(
            "Composition complete: %d Interacting, %d Not Interacting frames.",
            int(frames_df["is_interaction"].sum()),
            int(frames_df["is_not_interaction"].sum()),
        )
    else:
        logger.info(
            "Composition complete: %d Interacting, %d Available, %d Alone frames.",
            int(frames_df["is_interaction"].sum()),
            int(frames_df.get("is_available", pd.Series(0)).sum()),
            int(frames_df.get("is_alone", pd.Series(0)).sum()),
        )

    logger.info("Saved interaction composition dataset to: %s", output_path)


def main() -> None:
    """CLI entrypoint for RQ 05 Interaction Composition Analysis."""
    parser = argparse.ArgumentParser(description="Frame-level interaction composition analysis")
    parser.add_argument(
        "--social_state_mode",
        choices=["binary", "tertiary"],
        default="tertiary",
        help="Classification granularity mode",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=None,
        help="Target folder for output CSV files",
    )
    parser.add_argument(
        "--use_folds",
        action="store_true",
        default=False,
        help="Calculate per-fold CV metrics instead of overall recording metrics",
    )
    parser.add_argument(
        "--use_gt",
        action="store_true",
        default=False,
        help="Map vocalizations onto manual ground-truth segments instead of pipeline segments",
    )
    args = parser.parse_args()

    run_composition_analysis(
        social_state_mode=args.social_state_mode,
        output_folder=args.output_folder,
        use_folds=args.use_folds,
        use_ground_truth=args.use_gt,
    )


if __name__ == "__main__":
    main()