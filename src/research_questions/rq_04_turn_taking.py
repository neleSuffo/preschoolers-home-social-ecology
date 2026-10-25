"""
src/research_questions/rq_04_turn_taking.py

Research Question 4: Conversational Dynamics & Turn-Taking Analysis.

Categorizes conversational bouts into successful child-led initiations,
adult-led responses, and unanswered communicative bids. Computes conversational
turn density (turns/min), response latencies, and child vocalization turn durations.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import AnalysisConfig
from src.constants import AnalysisPaths, BasePaths
from src.research_questions.utils import get_child_fold_boundaries, parse_rttm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_block(block: List[Dict[str, Any]]) -> Tuple[int, int, int, int, int, float, float]:
    """
    Categorize a contiguous vocalization block into interaction initiation and response types.

    Parameters
    ----------
    block : List[Dict[str, Any]]
        Sequence of vocalizations from 'KCHI' or 'KCDS' with inter-turn gaps within threshold.

    Returns
    -------
    Tuple[int, int, int, int, int, float, float]
        (total_turns, succ_init, succ_resp, fail_init, fail_resp, succ_block_dur, block_duration)
    """
    if not block:
        return 0, 0, 0, 0, 0, 0.0, 0.0

    block_duration = block[-1]["end_time_seconds"] - block[0]["start_time_seconds"]

    turns = 0
    for i in range(1, len(block)):
        if block[i]["speech_type"] != block[i - 1]["speech_type"]:
            turns += 1

    succ_init, succ_resp, fail_init, fail_resp = 0, 0, 0, 0
    first_speaker = block[0]["speech_type"]

    if turns > 0:
        succ_block_dur = block_duration
        if first_speaker == "KCHI":
            succ_init = 1  # Successful exchange initiated by child
        else:
            succ_resp = 1  # Successful exchange led by adult
    else:
        succ_block_dur = 0.0
        if first_speaker == "KCHI":
            fail_init = 1  # Unanswered child bid
        else:
            fail_resp = 1  # Unanswered adult prompt

    return turns, succ_init, succ_resp, fail_init, fail_resp, succ_block_dur, block_duration


def count_directional_turns(
    vocalizations: pd.DataFrame,
    segments_df: pd.DataFrame,
    use_folds: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Any, List[Tuple[float, float]]]]:
    """
    Count directional turn-taking within interacting segments and measure individual turn durations.

    Parameters
    ----------
    vocalizations : pd.DataFrame
        DataFrame with ['video_name', 'start_time_seconds', 'end_time_seconds', 'speech_type'].
    segments_df : pd.DataFrame
        Interaction segments with start and end timestamps.
    use_folds : bool, default=True
        Whether to partition calculations across 5 cross-validation folds.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Any, List[Tuple[float, float]]]]
        (results_df, raw_turns_df, turn_gaps_df, fold_map)
    """
    # 1. Global Timeline Offsets
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

    # 2. Fold Boundaries
    if use_folds:
        fold_map = get_child_fold_boundaries(segments_df)
    else:
        fold_map = {}
        for child_id in segments_df["child_id"].unique():
            child_data = segments_df[segments_df["child_id"] == child_id]
            f_start = child_data["global_start"].min()
            f_end = child_data["global_end"].max()
            fold_map[child_id] = [(f_start, f_end)]

    results: List[Dict[str, Any]] = []
    raw_turns_df: List[Dict[str, Any]] = []
    turn_gaps_list: List[Dict[str, Any]] = []

    # 3. Iterate by Child -> Fold -> Segment
    for child_id, folds in fold_map.items():
        child_segs = segments_df[segments_df["child_id"] == child_id]

        for fold_idx, (f_start, f_end) in enumerate(folds, 1):
            for _, seg in child_segs.iterrows():
                state_val = seg.get("social_state", "Unknown").capitalize()
                if state_val != "Interacting":
                    continue

                # STEP A: Global Clipping
                overlap_start = max(seg["global_start"], f_start)
                overlap_end = min(seg["global_end"], f_end)

                if overlap_start < overlap_end:
                    current_duration = overlap_end - overlap_start
                    local_overlap_start = overlap_start - seg["offset"]
                    local_overlap_end = overlap_end - seg["offset"]

                    # STEP B: Local Filter
                    seg_vocs = vocalizations[
                        (vocalizations["video_name"] == seg["video_name"])
                        & (vocalizations["start_time_seconds"] < local_overlap_end)
                        & (vocalizations["end_time_seconds"] > local_overlap_start)
                    ].copy().sort_values("start_time_seconds")

                    # STEP C: Triple Clip
                    seg_vocs["clipped_start"] = np.maximum(seg_vocs["start_time_seconds"], local_overlap_start)
                    seg_vocs["clipped_end"] = np.minimum(seg_vocs["end_time_seconds"], local_overlap_end)
                    seg_vocs["duration_seconds"] = seg_vocs["clipped_end"] - seg_vocs["clipped_start"]

                    seg_vocs["start_time_seconds"] = seg_vocs["clipped_start"]
                    seg_vocs["end_time_seconds"] = seg_vocs["clipped_end"]
                    seg_vocs = seg_vocs[seg_vocs["speech_type"].isin(["KCHI", "KCDS"])].reset_index(drop=True)

                    # Capture child-only turn durations
                    child_only = seg_vocs[seg_vocs["speech_type"] == "KCHI"]
                    for _, voc in child_only.iterrows():
                        raw_turns_df.append({
                            "child_id": child_id,
                            "fold": fold_idx,
                            "age_at_recording": seg.get("age_at_recording", np.nan),
                            "video_name": seg["video_name"],
                            "vocalization_duration_sec": voc["duration_seconds"],
                            "start_time": voc["start_time_seconds"],
                        })

                    # Process Turn Blocks
                    total_turns, s_init, s_resp, f_init, f_resp = 0, 0, 0, 0, 0
                    succ_block_duration = 0.0
                    all_block_duration = 0.0

                    if not seg_vocs.empty:
                        current_block = [seg_vocs.iloc[0].to_dict()]
                        for i in range(1, len(seg_vocs)):
                            prev = seg_vocs.iloc[i - 1]
                            curr = seg_vocs.iloc[i]
                            gap = curr["start_time_seconds"] - prev["end_time_seconds"]
                            threshold = (
                                AnalysisConfig.MAX_SAME_SPEAKER_GAP_SEC
                                if curr["speech_type"] == prev["speech_type"]
                                else AnalysisConfig.MAX_TURN_TAKING_GAP_SEC
                            )

                            if gap <= threshold:
                                current_block.append(curr.to_dict())
                                if curr["speech_type"] != prev["speech_type"]:
                                    turn_gaps_list.append({
                                        "child_id": child_id,
                                        "age_at_recording": seg.get("age_at_recording", np.nan),
                                        "video_name": seg["video_name"],
                                        "speaker_from": prev["speech_type"],
                                        "speaker_to": curr["speech_type"],
                                        "gap_seconds": max(0.0, gap),
                                    })
                            else:
                                t_c, si, sr, fi, fr, sbd, bd = process_block(current_block)
                                total_turns += t_c
                                s_init += si
                                s_resp += sr
                                f_init += fi
                                f_resp += fr
                                succ_block_duration += sbd
                                all_block_duration += bd
                                current_block = [curr.to_dict()]

                        t_c, si, sr, fi, fr, sbd, bd = process_block(current_block)
                        total_turns += t_c
                        s_init += si
                        s_resp += sr
                        f_init += fi
                        f_resp += fr
                        succ_block_duration += sbd
                        all_block_duration += bd

                    results.append({
                        "child_id": child_id,
                        "fold": fold_idx,
                        "video_name": seg["video_name"],
                        "age_at_recording": seg.get("age_at_recording", np.nan),
                        "segment_start": seg["start_time_sec"],
                        "segment_end": seg["end_time_sec"],
                        "duration_sec": current_duration,
                        "total_turns": total_turns,
                        "successful_initiations": s_init,
                        "successful_responses": s_resp,
                        "unanswered_child_bids": f_init,
                        "unanswered_adult_prompts": f_resp,
                        "successful_block_duration_sec": succ_block_duration,
                        "all_block_duration_sec": all_block_duration,
                        "segment_duration_minutes": current_duration / 60.0,
                    })

    return pd.DataFrame(results), pd.DataFrame(raw_turns_df), pd.DataFrame(turn_gaps_list), fold_map


def run_turn_taking_analysis(
    social_state_mode: str = "tertiary",
    output_folder: Optional[Path] = None,
    use_folds: bool = False,
    use_ground_truth: bool = False,
) -> None:
    """
    Execute conversational turn-taking, response latency, and duration analyses.

    Parameters
    ----------
    social_state_mode : str, default='tertiary'
        Granularity mode: 'tertiary' or 'binary'.
    output_folder : Optional[Path], default=None
        Target output directory.
    use_folds : bool, default=False
        If True, calculates per-fold metrics across 5 folds.
    use_ground_truth : bool, default=False
        If True, maps vocalizations onto ground_truth_segments.csv.
    """
    mode_str = "per-fold metrics" if use_folds else "overall recording metrics"
    benchmark_str = "Ground Truth Validation" if use_ground_truth else "Standard Pipeline Predictions"
    logger.info("Executing RQ 04: Turn-Taking & Conversational Dynamics [%s | %s]", mode_str, benchmark_str)

    AnalysisConfig.apply_mode(social_state_mode)

    # 1. Load Segments and Diarized Speech
    if use_ground_truth:
        segments_path = AnalysisPaths.INTERACTION_SEGMENTS_CSV
    else:
        segments_path = (output_folder / AnalysisPaths.INTERACTION_SEGMENTS_CSV.name) if output_folder else AnalysisPaths.INTERACTION_SEGMENTS_CSV

    if not segments_path.exists():
        logger.error("Segments file not found: %s", segments_path)
        return

    segments_df = pd.read_csv(segments_path)

    all_vocalizations = parse_rttm(target_speech_types=["KCHI", "KCDS"])
    if all_vocalizations.empty:
        logger.warning("No KCHI or KCDS speech tokens found in RTTM file.")

    # 2. Extract Turn Events and Bouts
    final_df, raw_turns_df, turn_gaps_df, fold_map = count_directional_turns(
        all_vocalizations, segments_df, use_folds=use_folds
    )

    if final_df.empty:
        logger.warning("No interacting segments found to evaluate turn taking.")
        return

    # 3. Calculate Global Totals and Proportions
    final_df["total_successful_blocks"] = (
        final_df["successful_initiations"] + final_df["successful_responses"]
    )
    final_df["total_attempts"] = (
        final_df["successful_initiations"]
        + final_df["successful_responses"]
        + final_df["unanswered_child_bids"]
        + final_df["unanswered_adult_prompts"]
    )

    final_df["avg_conversational_block_duration_sec"] = np.where(
        final_df["total_successful_blocks"] > 0,
        final_df["successful_block_duration_sec"] / final_df["total_successful_blocks"],
        np.nan,
    )

    for col in ["successful_initiations", "successful_responses", "unanswered_child_bids", "unanswered_adult_prompts"]:
        final_df[f"pct_{col}"] = (final_df[col] / final_df["total_attempts"]).fillna(0.0)

    final_df["turns_per_minute"] = (final_df["total_turns"] / final_df["segment_duration_minutes"]).fillna(0.0)

    if "age_at_recording" in final_df.columns:
        final_df["age_at_recording"] = (
            final_df["age_at_recording"]
            .astype(str)
            .str.replace('"', "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        final_df["age_at_recording"] = pd.to_numeric(final_df["age_at_recording"], errors="coerce")

    final_output = final_df.sort_values(["video_name", "segment_start"])

    dest_folder = output_folder or AnalysisPaths.OUTPUT_FOLDER
    dest_folder.mkdir(parents=True, exist_ok=True)
    suffix = "_gt" if use_ground_truth else ""

    output_path_tt = dest_folder / f"{AnalysisPaths.TURN_TAKING_CSV.stem}{suffix}.csv"
    final_output.to_csv(output_path_tt, index=False)
    logger.info("Saved full four-category turn analysis to: %s", output_path_tt)

    # 4. Child-Level Aggregation (Relative to Total Recording Duration)
    child_total_durations = segments_df.groupby("child_id")["duration_sec"].sum().reset_index()
    child_total_durations.rename(columns={"duration_sec": "total_recording_duration_sec"}, inplace=True)

    if use_folds:
        fold_durations = []
        for child_id, folds in fold_map.items():
            child_segs = segments_df[segments_df["child_id"] == child_id]
            for fold_idx, (f_start, f_end) in enumerate(folds, 1):
                f_dur = 0.0
                for _, seg in child_segs.iterrows():
                    o_start = max(seg["global_start"], f_start)
                    o_end = min(seg["global_end"], f_end)
                    if o_start < o_end:
                        f_dur += o_end - o_start
                fold_durations.append({
                    "child_id": child_id,
                    "fold": fold_idx,
                    "total_recording_duration_sec": f_dur,
                })
        df_fold_durations = pd.DataFrame(fold_durations)

        child_level_turns = final_df.groupby(["child_id", "fold"]).agg({
            "total_turns": "sum",
            "duration_sec": "sum",
            "age_at_recording": "min",
        }).reset_index()
        child_level_turns.rename(columns={"duration_sec": "total_interacting_sec"}, inplace=True)
        child_level_turns = child_level_turns.merge(df_fold_durations, on=["child_id", "fold"], how="left")
    else:
        child_level_turns = final_df.groupby("child_id").agg({
            "total_turns": "sum",
            "duration_sec": "sum",
            "age_at_recording": "min",
        }).reset_index()
        child_level_turns.rename(columns={"duration_sec": "total_interacting_sec"}, inplace=True)
        child_level_turns = child_level_turns.merge(child_total_durations, on="child_id", how="left")

    child_level_turns["total_recording_minutes"] = child_level_turns["total_recording_duration_sec"] / 60.0
    child_level_turns["turns_per_minute"] = (
        child_level_turns["total_turns"] / child_level_turns["total_recording_minutes"]
    ).fillna(0.0)

    output_path_gtt = dest_folder / f"{AnalysisPaths.GLOBAL_TURN_TAKING_CSV.stem}{suffix}.csv"
    child_level_turns.to_csv(output_path_gtt, index=False)
    logger.info("Saved child-level turn-taking summary to: %s", output_path_gtt)

    # 5. Individual Child Turn Durations
    if not raw_turns_df.empty:
        raw_turns_df["age_at_recording"] = (
            raw_turns_df["age_at_recording"]
            .astype(str)
            .str.replace('"', "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        raw_turns_df["age_at_recording"] = pd.to_numeric(raw_turns_df["age_at_recording"], errors="coerce")

        output_path_td = dest_folder / f"{AnalysisPaths.TURN_DURATION_CSV.stem}{suffix}.csv"
        raw_turns_df.to_csv(output_path_td, index=False)
        logger.info("Saved individual child turn durations to: %s", output_path_td)

    # 6. Mean Inter-Speaker Latency (Turn Gaps) per Child
    if not turn_gaps_df.empty:
        turn_gaps_df["age_at_recording"] = (
            turn_gaps_df["age_at_recording"]
            .astype(str)
            .str.replace('"', "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        turn_gaps_df["age_at_recording"] = pd.to_numeric(turn_gaps_df["age_at_recording"], errors="coerce")

        child_gaps = turn_gaps_df.groupby("child_id").agg(
            mean_gap_sec=("gap_seconds", "mean"),
            median_gap_sec=("gap_seconds", "median"),
            std_gap_sec=("gap_seconds", "std"),
            min_gap_sec=("gap_seconds", "min"),
            max_gap_sec=("gap_seconds", "max"),
            total_turn_transitions=("gap_seconds", "count"),
            age_at_recording=("age_at_recording", "min"),
        ).reset_index()

        output_path_tg = dest_folder / f"{AnalysisPaths.TURN_GAPS_CSV.stem}{suffix}.csv"
        child_gaps.to_csv(output_path_tg, index=False)
        logger.info("Saved mean inter-speaker turn gap per child to: %s", output_path_tg)


def main() -> None:
    """CLI entrypoint for RQ 04 Conversational Dynamics Analysis."""
    parser = argparse.ArgumentParser(description="Turn-Taking Analysis for Naturalistic Social Interactions")
    parser.add_argument("--social_state_mode", choices=["binary", "tertiary"], default="tertiary", help="State granularity")
    parser.add_argument("--output_folder", type=Path, default=None, help="Target folder for output CSVs")
    parser.add_argument("--use_folds", action="store_true", default=False, help="Calculate per-fold CV metrics")
    parser.add_argument("--use_gt", action="store_true", default=False, help="Map vocalizations onto ground truth segments")
    args = parser.parse_args()

    run_turn_taking_analysis(
        social_state_mode=args.social_state_mode,
        output_folder=args.output_folder,
        use_folds=args.use_folds,
        use_ground_truth=args.use_gt,
    )


if __name__ == "__main__":
    main()