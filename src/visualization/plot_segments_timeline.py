"""
src/visualization/plot_segments_timeline.py

Visualizes temporal alignment between predicted social ecology segments
and human ground-truth annotations across time for individual video recordings.
Supports both tertiary (Interacting, Available, Alone) and binary modes.
"""

import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import AnalysisPaths, BasePaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_clean_segments(df: pd.DataFrame, video_name: str, mode: str) -> pd.DataFrame:
    """
    Filter segment data for a target recording and standardize state labels.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe containing segment annotations.
    video_name : str
        Target video identifier (stem or filename).
    mode : str
        Classification granularity: 'tertiary' or 'binary'.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe containing valid, sorted segments for the requested video.
    """
    clean_target = Path(video_name).stem.strip().lower()

    df_work = df.copy()
    df_work["clean_stem"] = (
        df_work["video_name"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.[^.]+$", "", regex=True)
        .str.lower()
    )

    df_filtered = df_work[df_work["clean_stem"] == clean_target].copy()
    if df_filtered.empty:
        return df_filtered

    df_filtered["start_time_sec"] = pd.to_numeric(df_filtered["start_time_sec"], errors="coerce")
    df_filtered["end_time_sec"] = pd.to_numeric(df_filtered["end_time_sec"], errors="coerce")
    df_filtered["duration_sec"] = pd.to_numeric(df_filtered["duration_sec"], errors="coerce")

    # Support both 'social_state' and legacy 'interaction_type' columns
    state_col = "social_state" if "social_state" in df_filtered.columns else "interaction_type"
    df_filtered["social_state"] = df_filtered[state_col].astype(str).str.lower().str.strip()

    if mode.lower() == "binary":
        binary_map = {
            "interacting": "interacting",
            "available": "not interacting",
            "alone": "not interacting",
            "not interacting": "not interacting",
        }
        df_filtered["social_state"] = df_filtered["social_state"].map(binary_map).fillna("not interacting")

    df_filtered.dropna(subset=["start_time_sec", "end_time_sec", "duration_sec"], inplace=True)
    df_filtered.sort_values(by="start_time_sec", inplace=True)
    return df_filtered


def plot_segment_timeline(
    video_name: str,
    output_path: Path,
    pred_df_raw: pd.DataFrame,
    gt_df_raw: pd.DataFrame,
    mode: str = "tertiary",
    show_legend: bool = True,
) -> None:
    """
    Render comparative timeline horizontal bar plot (Ground Truth vs. Model Prediction).

    Parameters
    ----------
    video_name : str
        Target video identifier.
    output_path : Path
        Destination image path (.png).
    pred_df_raw : pd.DataFrame
        Dataframe containing model-generated segment predictions.
    gt_df_raw : pd.DataFrame
        Dataframe containing human validation ground-truth segments.
    mode : str, default='tertiary'
        Classification granularity: 'tertiary' or 'binary'.
    show_legend : bool, default=True
        Whether to render the category legend atop the figure.
    """
    pred = load_and_clean_segments(pred_df_raw, video_name, mode)
    gt = load_and_clean_segments(gt_df_raw, video_name, mode)

    if pred.empty and gt.empty:
        logger.warning("No segments found for video '%s' in either dataset. Skipping.", video_name)
        return

    is_binary = mode.lower() == "binary"
    if is_binary:
        interaction_colors = {
            "interacting": "#691633",      # Burgundy
            "not interacting": "#69777B",  # Slate Gray
        }
    else:
        interaction_colors = {
            "interacting": "#691633",  # Deep Burgundy
            "available": "#8E7C3B",    # Muted Olive / Gold
            "alone": "#69777B",        # Slate Gray
        }

    # Determine timeline upper boundary rounded up to the nearest minute
    max_time = max(
        pred["end_time_sec"].max() if not pred.empty else 0.0,
        gt["end_time_sec"].max() if not gt.empty else 0.0,
    )
    max_time = float(np.ceil(max_time / 60.0) * 60.0)

    fig_height = 4.0 if show_legend else 3.2
    fig, ax = plt.subplots(figsize=(16, fig_height))

    # 1. Ground Truth Bar (Y = 1.2)
    y_pos_gt = 1.2
    for _, row in gt.iterrows():
        color = interaction_colors.get(row["social_state"], "#808080")
        ax.barh(
            y=y_pos_gt,
            width=row["duration_sec"],
            left=row["start_time_sec"],
            height=0.22,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

    # 2. Prediction Bar (Y = 0.8)
    y_pos_pred = 0.8
    for _, row in pred.iterrows():
        color = interaction_colors.get(row["social_state"], "#808080")
        ax.barh(
            y=y_pos_pred,
            width=row["duration_sec"],
            left=row["start_time_sec"],
            height=0.22,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

    ax.set_yticks([y_pos_pred, y_pos_gt])
    ax.set_yticklabels(["Prediction", "Ground Truth"], fontsize=18, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=18, fontweight="bold", labelpad=8)
    ax.tick_params(axis="x", labelsize=16)
    ax.set_xlim(0, max_time)

    if show_legend:
        ax.set_ylim(0.5, 1.5)
        legend_patches = [
            mpatches.Patch(color=interaction_colors[cat], alpha=0.85, label=cat.title())
            for cat in sorted(interaction_colors.keys())
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.30),
            ncol=len(legend_patches),
            frameon=False,
            fontsize=18,
        )
    else:
        ax.set_ylim(0.5, 1.45)

    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved timeline comparison to: %s", output_path)


def main() -> None:
    """CLI entrypoint for rendering comparative segment timeline plots."""
    parser = argparse.ArgumentParser(
        description="Plot Ground Truth vs. Prediction segment timelines for video sessions."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="all",
        help="Target video stem, or 'all' to plot all benchmark sessions",
    )
    parser.add_argument(
        "--mode",
        choices=["binary", "tertiary"],
        default="tertiary",
        help="Classification granularity: 'tertiary' or 'binary'",
    )
    parser.add_argument(
        "--gt_csv",
        type=Path,
        default=BasePaths.DATA_DIR / "ground_truth_segments.csv",
        help="Path to ground truth validation segments CSV",
    )
    parser.add_argument(
        "--pred_csv",
        type=Path,
        default=AnalysisPaths.INTERACTION_SEGMENTS_CSV,
        help="Path to predicted interaction_segments.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=AnalysisPaths.SEGMENT_TIMELINE_DIR,
        help="Directory to save generated timeline figures",
    )
    parser.add_argument(
        "--no_legend",
        action="store_true",
        help="Disable the top legend in the output plot",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.gt_csv.exists():
        logger.error("Ground truth file not found: %s", args.gt_csv)
        return
    if not args.pred_csv.exists():
        logger.error("Prediction file not found: %s", args.pred_csv)
        return

    logger.info("Reading ground truth from: %s", args.gt_csv)
    gt_df = pd.read_csv(args.gt_csv)

    logger.info("Reading predictions from: %s", args.pred_csv)
    pred_df = pd.read_csv(args.pred_csv)

    if args.video.strip().lower() == "all":
        video_list: List[str] = sorted(gt_df["video_name"].dropna().unique().tolist())
        logger.info("Identified %d unique recordings in ground truth benchmark.", len(video_list))
    else:
        video_list = [args.video.strip()]

    for vid in video_list:
        clean_name = Path(vid).stem
        out_file = args.output_dir / f"{clean_name}_segment_timeline.png"
        plot_segment_timeline(
            video_name=vid,
            output_path=out_file,
            pred_df_raw=pred_df,
            gt_df_raw=gt_df,
            mode=args.mode,
            show_legend=not args.no_legend,
        )


if __name__ == "__main__":
    main()