"""
src/evaluation/eval_segment_performance.py

Evaluates predicted social interaction segments against human ground-truth validation data.
Computes second-by-second accuracy, Cohen's kappa (κ), precision, recall, macro F1-score,
generates publication-ready confusion matrix heatmaps, and extracts continuous misclassification events.
"""

import argparse
from collections import defaultdict
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

from src.config import AnalysisConfig
from src.constants import AnalysisPaths, BasePaths
from src.research_questions.utils import create_second_level_labels, time_to_seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

UNCLASSIFIED_LABEL: str = "unclassified"

# BGR / Hex Palette matching publication styling
INTERACTION_COLORS_TERTIARY: Dict[str, str] = {
    "interacting": "#691633",  # Deep Burgundy
    "available": "#8E7C3B",    # Muted Olive / Gold
    "alone": "#69777B",        # Slate Gray
}
INTERACTION_COLORS_BINARY: Dict[str, str] = {
    "interacting": "#691633",
    "not interacting": "#69777B",
}


def normalize_stem(val: Any) -> str:
    """Normalize video identifiers by stripping file extensions and whitespace."""
    s = str(val).strip()
    return re.sub(r"\.[^.]+$", "", s).lower()


def reclassify_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reclassify tertiary 3-state labels to binary ('interacting' vs. 'not interacting').

    Parameters
    ----------
    df : pd.DataFrame
        Input segments dataframe containing 'social_state'.

    Returns
    -------
    pd.DataFrame
        Dataframe with mapped binary labels.
    """
    df_copy = df.copy()
    df_copy["social_state"] = df_copy["social_state"].astype(str).str.lower().str.strip()

    mapping = {
        "interacting": "interacting",
        "available": "not interacting",
        "alone": "not interacting",
        "not interacting": "not interacting",
    }

    df_copy["social_state"] = df_copy["social_state"].map(mapping).fillna("not interacting")
    return df_copy


def save_second_wise_labels(df: pd.DataFrame, output_folder: Path, filename: str) -> None:
    """
    Generate dense second-by-second classification series and persist to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with start and end times in seconds.
    output_folder : Path
        Target directory.
    filename : str
        Target CSV filename.
    """
    all_second_wise_data = []

    # Ensure time columns are numeric seconds
    if "start_time_sec" not in df.columns and "start_time_min" in df.columns:
        df["start_time_sec"] = df["start_time_min"].apply(time_to_seconds)
        df["end_time_sec"] = df["end_time_min"].apply(time_to_seconds)

    for video_name in df["video_name"].unique():
        video_df = df[df["video_name"] == video_name].copy()
        max_end = video_df["end_time_sec"].max() if not video_df.empty else 0
        video_duration_seconds = int(max_end) + 1

        if video_duration_seconds <= 1:
            continue

        labels_raw = create_second_level_labels(video_df, video_duration_seconds)
        for sec in range(video_duration_seconds):
            label = labels_raw[sec]
            all_second_wise_data.append({
                "video_name": video_name,
                "second": sec,
                "social_state": str(label).lower() if label is not None else UNCLASSIFIED_LABEL,
            })

    if all_second_wise_data:
        df_out = pd.DataFrame(all_second_wise_data)
        out_path = output_folder / filename
        df_out.to_csv(out_path, index=False)
        logger.info("Saved dense second-wise labels to: %s", out_path)


def plot_segment_timeline(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    video_name: str,
    save_path: Path,
    mode: str = "tertiary",
) -> None:
    """
    Plot comparative timeline bar chart between ground truth and model predictions.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predicted interaction segments.
    ground_truth_df : pd.DataFrame
        Ground-truth validation segments.
    video_name : str
        Target video identifier.
    save_path : Path
        Output image path (.png).
    mode : str, default='tertiary'
        Granularity: 'tertiary' or 'binary'.
    """
    is_binary = mode == "binary"
    colors = INTERACTION_COLORS_BINARY if is_binary else INTERACTION_COLORS_TERTIARY
    target_clean = normalize_stem(video_name)

    # Filter by exact normalized stem
    pred_work = predictions_df.copy()
    pred_work["stem"] = pred_work["video_name"].apply(normalize_stem)
    pred = pred_work[pred_work["stem"] == target_clean].copy()

    gt_work = ground_truth_df.copy()
    gt_work["stem"] = gt_work["video_name"].apply(normalize_stem)
    gt = gt_work[gt_work["stem"] == target_clean].copy()

    if gt.empty and pred.empty:
        logger.warning("No data found for video '%s' in ground truth or predictions.", video_name)
        return

    for df_sub in [pred, gt]:
        df_sub["social_state"] = df_sub["social_state"].astype(str).str.lower().str.strip()
        df_sub["duration_sec"] = df_sub["end_time_sec"] - df_sub["start_time_sec"]
        df_sub.dropna(subset=["start_time_sec", "end_time_sec"], inplace=True)

    max_time = max(
        pred["end_time_sec"].max() if not pred.empty else 0.0,
        gt["end_time_sec"].max() if not gt.empty else 0.0,
    )
    max_time = float(np.ceil(max_time / 60.0) * 60.0)

    fig, ax = plt.subplots(figsize=(15, 4))
    y_pos_gt = 1.2
    y_pos_pred = 0.8

    for _, row in gt.iterrows():
        c = colors.get(row["social_state"], "#808080")
        ax.barh(
            y=y_pos_gt,
            width=row["duration_sec"],
            left=row["start_time_sec"],
            height=0.22,
            color=c,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

    for _, row in pred.iterrows():
        c = colors.get(row["social_state"], "#808080")
        ax.barh(
            y=y_pos_pred,
            width=row["duration_sec"],
            left=row["start_time_sec"],
            height=0.22,
            color=c,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

    ax.set_yticks([y_pos_pred, y_pos_gt])
    ax.set_yticklabels(["Prediction", "Ground Truth"], fontsize=14, fontweight="bold")
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("Time (seconds)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max_time)
    ax.set_title(f"Segment Timeline: {video_name} ({mode.capitalize()})", fontsize=15, pad=12)

    legend_patches = [
        mpatches.Patch(color=colors[k], alpha=0.85, label=k.title())
        for k in sorted(colors.keys())
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.32),
        ncol=len(legend_patches),
        frameon=False,
        fontsize=13,
    )

    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_performance_by_seconds(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    video_subset: Optional[List[str]] = None,
    exclusion_seconds: int = AnalysisConfig.EXCLUSION_SECONDS,
) -> Dict[str, Any]:
    """
    Evaluate second-by-second performance excluding the boundary padding windows.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predicted segment intervals.
    ground_truth_df : pd.DataFrame
        Ground-truth segment intervals.
    video_subset : Optional[List[str]], default=None
        Restricts evaluation to specific video stems (e.g., cross-validation fold).
    exclusion_seconds : int, default=30
        Boundary padding excluded at start and end of each video.

    Returns
    -------
    Dict[str, Any]
        Dictionary with overall accuracy, Cohen's kappa, category accuracies,
        and confusion matrix.
    """
    # Build clean mapping
    pred_work = predictions_df.copy()
    pred_work["stem"] = pred_work["video_name"].apply(normalize_stem)
    gt_work = ground_truth_df.copy()
    gt_work["stem"] = gt_work["video_name"].apply(normalize_stem)

    videos_with_gt = set(gt_work["stem"].unique())
    videos_with_pred = set(pred_work["stem"].unique())
    common_videos = videos_with_gt.intersection(videos_with_pred)

    if video_subset:
        subset_stems = {normalize_stem(v) for v in video_subset}
        videos_to_eval = common_videos.intersection(subset_stems)
        logger.info("Evaluating subset of %d recording(s).", len(videos_to_eval))
    else:
        videos_to_eval = common_videos
        logger.info("Evaluating all %d available recording(s).", len(videos_to_eval))

    gt_social_states = sorted(list({str(t).lower() for t in gt_work["social_state"].unique()}))
    total_seconds_all = 0
    correct_seconds_all = 0

    category_stats = {cat: {"total": 0, "correct": 0} for cat in gt_social_states}
    confusion_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    misclassifications: Dict[str, int] = defaultdict(int)
    video_results = []

    all_gt_seconds = []
    all_pred_seconds = []

    for video_stem in sorted(videos_to_eval):
        pred_video = pred_work[pred_work["stem"] == video_stem].copy()
        gt_video = gt_work[gt_work["stem"] == video_stem].copy()

        if pred_video.empty or gt_video.empty:
            continue

        for df_sub in [pred_video, gt_video]:
            df_sub["social_state"] = df_sub["social_state"].astype(str).str.lower().str.strip()
            df_sub["start_time_sec"] = pd.to_numeric(df_sub["start_time_sec"], errors="coerce")
            df_sub["end_time_sec"] = pd.to_numeric(df_sub["end_time_sec"], errors="coerce")
            df_sub.dropna(subset=["start_time_sec", "end_time_sec"], inplace=True)

        if pred_video.empty or gt_video.empty:
            continue

        max_end_gt = gt_video["end_time_sec"].max()
        video_duration_seconds = int(max_end_gt) + 1

        if video_duration_seconds < exclusion_seconds * 2:
            logger.warning(
                "Video '%s' is too short (%ds) for %ds boundary exclusion. Skipping.",
                video_stem,
                video_duration_seconds,
                exclusion_seconds * 2,
            )
            continue

        start_sec = exclusion_seconds
        end_sec = video_duration_seconds - exclusion_seconds

        pred_labels = create_second_level_labels(pred_video, video_duration_seconds)
        gt_labels = create_second_level_labels(gt_video, video_duration_seconds)

        v_total = 0
        v_correct = 0

        for sec in range(start_sec, end_sec):
            gt_label = gt_labels[sec]
            pred_label = pred_labels[sec]

            if gt_label is not None and gt_label != UNCLASSIFIED_LABEL:
                effective_pred = pred_label if pred_label is not None else UNCLASSIFIED_LABEL

                all_gt_seconds.append(str(gt_label).lower())
                all_pred_seconds.append(str(effective_pred).lower())

                v_total += 1
                total_seconds_all += 1

                if gt_label not in category_stats:
                    category_stats[gt_label] = {"total": 0, "correct": 0}
                category_stats[gt_label]["total"] += 1

                if pred_label is not None:
                    confusion_matrix[gt_label][pred_label] += 1
                    if pred_label != gt_label:
                        misclassifications[f"{gt_label} -> {pred_label}"] += 1

                if pred_label == gt_label:
                    v_correct += 1
                    correct_seconds_all += 1
                    category_stats[gt_label]["correct"] += 1

        video_accuracy = v_correct / v_total if v_total > 0 else 0.0
        video_results.append({
            "video_name": video_stem,
            "total_seconds": v_total,
            "correct_seconds": v_correct,
            "accuracy": video_accuracy,
        })

    overall_kappa = (
        float(cohen_kappa_score(all_gt_seconds, all_pred_seconds))
        if all_gt_seconds and all_pred_seconds
        else 0.0
    )
    overall_accuracy = correct_seconds_all / total_seconds_all if total_seconds_all > 0 else 0.0

    category_accuracies = {
        cat: {
            "accuracy": (stats["correct"] / stats["total"]) if stats["total"] > 0 else 0.0,
            "total_seconds": stats["total"],
            "correct_seconds": stats["correct"],
        }
        for cat, stats in category_stats.items()
    }

    return {
        "overall_accuracy": overall_accuracy,
        "overall_kappa": overall_kappa,
        "total_seconds": total_seconds_all,
        "correct_seconds": correct_seconds_all,
        "category_accuracies": category_accuracies,
        "video_results": video_results,
        "confusion_matrix": confusion_matrix,
        "social_state_types": list(category_stats.keys()),
        "misclassifications": misclassifications,
    }


def calculate_detailed_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate precision, recall, support, and F1-score for each class and macro averages."""
    confusion_matrix = results["confusion_matrix"]
    social_state_types = results["social_state_types"]
    detailed_metrics: Dict[str, Any] = {}

    for class_name in social_state_types:
        tp = confusion_matrix[class_name].get(class_name, 0) if class_name in confusion_matrix else 0

        fp = sum(
            confusion_matrix[gt_class].get(class_name, 0)
            for gt_class in confusion_matrix
            if gt_class != class_name
        )

        cat_stats = results.get("category_accuracies", {}).get(class_name, {})
        total_actual = cat_stats.get("total_seconds", 0)
        fn = total_actual - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        detailed_metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": total_actual,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    overall_k = results.get("overall_kappa", 0.0)
    non_macro = [m for k, m in detailed_metrics.items() if isinstance(m, dict)]

    if non_macro:
        detailed_metrics["macro_avg"] = {
            "precision": float(np.mean([m["precision"] for m in non_macro])),
            "recall": float(np.mean([m["recall"] for m in non_macro])),
            "f1_score": float(np.mean([m["f1_score"] for m in non_macro])),
            "kappa": float(overall_k),
        }

    detailed_metrics["overall_kappa"] = float(overall_k)
    return detailed_metrics


def generate_confusion_matrix_plot(results: Dict[str, Any], output_folder: Path) -> Path:
    """Render publication-quality normalized confusion matrix heatmap."""
    confusion_matrix = results["confusion_matrix"]
    social_state_types = results["social_state_types"]

    if len(social_state_types) <= 2:
        labels = ["not interacting", "interacting"]
    else:
        labels = ["alone", "available", "interacting"]

    ordered_labels = [l for l in labels if l in social_state_types]
    n_classes = len(ordered_labels)

    cm_counts_gt_rows = np.array(
        [[confusion_matrix[gt].get(pred, 0) for pred in ordered_labels] for gt in ordered_labels],
        dtype=float,
    )

    row_sums = cm_counts_gt_rows.sum(axis=1, keepdims=True)
    cm_pct_gt_rows = np.divide(
        cm_counts_gt_rows * 100.0,
        row_sums,
        out=np.zeros_like(cm_counts_gt_rows),
        where=row_sums != 0,
    )

    # Transpose so True is on X (Cols) and Predicted is on Y (Rows)
    cm_pct = cm_pct_gt_rows.T
    cm_counts = cm_counts_gt_rows.T.astype(int)

    annot_labels = np.empty((n_classes, n_classes), dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            annot_labels[i, j] = f"{cm_pct[i, j]:.1f}%\n({cm_counts[i, j]:,})"

    display_names = [label.title() for label in ordered_labels]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        cm_pct,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        cbar_kws={"label": "Recall Percentage (%)"},
        xticklabels=display_names,
        yticklabels=display_names,
        annot_kws={"fontsize": 16, "weight": "bold"},
        ax=ax,
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(15)
    cbar.ax.tick_params(labelsize=13)

    ax.set_xlabel("True (Ground Truth)", fontsize=16, fontweight="bold", labelpad=12)
    ax.set_ylabel("Predicted (Pipeline)", fontsize=16, fontweight="bold", labelpad=12)
    ax.tick_params(axis="both", which="major", labelsize=14)

    plt.tight_layout()
    output_folder.mkdir(parents=True, exist_ok=True)
    save_path = output_folder / "social_states_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix heatmap to: %s", save_path)
    return save_path


def save_performance_results(
    results: Dict[str, Any],
    detailed_metrics: Dict[str, Any],
    total_seconds: int,
    total_hours: float,
    filename: Path,
) -> None:
    """Save performance summary and per-class metrics to text file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total seconds analyzed: {total_seconds:,}\n")
        f.write(f"Total duration:         {total_hours:.2f} hours\n\n")

        kappa_val = detailed_metrics.get("overall_kappa", 0.0)
        macro = detailed_metrics.get("macro_avg", {})

        f.write("OVERALL PERFORMANCE METRICS (Macro Average)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Accuracy (second-level):  {results['overall_accuracy']:.4f}\n")
        f.write(f"Cohen's Kappa (κ):        {kappa_val:.4f}\n")
        f.write(f"Macro Precision:          {macro.get('precision', 0.0):.4f}\n")
        f.write(f"Macro Recall:             {macro.get('recall', 0.0):.4f}\n")
        f.write(f"Macro F1-Score:           {macro.get('f1_score', 0.0):.4f}\n\n")

        f.write("CATEGORY-SPECIFIC PERFORMANCE\n")
        f.write("=" * 70 + "\n\n")
        for category, metrics in detailed_metrics.items():
            if category in ["macro_avg", "overall_kappa"] or not isinstance(metrics, dict):
                continue

            stats = results["category_accuracies"].get(category, {})
            f.write(f"{category.upper()}:\n")
            f.write(f"  Support (GT Seconds):    {stats.get('total_seconds', 0):,}\n")
            f.write(f"  Accuracy (second-level): {stats.get('accuracy', 0.0):.4f}\n")
            f.write(f"  Precision:               {metrics.get('precision', 0.0):.4f}\n")
            f.write(f"  Recall:                  {metrics.get('recall', 0.0):.4f}\n")
            f.write(f"  F1-Score:                {metrics.get('f1_score', 0.0):.4f}\n")
            f.write(f"  True Positives:          {metrics.get('true_positives', 0):,}\n")
            f.write(f"  False Positives:         {metrics.get('false_positives', 0):,}\n")
            f.write(f"  False Negatives:         {metrics.get('false_negatives', 0):,}\n\n")

    logger.info("Saved performance summary text to: %s", filename)


def extract_misclassification_segments(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    exclusion_seconds: int = AnalysisConfig.EXCLUSION_SECONDS,
) -> pd.DataFrame:
    """Extract continuous runs of misclassified seconds into coherent event segments."""
    misclassified_segments = []
    pred_work = predictions_df.copy()
    pred_work["stem"] = pred_work["video_name"].apply(normalize_stem)
    gt_work = ground_truth_df.copy()
    gt_work["stem"] = gt_work["video_name"].apply(normalize_stem)

    common_videos = sorted(list(set(pred_work["stem"]).intersection(set(gt_work["stem"]))))

    for video in common_videos:
        pred_video = pred_work[pred_work["stem"] == video].copy()
        gt_video = gt_work[gt_work["stem"] == video].copy()

        max_end_gt = gt_video["end_time_sec"].max() if not gt_video.empty else 0
        video_duration_seconds = int(max_end_gt) + 1

        if video_duration_seconds < exclusion_seconds * 2:
            continue

        start_sec = exclusion_seconds
        end_sec = video_duration_seconds - exclusion_seconds

        pred_labels = create_second_level_labels(pred_video, video_duration_seconds)
        gt_labels = create_second_level_labels(gt_video, video_duration_seconds)

        current_segment = None

        for sec in range(start_sec, end_sec):
            gt_label = gt_labels[sec]
            pred_label = pred_labels[sec]
            pred_label_compare = UNCLASSIFIED_LABEL if pred_label is None else pred_label

            is_misclassified = (gt_label is not None) and (gt_label != pred_label_compare)

            if is_misclassified:
                actual_pred = pred_label if pred_label is not None else UNCLASSIFIED_LABEL
                if current_segment is None:
                    current_segment = {
                        "video_name": video,
                        "start_sec": sec,
                        "end_sec": sec,
                        "gt_label": gt_label,
                        "pred_label": actual_pred,
                    }
                elif (
                    current_segment["gt_label"] == gt_label
                    and current_segment["pred_label"] == actual_pred
                ):
                    current_segment["end_sec"] = sec
                else:
                    misclassified_segments.append(current_segment)
                    current_segment = {
                        "video_name": video,
                        "start_sec": sec,
                        "end_sec": sec,
                        "gt_label": gt_label,
                        "pred_label": actual_pred,
                    }
            else:
                if current_segment is not None:
                    misclassified_segments.append(current_segment)
                    current_segment = None

        if current_segment is not None:
            misclassified_segments.append(current_segment)

    if misclassified_segments:
        df_miss = pd.DataFrame(misclassified_segments)
        df_miss["duration_sec"] = df_miss["end_sec"] - df_miss["start_sec"] + 1
        return df_miss

    return pd.DataFrame()


def run_evaluation(
    predictions_path: Path,
    output_folder: Path,
    mode: str = "tertiary",
    video_list: Optional[List[str]] = None,
    ground_truth_path: Path = AnalysisPaths.GT_SEGMENTS_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load data, execute validation evaluation, and export benchmark outputs.

    Parameters
    ----------
    predictions_path : Path
        Path to predicted segments CSV.
    output_folder : Path
        Destination directory.
    mode : str, default='tertiary'
        Granularity: 'tertiary' or 'binary'.
    video_list : Optional[List[str]], default=None
        Subset list of video names to evaluate.
    ground_truth_path : Path
        Path to ground_truth_segments.csv.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        (predictions_df, ground_truth_df, detailed_metrics)
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        logger.error("Predictions file not found at: %s", predictions_path)
        sys.exit(1)

    if not ground_truth_path.exists():
        logger.error("Ground truth file not found at: %s", ground_truth_path)
        sys.exit(1)

    predictions_df = pd.read_csv(predictions_path)
    ground_truth_df = pd.read_csv(ground_truth_path)

    if mode == "binary":
        predictions_df = reclassify_to_binary(predictions_df)
        ground_truth_df = reclassify_to_binary(ground_truth_df)

    for df in [ground_truth_df, predictions_df]:
        if "start_time_sec" in df.columns:
            df["start_time_sec"] = pd.to_numeric(df["start_time_sec"], errors="coerce")
            df["end_time_sec"] = pd.to_numeric(df["end_time_sec"], errors="coerce")
        elif "start_time_min" in df.columns:
            df["start_time_sec"] = df["start_time_min"].apply(time_to_seconds)
            df["end_time_sec"] = df["end_time_min"].apply(time_to_seconds)

    results = evaluate_performance_by_seconds(
        predictions_df, ground_truth_df, video_subset=video_list
    )
    total_seconds = results["total_seconds"]
    total_hours = total_seconds / 3600.0
    detailed_metrics = calculate_detailed_metrics(results)

    df_misclassified = extract_misclassification_segments(predictions_df, ground_truth_df)
    if video_list and not df_misclassified.empty:
        clean_subset = {normalize_stem(v) for v in video_list}
        df_misclassified = df_misclassified[
            df_misclassified["video_name"].apply(normalize_stem).isin(clean_subset)
        ]

    misclassified_path = output_folder / "misclassified_segments.csv"
    if not df_misclassified.empty:
        df_misclassified.to_csv(misclassified_path, index=False)
        logger.info("Saved misclassified segments to: %s", misclassified_path)

    generate_confusion_matrix_plot(results, output_folder)
    performance_path = output_folder / "performance_summary.txt"
    save_performance_results(
        results, detailed_metrics, total_seconds, total_hours, filename=performance_path
    )

    return predictions_df, ground_truth_df, detailed_metrics


def main() -> None:
    """CLI entrypoint for running validation evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate social interaction predictions against human ground truth"
    )
    parser.add_argument(
        "--predictions_path",
        type=Path,
        default=AnalysisPaths.INTERACTION_SEGMENTS_CSV,
        help="Path to interaction_segments.csv",
    )
    parser.add_argument(
        "--gt_path",
        type=Path,
        default=AnalysisPaths.GT_SEGMENTS_CSV,
        help="Path to ground_truth_segments.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=AnalysisPaths.OUTPUT_FOLDER / "evaluation",
        help="Directory to save evaluation reports and plots",
    )
    parser.add_argument(
        "--mode",
        choices=["binary", "tertiary"],
        default="tertiary",
        help="Evaluation granularity mode (default: tertiary)",
    )
    parser.add_argument(
        "--video_list",
        nargs="+",
        default=None,
        help="Optional list of specific video names to evaluate",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="all",
        default=None,
        help="Plot timeline comparison ('all' for all videos, or specify video stem)",
    )
    args = parser.parse_args()

    predictions_df, ground_truth_df, _ = run_evaluation(
        predictions_path=args.predictions_path,
        output_folder=args.output_dir,
        mode=args.mode,
        video_list=args.video_list,
        ground_truth_path=args.gt_path,
    )

    if args.plot:
        timeline_dir = args.output_dir / "timeline_plots"
        timeline_dir.mkdir(parents=True, exist_ok=True)

        if args.plot.lower() == "all":
            videos = sorted(ground_truth_df["video_name"].unique())
            logger.info("Rendering timeline comparison plots for %d recordings...", len(videos))
            for vid in videos:
                clean_name = normalize_stem(vid)
                out_path = timeline_dir / f"{clean_name}_segment_timeline.png"
                plot_segment_timeline(predictions_df, ground_truth_df, vid, out_path, args.mode)
        else:
            vid = args.plot
            clean_name = normalize_stem(vid)
            out_path = timeline_dir / f"{clean_name}_segment_timeline.png"
            plot_segment_timeline(predictions_df, ground_truth_df, vid, out_path, args.mode)


if __name__ == "__main__":
    main()