"""
src/pipeline/pipeline_orchestrator.py

Group-K-Fold cross-validation controller for the social ecology analysis pipeline.
Groups video sessions by Child ID to ensure participant independence across splits,
supporting both standard multi-fold validation and nested hyperparameter optimization.
"""

import argparse
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import AnalysisConfig
from src.constants import AnalysisPaths, BasePaths
from src.evaluation.tune_hyperparameters import (
    evaluate_combination,
    generate_hyperparameter_combinations,
    run_analysis_with_config,
)
from src.research_questions.utils import extract_child_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

NUM_FOLDS = 5


def get_folds(
    video_list_path: Path = AnalysisPaths.VIDEOS_LIST_TUNING,
    n_splits: int = NUM_FOLDS,
) -> Tuple[List[Tuple[Any, Any]], pd.DataFrame]:
    """
    Split annotated benchmark video sessions into group-k-folds partitioned by Child ID.

    Parameters
    ----------
    video_list_path : Path
        Path to text file containing video stems.
    n_splits : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    Tuple[List[Tuple[Any, Any]], pd.DataFrame]
        List of (train_indices, test_indices) tuples and the participant mapping DataFrame.
    """
    if not video_list_path.exists():
        raise FileNotFoundError(f"Video list file not found: {video_list_path}")

    with open(video_list_path, "r", encoding="utf-8") as f:
        videos = [line.strip() for line in f if line.strip()]

    df = pd.DataFrame({"video_name": videos})
    df["child_id"] = df["video_name"].apply(lambda x: extract_child_id(x) or "unknown")

    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(df, groups=df["child_id"]))
    return folds, df


def flatten_metrics(detailed_metrics: Dict[str, Any]) -> Dict[str, float]:
    """Flatten hierarchical evaluation dictionary into a single-level tabular row."""
    flat: Dict[str, float] = {}
    for category, scores in detailed_metrics.items():
        cat_clean = category.lower().replace(" ", "_")

        if not isinstance(scores, dict):
            flat[cat_clean] = float(scores)
            continue

        for metric, value in scores.items():
            metric_clean = metric.replace("_score", "")
            flat[f"{cat_clean}_{metric_clean}"] = float(value)

    return flat


def run_cross_validation(
    mode: str = "validation",
    social_state_mode: str = "tertiary",
    max_combos: int = 20,
    video_list_path: Path = AnalysisPaths.VIDEOS_LIST_TUNING,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Execute participant-independent cross-validation.

    Parameters
    ----------
    mode : str, default='validation'
        'validation' to test fixed baseline parameters, or 'tuning' for nested random search.
    social_state_mode : str, default='tertiary'
        Granularity: 'tertiary' or 'binary'.
    max_combos : int, default=20
        Number of parameter combinations per fold (tuning mode only).
    video_list_path : Path
        Source list of videos.
    output_dir : Optional[Path], default=None
        Destination root directory.
    """
    AnalysisConfig.apply_mode(social_state_mode)

    folds, df = get_folds(video_list_path=video_list_path, n_splits=NUM_FOLDS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = output_dir or (BasePaths.OUTPUT_DIR / f"cv_{mode}_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    fold_results: List[Dict[str, float]] = []

    for i, (train_idx, test_idx) in enumerate(folds, 1):
        train_videos = df.iloc[train_idx]["video_name"].tolist()
        test_videos = df.iloc[test_idx]["video_name"].tolist()

        logger.info("=" * 40)
        logger.info("Executing Fold %d/%d (%d train, %d test videos)", i, NUM_FOLDS, len(train_videos), len(test_videos))
        logger.info("=" * 40)

        if mode == "tuning":
            logger.info("Phase 1: Tuning on %d training videos...", len(train_videos))
            combos = generate_hyperparameter_combinations(max_combinations=max_combos, random_sample=True)

            best_f1 = -1.0
            best_params: Optional[Dict[str, Any]] = None
            tuning_dir = output_root / f"fold_{i}_tuning_search"

            for c_idx, combo in enumerate(combos, 1):
                success, _, seg_path, error = run_analysis_with_config(
                    hyperparameters=combo,
                    combo_id=c_idx,
                    output_base_dir=tuning_dir,
                    social_state_mode=social_state_mode,
                    hyperparameter_tuning=True,
                    video_list=train_videos,
                )

                if success and seg_path:
                    res = evaluate_combination(seg_path, social_state_mode=social_state_mode, video_list=train_videos)
                    if res.get("success") and res.get("detailed_metrics"):
                        current_f1 = res["detailed_metrics"]["macro_avg"].get("f1_score", 0.0)
                        if current_f1 > best_f1:
                            best_f1 = current_f1
                            best_params = combo

            if best_params is None:
                logger.error("Fold %d: Tuning failed across all candidates.", i)
                continue

            logger.info("Fold %d: Winner Macro F1 = %.4f. Validating on test set...", i, best_f1)

            success, _, seg_path, error = run_analysis_with_config(
                hyperparameters=best_params,
                combo_id=i,
                output_base_dir=output_root / f"fold_{i}_final_test",
                social_state_mode=social_state_mode,
                hyperparameter_tuning=True,
                video_list=test_videos,
            )

            if success and seg_path:
                final_res = evaluate_combination(seg_path, social_state_mode=social_state_mode, video_list=test_videos)
                if final_res.get("success") and final_res.get("detailed_metrics"):
                    fold_results.append(flatten_metrics(final_res["detailed_metrics"]))

        else:
            # Standard validation mode under active parameters
            current_params = {
                item: getattr(AnalysisConfig, item)
                for item in dir(AnalysisConfig)
                if not item.startswith("__") and not callable(getattr(AnalysisConfig, item))
            }

            success, _, seg_path, error = run_analysis_with_config(
                hyperparameters=current_params,
                combo_id=i,
                output_base_dir=output_root / f"fold_{i}",
                social_state_mode=social_state_mode,
                hyperparameter_tuning=False,
                video_list=test_videos,
            )

            if success and seg_path:
                eval_res = evaluate_combination(seg_path, social_state_mode=social_state_mode, video_list=test_videos)
                if eval_res.get("success") and eval_res.get("detailed_metrics"):
                    fold_results.append(flatten_metrics(eval_res["detailed_metrics"]))
                else:
                    logger.warning("Fold %d evaluation failed: %s", i, eval_res.get("error"))
            else:
                logger.error("Fold %d pipeline failed: %s", i, error)

    if not fold_results:
        logger.error("No validation results gathered across folds.")
        return

    results_df = pd.DataFrame(fold_results)
    stats = results_df.describe()

    if not stats.empty and "mean" in stats.index:
        summary_rows = stats.loc[["mean", "std", "min", "max"]]
        output_df = pd.concat([results_df, summary_rows])
    else:
        output_df = results_df

    # Record fold composition for provenance
    fold_video_info = []
    for i, (_, test_idx) in enumerate(folds, 1):
        test_videos = df.iloc[test_idx]["video_name"].tolist()
        fold_video_info.append({
            "fold_id": i,
            "num_test_videos": len(test_videos),
            "test_videos": ", ".join(test_videos),
        })

    pd.DataFrame(fold_video_info).to_csv(output_root / "fold_video_info.csv", index=False)
    output_df.to_csv(output_root / "cv_summary.csv", index=True)

    logger.info("Cross-Validation Complete. Summary saved to: %s", output_root / "cv_summary.csv")


def main() -> None:
    """CLI entrypoint for running cross-validation."""
    parser = argparse.ArgumentParser(description="Cross-Validation Orchestration Controller")
    parser.add_argument("--mode", choices=["validation", "tuning"], default="validation", help="Evaluation mode")
    parser.add_argument("--social_state_mode", choices=["binary", "tertiary"], default="tertiary", help="Classification granularity")
    parser.add_argument("--max_combos", type=int, default=20, help="Tuning candidate budget per fold")
    parser.add_argument("--video_list", type=Path, default=AnalysisPaths.VIDEOS_LIST_TUNING, help="Video list file")
    args = parser.parse_args()

    run_cross_validation(
        mode=args.mode,
        social_state_mode=args.social_state_mode,
        max_combos=args.max_combos,
        video_list_path=args.video_list,
    )


if __name__ == "__main__":
    main()