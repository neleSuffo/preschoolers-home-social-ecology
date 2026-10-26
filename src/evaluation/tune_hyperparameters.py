"""
src/evaluation/tune_hyperparameters.py

Systematic hyperparameter tuning and sensitivity analysis for multimodal social ecology classification.
Supports Cartesian grid search and memory-efficient random sampling across candidate parameter spaces
while evaluating predictions against human ground truth to prevent data leakage.
"""

import argparse
from datetime import datetime
from itertools import product
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config import AnalysisConfig, HyperparameterConfig
from src.constants import AnalysisPaths, BasePaths
from src.evaluation.eval_segment_performance import run_evaluation
from src.pipeline.frame_level_rules import main as frame_analysis_main
from src.pipeline.segment_aggregation import main as segment_analysis_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_hyperparameter_combinations(
    max_combinations: Optional[int] = 20,
    random_sample: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate unique combinations of hyperparameters from HyperparameterConfig.

    Parameters
    ----------
    max_combinations : Optional[int], default=20
        Maximum number of combinations to sample. If None or random_sample is False,
        all Cartesian product combinations are returned.
    random_sample : bool, default=True
        Whether to sample randomly without replacement or compute the full Cartesian product.

    Returns
    -------
    List[Dict[str, Any]]
        List of configuration dictionaries.
    """
    ranges = HyperparameterConfig.HYPERPARAMETER_RANGES
    param_names = list(ranges.keys())

    if not random_sample:
        all_combos = [
            dict(zip(param_names, c))
            for c in product(*[ranges[p] for p in param_names])
        ]
        return all_combos[:max_combinations] if max_combinations else all_combos

    # Calculate total possible combinations across search space
    total_possible = 1
    for r in ranges.values():
        total_possible *= len(r)

    limit = min(max_combinations or total_possible, total_possible)
    combinations: List[Dict[str, Any]] = []
    seen = set()

    while len(combinations) < limit:
        combo = {p: random.choice(ranges[p]) for p in param_names}
        combo_frozen = tuple(sorted(combo.items()))

        if combo_frozen not in seen:
            seen.add(combo_frozen)
            combinations.append(combo)

    return combinations


def run_pipeline_for_combo(
    hyperparameters: Dict[str, Any],
    combo_dir: Path,
    social_state_mode: str = "tertiary",
    hyperparameter_tuning: bool = True,
    video_list: Optional[List[str]] = None,
) -> Tuple[bool, Optional[Path], Optional[Path], Optional[str]]:
    """
    Execute frame-level rules and PELT change-point smoothing under candidate hyperparameters.

    Parameters
    ----------
    hyperparameters : Dict[str, Any]
        Parameter dictionary overriding AnalysisConfig defaults.
    combo_dir : Path
        Output destination directory for this candidate run.
    social_state_mode : str, default='tertiary'
        Granularity: 'tertiary' or 'binary'.
    hyperparameter_tuning : bool, default=True
        Flag to suppress verbose intermediate pipeline logging.
    video_list : Optional[List[str]], default=None
        Restricts execution to specific video identifiers.

    Returns
    -------
    Tuple[bool, Optional[Path], Optional[Path], Optional[str]]
        (success, frame_output_path, segment_output_path, error_message)
    """
    original_config: Dict[str, Any] = {}

    try:
        # Override AnalysisConfig attributes in memory
        for key, value in hyperparameters.items():
            if hasattr(AnalysisConfig, key):
                original_config[key] = getattr(AnalysisConfig, key)
                setattr(AnalysisConfig, key, value)

        frame_output_path = combo_dir / AnalysisPaths.FRAME_INTERACTIONS_CSV.name
        segment_output_path = combo_dir / AnalysisPaths.INTERACTION_SEGMENTS_CSV.name

        # 1. Vectorized Frame-Level Evaluation
        frame_analysis_main(
            db_path=AnalysisPaths.INFERENCE_DB,
            output_dir=combo_dir,
            video_list=video_list,
            social_state_mode=social_state_mode,
            hyperparameter_tuning=hyperparameter_tuning,
        )

        # 2. PELT Change-Point Temporal Aggregation
        segment_analysis_main(
            output_file_path=segment_output_path,
            frame_data_path=frame_output_path,
            social_state_mode=social_state_mode,
            hyperparameter_tuning=hyperparameter_tuning,
        )

        return True, frame_output_path, segment_output_path, None

    except Exception as e:
        return False, None, None, f"Pipeline execution failed: {str(e)}"

    finally:
        # Restore original AnalysisConfig values
        for key, value in original_config.items():
            setattr(AnalysisConfig, key, value)


def run_analysis_with_config(
    hyperparameters: Dict[str, Any],
    combo_id: int,
    output_base_dir: Path,
    social_state_mode: str = "tertiary",
    hyperparameter_tuning: bool = True,
    video_list: Optional[List[str]] = None,
) -> Tuple[bool, Optional[Path], Optional[Path], Optional[str]]:
    """
    Wrap directory creation, parameter logging, and candidate execution.

    Parameters
    ----------
    hyperparameters : Dict[str, Any]
        Candidate parameter dictionary.
    combo_id : int
        Unique integer index of this combination.
    output_base_dir : Path
        Base directory containing candidate subfolders.
    social_state_mode : str, default='tertiary'
        Granularity: 'tertiary' or 'binary'.
    hyperparameter_tuning : bool, default=True
        Whether running in tuning mode.
    video_list : Optional[List[str]], default=None
        Optional subset of videos to process.

    Returns
    -------
    Tuple[bool, Optional[Path], Optional[Path], Optional[str]]
        (success, frame_output_path, segment_output_path, error_message)
    """
    combo_dir = output_base_dir / f"combo_{combo_id:04d}"
    combo_dir.mkdir(exist_ok=True, parents=True)

    with open(combo_dir / "hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(hyperparameters, f, indent=2)

    return run_pipeline_for_combo(
        hyperparameters=hyperparameters,
        combo_dir=combo_dir,
        social_state_mode=social_state_mode,
        hyperparameter_tuning=hyperparameter_tuning,
        video_list=video_list,
    )


def evaluate_combination(
    segment_output_path: Path,
    social_state_mode: str = "tertiary",
    video_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate predicted segments of a candidate parameter combination against human ground truth.

    Parameters
    ----------
    segment_output_path : Path
        Path to candidate interaction_segments.csv.
    social_state_mode : str, default='tertiary'
        Granularity: 'tertiary' or 'binary'.
    video_list : Optional[List[str]], default=None
        Optional subset of videos to evaluate.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing 'success', 'detailed_metrics', and 'error'.
    """
    try:
        _, _, detailed_metrics = run_evaluation(
            predictions_path=segment_output_path,
            output_folder=segment_output_path.parent,
            mode=social_state_mode,
            video_list=video_list,
            ground_truth_path=AnalysisPaths.GT_SEGMENTS_CSV
        )

        if detailed_metrics:
            return {"success": True, "detailed_metrics": detailed_metrics, "error": None}

        return {"success": False, "detailed_metrics": None, "error": "No metrics returned"}

    except Exception as e:
        return {"success": False, "detailed_metrics": None, "error": str(e)}


def find_best_configuration(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Identify the configuration achieving the highest Macro F1 score."""
    return max(
        results,
        key=lambda x: x["evaluation"]["detailed_metrics"]["macro_avg"]["f1_score"],
    )


def save_final_results(
    all_results: List[Dict[str, Any]],
    best_config: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save parameter grid summary CSV and winning parameter JSON configuration."""
    summary_data = []
    for res in all_results:
        metrics = res["evaluation"]["detailed_metrics"]
        kappa_val = metrics.get("overall_kappa", metrics.get("macro_avg", {}).get("kappa", None))

        row = {
            "combo_id": res["combo_id"],
            "macro_f1": metrics["macro_avg"]["f1_score"],
            "macro_precision": metrics["macro_avg"]["precision"],
            "macro_recall": metrics["macro_avg"]["recall"],
            "cohen_kappa": kappa_val,
            **res["hyperparameters"],
        }
        summary_data.append(row)

    pd.DataFrame(summary_data).to_csv(output_dir / "results_summary.csv", index=False)
    with open(output_dir / "best_configuration.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    logger.info("Saved search summary and best configuration to: %s", output_dir)


def run_tuning_grid(
    max_combinations: Optional[int] = 20,
    video_list: Optional[List[str]] = None,
    social_state_mode: str = "tertiary",
    hyperparameter_tuning: bool = True,
    output_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Main tuning execution loop across candidate parameter combinations.

    Parameters
    ----------
    max_combinations : Optional[int], default=20
        Number of candidate parameter combinations to test.
    video_list : Optional[List[str]], default=None
        List of video IDs for evaluation.
    social_state_mode : str, default='tertiary'
        Granularity: 'tertiary' or 'binary'.
    hyperparameter_tuning : bool, default=True
        Whether running in tuning mode.
    output_dir : Optional[Path], default=None
        Target output directory.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing the winning configuration and evaluation metrics.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_output = output_dir or (BasePaths.OUTPUT_DIR / f"hyperparameter_search_{timestamp}")
    target_output.mkdir(exist_ok=True, parents=True)

    combinations = generate_hyperparameter_combinations(
        max_combinations=max_combinations,
        random_sample=True,
    )
    logger.info("Testing %d parameter combination(s)...", len(combinations))

    all_results = []
    for i, hyperparams in enumerate(combinations, 1):
        logger.info("[%d/%d] Evaluating Combination %d", i, len(combinations), i)

        success, _, seg_out, error = run_analysis_with_config(
            hyperparameters=hyperparams,
            combo_id=i,
            output_base_dir=target_output,
            social_state_mode=social_state_mode,
            hyperparameter_tuning=hyperparameter_tuning,
            video_list=video_list,
        )

        if not success or seg_out is None:
            logger.warning("Combination %d failed: %s", i, error)
            continue

        eval_res = evaluate_combination(
            segment_output_path=seg_out,
            social_state_mode=social_state_mode,
            video_list=video_list,
        )

        if eval_res["success"]:
            all_results.append({
                "combo_id": i,
                "hyperparameters": hyperparams,
                "evaluation": eval_res,
            })
            logger.info("Combination %d Macro F1: %.4f", i, eval_res["detailed_metrics"]["macro_avg"]["f1_score"])

    if all_results:
        best_config = find_best_configuration(all_results)
        save_final_results(all_results, best_config, target_output)
        logger.info("Tuning Complete. Top Macro F1: %.4f", best_config["evaluation"]["detailed_metrics"]["macro_avg"]["f1_score"])
        return best_config

    logger.warning("No valid evaluation combinations recorded.")
    return None


def main() -> None:
    """CLI entrypoint for standalone hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Hyperparameter Search & Evaluation Engine")
    parser.add_argument("--max_combos", type=int, default=20, help="Number of random combinations to test")
    parser.add_argument("--video_list", nargs="+", default=None, help="Subset of video identifiers")
    parser.add_argument("--social_state_mode", choices=["binary", "tertiary"], default="tertiary", help="State granularity")
    args = parser.parse_args()

    run_tuning_grid(
        max_combinations=args.max_combos,
        video_list=args.video_list,
        social_state_mode=args.social_state_mode,
    )


if __name__ == "__main__":
    main()