"""
src/pipeline/pipeline_runner.py

Production batch runner executing the full end-to-end social ecology reconstruction:
1. Frame-level multimodal rule evaluation (Rules 1-5).
2. Temporal change-point detection (PELT) and segment smoothing.
3. Downstream behavioral research question extraction (KCS, KCDS, Turn-Taking, Composition) 
across both standard predictions and human ground-truth validation data.
"""

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import List, Optional

from src.constants import AnalysisPaths, BasePaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_command(
    cmd: List[str],
    step_name: str,
    run_folder: Optional[Path] = None,
) -> bool:
    """Execute a subprocess command, stream output, and monitor return status."""
    logger.info("--- Starting: %s ---", step_name)
    env = os.environ.copy()

    # Prepend the virtualenv binary directory to PATH
    venv_bin_dir = str(Path(sys.executable).parent)
    env["PATH"] = f"{venv_bin_dir}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(BasePaths.BASE_DIR)
    
    if run_folder:
        env["RUN_FOLDER"] = str(run_folder)

    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(BasePaths.BASE_DIR),
            text=True,
            env=env,
            encoding="utf-8",
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error("ERROR in %s: Command failed with code %d", step_name, e.returncode)
        sys.exit(1)


def run_pipeline(
    social_state_mode: str = "tertiary",
    video_list: Optional[List[str]] = None,
) -> Path:
    """
    Execute full inference and downstream behavioral analyses.

    Parameters
    ----------
    social_state_mode : str, default='tertiary'
        Granularity: 'tertiary' or 'binary'.
    video_list : Optional[List[str]], default=None
        Optional list of video stems to process.

    Returns
    -------
    Path
        Destination run folder holding generated artifacts.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = BasePaths.OUTPUT_DIR/ f"analysis_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    logger.info("Active Output Directory: %s", run_folder)

    # 1. Step 1: Vectorized Frame-Level Evaluation
    frame_cmd = [
        sys.executable, "-m", "src.pipeline.frame_level_rules",
        "--social_state_mode", social_state_mode,
        "--output_dir", str(run_folder),
    ]
    if video_list:
        frame_cmd.extend(["--video_list"] + video_list)

    run_command(frame_cmd, "Step 1: Frame-Level Multimodal Rule Evaluation", run_folder=run_folder)

    # 2. Step 2: PELT Change-Point Segmentation
    frame_csv = run_folder / AnalysisPaths.FRAME_INTERACTIONS_CSV.name
    segment_csv = run_folder / AnalysisPaths.INTERACTION_SEGMENTS_CSV.name

    segment_cmd = [
        sys.executable, "-m", "src.pipeline.segment_aggregation",
        "--frame_data_path", str(frame_csv),
        "--output_file_path", str(segment_csv),
        "--social_state_mode", social_state_mode,
    ]
    run_command(segment_cmd, "Step 2: PELT Change-Point Segmentation & Temporal Smoothing", run_folder=run_folder)

    # 3. Step 3: Research Question Analyses (Standard vs. Ground Truth Benchmark)
    rq_configs = [
        ("STANDARD PIPELINE", []),
        ("GROUND TRUTH VALIDATION", ["--use_gt"]),
    ]

    for label, gt_flag in rq_configs:
        logger.info("=" * 60)
        logger.info("Executing Behavioral Analyses: %s", label)
        logger.info("=" * 60)

        # RQ 02: Child Vocalization Production (KCS)
        run_command(
            [sys.executable, "-m", "src.research_questions.rq_02_kcs", "--output_folder", str(run_folder)] + gt_flag,
            f"RQ 02: Child Language Production ({label})",
            run_folder=run_folder,
        )

        # RQ 03: Child-Directed Speech Exposure (KCDS)
        run_command(
            [sys.executable, "-m", "src.research_questions.rq_03_kcds", "--output_folder", str(run_folder)] + gt_flag,
            f"RQ 03: Speech Exposure ({label})",
            run_folder=run_folder,
        )

        # RQ 04: Turn-Taking & Conversational Dynamics
        run_command(
            [
                sys.executable, "-m", "src.research_questions.rq_04_turn_taking",
                "--social_state_mode", social_state_mode,
                "--output_folder", str(run_folder),
            ] + gt_flag,
            f"RQ 04: Conversational Dynamics ({label})",
            run_folder=run_folder,
        )

        # RQ 05: Interaction Composition & Cue Specificity
        run_command(
            [
                sys.executable, "-m", "src.research_questions.rq_05_cue_specificity",
                "--social_state_mode", social_state_mode,
                "--output_folder", str(run_folder),
            ] + gt_flag,
            f"RQ 05: Interaction Composition ({label})",
            run_folder=run_folder,
        )

    logger.info("Inference and Behavioral Research Question Analysis Complete.")
    return run_folder


def main() -> None:
    """CLI entrypoint for executing end-to-end analysis runs."""
    parser = argparse.ArgumentParser(description="Full Multimodal Analysis Runner")
    parser.add_argument(
        "--social_state_mode",
        choices=["binary", "tertiary"],
        default="tertiary",
        help="Classification granularity mode",
    )
    parser.add_argument(
        "--video_list",
        nargs="+",
        default=[str(AnalysisPaths.VIDEO_LIST_INFERENCE)],
        help="Path to .txt file of video IDs, a folder of videos, or space-separated video stems",
    )
    args = parser.parse_args()

    resolved_videos = None
    if args.video_list:
        # Check if a single target (file or directory) was provided
        if len(args.video_list) == 1:
            first_target = Path(args.video_list[0])
            if first_target.is_file() and first_target.suffix.lower() == ".txt":
                with open(first_target, "r", encoding="utf-8") as f:
                    resolved_videos = [line.strip() for line in f if line.strip()]
            elif first_target.is_dir():
                resolved_videos = [
                    f.stem for f in first_target.iterdir()
                    if f.is_file() and f.suffix.lower() == ".mp4"
                ]
            else:
                resolved_videos = [first_target.stem]
        else:
            resolved_videos = [Path(v).stem for v in args.video_list]

    run_pipeline(
        social_state_mode=args.social_state_mode,
        video_list=resolved_videos
        )


if __name__ == "__main__":
    main()