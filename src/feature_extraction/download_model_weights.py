"""
src/feature_extraction/download_model_weights.py

Downloads fine-tuned model checkpoints from Hugging Face Hub
into the project models directory.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from huggingface_hub import hf_hub_download

from src.constants import BasePaths, ModelAssets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_weights(
    weights_dir: Path = BasePaths.MODELS_DIR,
    repo_id: str = ModelAssets.REPO_ID,
    weights_list: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """
    Download missing model weights from Hugging Face.

    Parameters
    ----------
    weights_dir : Path, default=BasePaths.MODELS_DIR
        Directory where model checkpoint files (.pt) will be stored.
    repo_id : str, default=ModelAssets.REPO_ID
        Hugging Face repository identifier.
    weights_list : Optional[List[str]], default=None
        List of filenames to retrieve. Defaults to ModelAssets.WEIGHTS_LIST.
    force : bool, default=False
        If True, re-downloads and overwrites existing weight files.
    """
    weights_dir.mkdir(parents=True, exist_ok=True)
    targets = weights_list or ModelAssets.WEIGHTS_LIST

    logger.info("Verifying model weights in: %s", weights_dir)
    for filename in targets:
        dest_path = weights_dir / filename

        if dest_path.exists() and not force:
            logger.info("Weight file '%s' already exists. Skipping.", filename)
            continue

        logger.info("Downloading '%s' from Hugging Face repo '%s'...", filename, repo_id)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=weights_dir,
                local_dir_use_symlinks=False,
            )
            logger.info("Saved '%s' to: %s", filename, dest_path)
        except Exception as e:
            logger.error("Failed to download '%s': %s", filename, e)
            raise


def main() -> None:
    """CLI entrypoint for asset downloads."""
    parser = argparse.ArgumentParser(description="Download model weights from Hugging Face")
    parser.add_argument(
        "--weights_dir",
        type=Path,
        default=BasePaths.MODELS_DIR,
        help="Local target directory for weight checkpoints",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=ModelAssets.REPO_ID,
        help="Hugging Face repository ID",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of existing weights",
    )
    args = parser.parse_args()

    download_weights(
        weights_dir=args.weights_dir,
        repo_id=args.repo_id,
        force=args.force,
    )


if __name__ == "__main__":
    main()