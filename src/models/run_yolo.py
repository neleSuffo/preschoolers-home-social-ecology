"""
src/models/run_yolo.py

Unified inference and visualization script for YOLO person and face detectors.

This module consolidates inference, bounding box extraction, ground-truth IoU
evaluation (marking TP, FP, and FN), continuous face proximity computation,
and optional face crop exports into a single CLI tool.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from supervision import Detections
from ultralytics import YOLO

from src.constants import BasePaths, ModelAssets
from src.models.proximity import calculate_proximity

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual Styling Constants (OpenCV BGR Format)
# ---------------------------------------------------------------------------
BOX_THICKNESS: int = 8          # Main bounding box stroke width in pixels
BACKING_OFFSET: int = 6         # Extra border width for white high-contrast outline
BADGE_PAD: int = 8              # Internal padding around label text (in pixels)
TEXT_THICKNESS: int = 4         # Stroke thickness of rendered text
DEFAULT_FONT_SCALE: float = 1.8 # Relative OpenCV font scale

# Color definitions in BGR space
COLOR_TP: Tuple[int, int, int] = (40, 160, 40)    # Vivid Green for True Positives
COLOR_GT: Tuple[int, int, int] = (180, 140, 0)    # Cyan/Teal for matched Ground Truth
COLOR_ERR: Tuple[int, int, int] = (0, 70, 230)    # High-contrast Red-Orange for FP & FN
COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)


def process_image(model: YOLO, image_path: Path) -> Tuple[np.ndarray, Detections]:
    """
    Load an image from disk and execute YOLO object detection inference.

    Parameters
    ----------
    model : YOLO
        Loaded Ultralytics YOLO model instance.
    image_path : Path
        Filesystem path to the input image file.

    Returns
    -------
    Tuple[np.ndarray, Detections]
        A 2-tuple containing:
        - The source image as a NumPy BGR array (H, W, C).
        - A supervision Detections object containing coordinates (.xyxy),
          confidences (.confidence), and predicted class IDs (.class_id).

    Raises
    ------
    ValueError
        If OpenCV fails to read the image file at the provided path.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image from {image_path}")

    # Pass PIL image to Ultralytics YOLO inference
    output = model(Image.open(image_path), verbose=False)
    results = Detections.from_ultralytics(output[0])
    return image, results


def draw_dashed_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 10,
    gap_length: int = 12,
) -> None:
    """
    Draw a rectangular bounding box with dashed edges directly onto an image array.

    Used specifically to visualize False Negatives (missed ground truth annotations).

    Parameters
    ----------
    img : np.ndarray
        Target image array modified in-place.
    pt1 : Tuple[int, int]
        Top-left vertex coordinates (x1, y1).
    pt2 : Tuple[int, int]
        Bottom-right vertex coordinates (x2, y2).
    color : Tuple[int, int, int]
        BGR color tuple for the dashes.
    thickness : int, default=2
        Pixel stroke thickness of each dash segment.
    dash_length : int, default=10
        Pixel length of each solid dash segment.
    gap_length : int, default=12
        Pixel spacing between consecutive dash segments.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Define the 4 perimeter segments of the rectangle
    lines = [
        ((x1, y1), (x2, y1)),  # Top horizontal edge
        ((x2, y1), (x2, y2)),  # Right vertical edge
        ((x2, y2), (x1, y2)),  # Bottom horizontal edge
        ((x1, y2), (x1, y1)),  # Left vertical edge
    ]

    for start, end in lines:
        dist = np.hypot(end[0] - start[0], end[1] - start[1])
        if dist <= 0:
            continue

        dashes = int(dist / (dash_length + gap_length))
        for i in range(dashes + 1):
            s = i * (dash_length + gap_length)
            e = min(s + dash_length, dist)
            if s >= dist:
                break

            p1 = (
                int(start[0] + (end[0] - start[0]) * (s / dist)),
                int(start[1] + (end[1] - start[1]) * (s / dist)),
            )
            p2 = (
                int(start[0] + (end[0] - start[0]) * (e / dist)),
                int(start[1] + (end[1] - start[1]) * (e / dist)),
            )
            cv2.line(img, p1, p2, color, thickness)


def calculate_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """
    Compute the Intersection-over-Union (IoU) overlap between two bounding boxes.

    Parameters
    ----------
    box_a : Sequence[float]
        First bounding box coordinates formatted as [x1, y1, x2, y2].
    box_b : Sequence[float]
        Second bounding box coordinates formatted as [x1, y1, x2, y2].

    Returns
    -------
    float
        Calculated IoU score bounded within [0.0, 1.0].
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return float(inter_area / union_area) if union_area > 0.0 else 0.0


def load_ground_truth(
    label_path: Path, img_width: int, img_height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a standard YOLO-format text annotation file into absolute pixel coordinates.

    Parameters
    ----------
    label_path : Path
        Path to the corresponding ground truth `.txt` file containing space-separated
        records of format: `<class_id> <x_center> <y_center> <width> <height>`.
    img_width : int
        Pixel width of the target frame image.
    img_height : int
        Pixel height of the target frame image.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A 2-tuple containing:
        - Bounding boxes as an (N, 4) array in [x1, y1, x2, y2] pixel format.
        - Class IDs as an (N,) integer array.
    """
    gt_boxes: List[List[int]] = []
    gt_classes: List[int] = []

    if not label_path.exists():
        return np.empty((0, 4), dtype=int), np.empty((0,), dtype=int)

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id, x_center, y_center, w, h = map(float, parts[:5])
            x1 = int((x_center - w / 2.0) * img_width)
            y1 = int((y_center - h / 2.0) * img_height)
            x2 = int((x_center + w / 2.0) * img_width)
            y2 = int((y_center + h / 2.0) * img_height)

            gt_boxes.append([x1, y1, x2, y2])
            gt_classes.append(int(class_id))

    return np.array(gt_boxes, dtype=int), np.array(gt_classes, dtype=int)


def draw_visualizations(
    image: np.ndarray,
    predictions: Detections,
    task: str,
    ground_truth_boxes: Optional[np.ndarray] = None,
    proximities: Optional[List[float]] = None,
    print_proximity: bool = False,
    iou_threshold: float = 0.3,
) -> np.ndarray:
    """
    Render overlaid bounding boxes, category text badges, and proximity indicators.

    Handles high-contrast white background outlines and boundary clamping
    to prevent text from rendering outside the frame edges.

    Parameters
    ----------
    image : np.ndarray
        Source BGR image array.
    predictions : Detections
        Predicted detections returned by YOLO inference.
    task : str
        Target model domain: 'face' or 'person'.
    ground_truth_boxes : Optional[np.ndarray], default=None
        Array of ground truth bounding boxes in [x1, y1, x2, y2] format.
    proximities : Optional[List[float]], default=None
        Normalized continuous proximity values corresponding to predicted boxes.
    print_proximity : bool, default=False
        Whether to append the proximity score to the badge label (faces only).
    iou_threshold : float, default=0.3
        IoU threshold to classify a detection as True Positive vs. False Positive.

    Returns
    -------
    np.ndarray
        A copy of the image with all visual overlays rendered.
    """
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    target_name = "Face" if task == "face" else "Person"

    # -----------------------------------------------------------------------
    # 1. Ground Truth Visualizations
    # -----------------------------------------------------------------------
    if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
        for gt_box in ground_truth_boxes:
            x1, y1, x2, y2 = map(int, gt_box)

            # Check if any model prediction matches this ground truth annotation
            max_iou = 0.0
            if predictions is not None and len(predictions.xyxy) > 0:
                iou_scores = [calculate_iou(gt_box, p) for p in predictions.xyxy]
                max_iou = max(iou_scores) if iou_scores else 0.0

            is_fn = max_iou < iou_threshold

            if is_fn:
                # Missed detection: Dashed Red-Orange box
                label = f"FN: GT {target_name}"
                color = COLOR_ERR
                draw_dashed_rect(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    COLOR_WHITE,
                    thickness=BOX_THICKNESS + BACKING_OFFSET,
                )
                draw_dashed_rect(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    color,
                    thickness=BOX_THICKNESS,
                )
            else:
                # Successfully recalled detection: Solid Cyan/Teal box
                label = f"GT {target_name}"
                color = COLOR_GT
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    COLOR_WHITE,
                    BOX_THICKNESS + BACKING_OFFSET,
                )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            # Measure text dimensions for solid white backing badge
            (tw, th), base = cv2.getTextSize(label, font, DEFAULT_FONT_SCALE, TEXT_THICKNESS)
            bx = min(x1 + 4, img_w - tw - BADGE_PAD - 6)
            by = y1 - base - BADGE_PAD - BOX_THICKNESS

            # Flip badge inside top of box if it exceeds the frame ceiling
            if by - th - BADGE_PAD < 0:
                by = y1 + th + BADGE_PAD + BOX_THICKNESS + 4

            # Draw white background rectangle and rendered label
            cv2.rectangle(
                annotated,
                (bx - BADGE_PAD, by - th - BADGE_PAD),
                (bx + tw + BADGE_PAD, by + base + BADGE_PAD),
                COLOR_WHITE,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (bx, by),
                font,
                DEFAULT_FONT_SCALE,
                color,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )

    # -----------------------------------------------------------------------
    # 2. Predicted Bounding Boxes
    # -----------------------------------------------------------------------
    if predictions is not None and len(predictions.xyxy) > 0:
        for i, (bbox, conf) in enumerate(zip(predictions.xyxy, predictions.confidence)):
            x1, y1, x2, y2 = map(int, bbox)

            # Determine whether prediction matches ground truth
            max_iou = 0.0
            if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
                iou_scores = [calculate_iou(bbox, gt) for gt in ground_truth_boxes]
                max_iou = max(iou_scores) if iou_scores else 0.0

            is_fp = (
                (ground_truth_boxes is None)
                or (len(ground_truth_boxes) == 0)
                or (max_iou < iou_threshold)
            )
            pred_color = COLOR_ERR if is_fp else COLOR_TP
            prefix = "FP" if is_fp else "TP"
            label = f"{prefix}: {target_name} ({conf:.2f})"

            # Append proximity score if requested for face evaluation
            if task == "face" and print_proximity and proximities and i < len(proximities):
                label += f", prox: {proximities[i]:.2f}"

            # Draw white backing border then solid colored bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                COLOR_WHITE,
                BOX_THICKNESS + BACKING_OFFSET,
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), pred_color, BOX_THICKNESS)

            (tw, th), base = cv2.getTextSize(label, font, DEFAULT_FONT_SCALE, TEXT_THICKNESS)

            # Handle edge crops: flip label horizontally if box hugs the left screen edge
            box_w = x2 - x1
            if x1 < 40 and box_w < tw:
                bx = x2 + BOX_THICKNESS + BADGE_PAD + 6
                by = int((y1 + y2) / 2) + int(th / 2)
                by = max(th + BADGE_PAD + 4, min(by, img_h - base - BADGE_PAD - 4))
            else:
                bx = max(BADGE_PAD + 4, min(x1 + 4, img_w - tw - BADGE_PAD - 6))
                by = y2 + th + BADGE_PAD + BOX_THICKNESS + 4
                # Flip above box bottom if it exceeds the frame floor
                if by + base + BADGE_PAD > img_h:
                    by = y2 - base - BADGE_PAD - BOX_THICKNESS - 4

            # Draw white background box and text
            cv2.rectangle(
                annotated,
                (bx - BADGE_PAD, by - th - BADGE_PAD),
                (bx + tw + BADGE_PAD, by + base + BADGE_PAD),
                COLOR_WHITE,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (bx, by),
                font,
                DEFAULT_FONT_SCALE,
                pred_color,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )

    return annotated


def process_file(
    image_path: Path,
    output_dir: Path,
    model: YOLO,
    task: str,
    cut_face: bool = False,
    print_proximity: bool = False,
    labels_dir: Optional[Path] = None,
) -> None:
    """
    Execute detection pipeline on a single image and export crops or visualizations.

    Parameters
    ----------
    image_path : Path
        Path to the target input image.
    output_dir : Path
        Destination directory for exported outputs.
    model : YOLO
        Loaded Ultralytics YOLO model.
    task : str
        Domain task: 'face' or 'person'.
    cut_face : bool, default=False
        If True, saves isolated bounding box face crops as PNGs instead of full overlays.
    print_proximity : bool, default=False
        If True, shows normalized physical proximity metrics on bounding boxes.
    labels_dir : Optional[Path], default=None
        Optional directory containing corresponding ground truth `.txt` files.
    """
    image, results = process_image(model, image_path)
    img_h, img_w = image.shape[:2]

    # Load corresponding ground truth annotations if available
    gt_boxes = None
    if labels_dir is not None:
        label_file = labels_dir / f"{image_path.stem}.txt"
        if label_file.exists():
            gt_boxes, _ = load_ground_truth(label_file, img_w, img_h)

    # Compute continuous proximity metrics for face detections
    proximities: List[float] = []
    if task == "face" and len(results.xyxy) > 0:
        for bbox, class_id in zip(results.xyxy, results.class_id):
            prox = calculate_proximity(bbox, class_id=int(class_id))
            proximities.append(prox if prox is not None else 0.0)

    # Export face crops if requested
    if task == "face" and cut_face:
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, (bbox, prox) in enumerate(zip(results.xyxy, proximities)):
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            face_crop = image[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            crop_path = output_dir / f"{image_path.stem}_face_{i + 1}.png"
            cv2.imwrite(str(crop_path), face_crop)
            logger.info("Saved face crop: %s (proximity=%.2f)", crop_path.name, prox)
        return

    # Check whether the frame contains any signals worth saving
    has_detections = len(results.xyxy) > 0
    has_gt = gt_boxes is not None and len(gt_boxes) > 0

    if has_detections or has_gt:
        annotated = draw_visualizations(
            image=image,
            predictions=results,
            task=task,
            ground_truth_boxes=gt_boxes,
            proximities=proximities,
            print_proximity=print_proximity,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info("Saved: %s", out_path)
    else:
        logger.info("Skipped %s: No detections or annotations.", image_path.name)


def main() -> int:
    """
    Parse command-line arguments and run YOLO detection workflow across images.

    Returns
    -------
    int
        Exit code: 0 on success, non-zero on failure.
    """
    parser = argparse.ArgumentParser(
        description="Unified YOLO Person & Face Detection Inference"
    )
    parser.add_argument(
        "--task",
        choices=["face", "person"],
        required=True,
        help="Detection task: 'face' or 'person'",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to an image file or directory of images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory (defaults to outputs/<task>_detections)",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="Optional path to directory containing YOLO-format ground truth .txt files",
    )
    parser.add_argument(
        "--cut_face",
        action="store_true",
        help="Extract and save cropped face regions as PNGs (face task only)",
    )
    parser.add_argument(
        "--print_proximity",
        action="store_true",
        help="Render estimated proximity value on bounding box badges (face task only)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return 1

    # Select appropriate checkpoint from ModelAssets constants
    if args.task == "face":
        weights_path = ModelAssets.FACE_MODEL_PATH
        default_out = BasePaths.OUTPUT_DIR / "face_detections"
    else:
        weights_path = ModelAssets.PERSON_MODEL_PATH
        default_out = BasePaths.OUTPUT_DIR / "person_detection"

    if not weights_path.exists():
        logger.error(
            "Model checkpoint not found at %s. Run download_assets.py first.",
            weights_path,
        )
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else default_out
    labels_dir = Path(args.labels_dir) if args.labels_dir else None

    logger.info("Loading YOLO %s model from: %s", args.task, weights_path)
    model = YOLO(str(weights_path))

    valid_exts = {".jpg", ".jpeg", ".png", ".PNG"}

    if input_path.is_dir():
        image_files = [p for p in input_path.iterdir() if p.suffix in valid_exts]
        target_dir = out_dir / f"{input_path.name}_annotated"
        logger.info("Found %d image(s) in %s", len(image_files), input_path)
        for img_path in sorted(image_files):
            process_file(
                image_path=img_path,
                output_dir=target_dir,
                model=model,
                task=args.task,
                cut_face=args.cut_face,
                print_proximity=args.print_proximity,
                labels_dir=labels_dir,
            )
    else:
        process_file(
            image_path=input_path,
            output_dir=out_dir,
            model=model,
            task=args.task,
            cut_face=args.cut_face,
            print_proximity=args.print_proximity,
            labels_dir=labels_dir,
        )

    logger.info("Inference run complete.")
    return 0


if __name__ == "__main__":
    exit(main())