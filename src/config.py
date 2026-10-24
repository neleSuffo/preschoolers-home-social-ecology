"""
src/config.py

Runtime hyperparameters, multimodal fusion settings, and grid-search ranges.
"""

from dynaconf import Dynaconf

SETTINGS = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", ".secrets.toml"],
)

class VideoSpecs:
    """Raw egocentric video recording specifications and extraction rates."""

    VIDEO_FILE_EXTENSION = ".mp4"
        # Valid image file extensions
    VALID_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    FPS = 30
    FRAME_STEP_INTERVAL = 10  # 3 Hz analysis rate (every 10th frame at 30 fps)
    FRAME_WIDTH = 2304
    FRAME_HEIGHT = 1296
    HORIZONTAL_FOV_DEG = 140.0

class DetectionConfig:
    """Inference confidence thresholds and proximity distance scaling."""

    # IDs for database storage
    FACE_MODEL_ID = 1
    PERSON_MODEL_ID = 2
    AUDIO_MODEL_ID = 3
    BOOK_MODEL_ID = 4
    
    # Confidence thresholds for YOLOv12 face and person detection models
    PERSON_CONFIDENCE_THRESHOLD = 0.170
    FACE_CONFIDENCE_THRESHOLD = 0.25

    # Valid audio classification labels for the pretrained BabyHuBERT model
    AUDIO_CLASSES = ['OHS', 'KCDS', 'KCHI']
    AUDIO_CLASS_DESCRIPTIONS = {
        "KCHI": "key_child",
        "KCDS": "child_directed_speech",
        "OHS": "overheard_speech",
    }

    # Book detection model and category IDs
    BOOK_MODEL_NAME = "yolo26x.pt"
    BOOK_CATEGORY_IDS = [73]

    # Empirical proximity conversion boundaries
    PROXIMITY_MIN_DISTANCE_M = 0.30  # Proximity 1.0 corresponds to ~0.3 m
    PROXIMITY_MAX_DISTANCE_M = 5.00  # Proximity 0.0 corresponds to ~5.0 m
    PLOT_HORIZON_METERS = 3.0        # Truncation limit for proximity visual plots

class AnalysisConfig:
    """Fixed winning classification rules and change-point segmentation parameters."""

    EXCLUSION_SECONDS = 30  # Margin excluded at recording boundaries
    SAMPLE_RATE = 10        # 3 Hz sampling step

    # Primary 3-state configuration (Interacting, Available, Alone)
    TERTIARY_PARAMETERS = {
        "SOCIAL_STATE_MAPPING": {1: "Interacting", 2: "Available", 3: "Alone"},
        "GAP_DEFAULT_LABEL": "Alone",
        # Rule 1: Turn-Taking
        "MAX_TURN_TAKING_GAP_SEC": 5.0,
        "MAX_SAME_SPEAKER_GAP_SEC": 1.0,
        # Gating & Visual Presence
        "AUDIO_VISUAL_GATING_FLOOR": 0.35,
        "PRESENCE_SCORE_FRAMES_PER_SEC": 3,
        "INSTANT_CONFIDENCE_THRESHOLD": 0.15,
        # Rule 2: Physical Proximity
        "PROXIMITY_THRESHOLD": 0.60,
        # Rule 3: Sustained KCDS
        "SUSTAINED_KCDS_WINDOW_SEC": 10.0,
        "SUSTAINED_KCDS_THRESHOLD": 0.95,
        # Rule 4: Visual Persistence
        "HIGH_CONFIDENCE_PROXIMITY_THRESHOLD": 0.65,
        "HIGH_CONFIDENCE_FACE_CONFIDENCE": 0.85,
        "VISUAL_PERSISTENCE_SEC": 3.0,
        "SHORT_TERM_VISUAL_MEMORY_SEC": 3.5,
        # Rule 5: Sustained OHS
        "SUSTAINED_OHS_WINDOW_SEC": 10.0,
        "SUSTAINED_OHS_THRESHOLD": 0.65,
        # Segment-Level PELT & Assignment
        "CPD_PENALTY": 5.0,
        "CPD_INTERACTING_THRESHOLD": 0.40,
        "CPD_TOTAL_PRESENCE_FLOOR": 0.45,
        # Post-Processing & Temporal Smoothing
        "SAME_SEGMENT_MERGE_THRESHOLD": 3.0,
        "GAP_STRETCH_THRESHOLD": 2.0,
    }

    # Binary classification baseline (Interacting vs. Not Interacting)
    BINARY_PARAMETERS = {
        "SOCIAL_STATE_MAPPING": {1: "Interacting", 2: "Not_Interacting"},
        "GAP_DEFAULT_LABEL": "Not_Interacting",
        "MAX_TURN_TAKING_GAP_SEC": 5.0,
        "MAX_SAME_SPEAKER_GAP_SEC": 1.0,
        "AUDIO_VISUAL_GATING_FLOOR": 0.45,
        "PRESENCE_SCORE_FRAMES_PER_SEC": 3,
        "INSTANT_CONFIDENCE_THRESHOLD": 0.60,
        "PROXIMITY_THRESHOLD": 0.65,
        "SUSTAINED_KCDS_WINDOW_SEC": 18.0,
        "SUSTAINED_KCDS_THRESHOLD": 0.98,
        "HIGH_CONFIDENCE_PROXIMITY_THRESHOLD": 0.90,
        "HIGH_CONFIDENCE_FACE_CONFIDENCE": 0.80,
        "VISUAL_PERSISTENCE_SEC": 1.0,
        "SHORT_TERM_VISUAL_MEMORY_SEC": 2.5,
        "SUSTAINED_OHS_WINDOW_SEC": 10.0,
        "SUSTAINED_OHS_THRESHOLD": 0.65,
        "CPD_PENALTY": 3.0,
        "CPD_INTERACTING_THRESHOLD": 0.95,
        "CPD_TOTAL_PRESENCE_FLOOR": 0.30,
        "SAME_SEGMENT_MERGE_THRESHOLD": 1.5,
        "GAP_STRETCH_THRESHOLD": 1.5,
    }

    @classmethod
    def apply_mode(cls, mode: str = "tertiary") -> None:
        """Hot-swap active pipeline parameters based on target classification mode."""
        params = cls.BINARY_PARAMETERS if mode == "binary" else cls.TERTIARY_PARAMETERS
        for key, value in params.items():
            setattr(cls, key, value)


class HyperparameterConfig:
    """
    Parameter search grid for sensitivity analysis and hyperparameter tuning.

    You can add candidate values to any list below to evaluate
    alternative thresholds or re-optimize parameters for other datasets.
    """

    HYPERPARAMETER_RANGES = {
        # --- 1. Turn-Taking & Acoustic Gaps ---
        "MAX_TURN_TAKING_GAP_SEC": [5.0],        # e.g., [3.0, 5.0, 7.0]
        "MAX_SAME_SPEAKER_GAP_SEC": [1.0],       # e.g., [0.8, 1.0, 1.5]

        # --- 2. Audio-Visual Gating & Visual Confidence ---
        "AUDIO_VISUAL_GATING_FLOOR": [0.35],     # e.g., [0.25, 0.35, 0.45]
        "PRESENCE_SCORE_FRAMES_PER_SEC": [3],    # e.g., [3, 6, 9]
        "INSTANT_CONFIDENCE_THRESHOLD": [0.15],  # e.g., [0.15, 0.25, 0.35]

        # --- 3. Physical Proximity & Face Anchors ---
        "PROXIMITY_THRESHOLD": [0.60],           # e.g., [0.55, 0.60, 0.65]
        "HIGH_CONFIDENCE_PROXIMITY_THRESHOLD": [0.65],
        "HIGH_CONFIDENCE_FACE_CONFIDENCE": [0.85],

        # --- 4. Speech Stream Gating (KCDS & OHS) ---
        "SUSTAINED_KCDS_WINDOW_SEC": [10.0],     # e.g., [6.0, 8.0, 10.0]
        "SUSTAINED_KCDS_THRESHOLD": [0.95],      # e.g., [0.90, 0.95, 1.00]
        "SUSTAINED_OHS_WINDOW_SEC": [10.0],      # e.g., [8.0, 10.0, 12.0]
        "SUSTAINED_OHS_THRESHOLD": [0.65],       # e.g., [0.60, 0.65, 0.70]

        # --- 5. Visual Persistence & Short-Term Memory ---
        "VISUAL_PERSISTENCE_SEC": [3.0],         # e.g., [2.0, 3.0, 4.0]
        "SHORT_TERM_VISUAL_MEMORY_SEC": [3.5],   # e.g., [2.5, 3.5, 4.5]

        # --- 6. PELT Change-Point Detection & State Assignment ---
        "CPD_PENALTY": [5.0],                    # e.g., [3.0, 5.0, 7.0]
        "CPD_INTERACTING_THRESHOLD": [0.40],     # e.g., [0.30, 0.40, 0.50]
        "CPD_TOTAL_PRESENCE_FLOOR": [0.45],      # e.g., [0.40, 0.45, 0.50]

        # --- 7. Temporal Smoothing & Post-Processing ---
        "SAME_SEGMENT_MERGE_THRESHOLD": [3.0],   # e.g., [2.0, 3.0, 4.0]
        "GAP_STRETCH_THRESHOLD": [2.0],          # e.g., [1.0, 2.0, 3.0]
    }