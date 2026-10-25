"""
src/constants.py

Consolidated path constants, model assets, and reference anchors for
inference, downstream aggregation, and figure generation.
"""

from pathlib import Path

class BasePaths:
    """Core directory hierarchy."""

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "outputs"
    LOGGING_DIR = OUTPUT_DIR / "logs"
    
class ModelAssets:
    """Pretrained and fine-tuned model checkpoints hosted on Hugging Face."""

    REPO_ID = "NeSuf/preschoolers-home-social-ecology-weights"

    FACE_WEIGHTS_NAME = "yolov12_face_best.pt"
    PERSON_WEIGHTS_NAME = "yolov12_person_best.pt"
    AUDIO_WEIGHTS_NAME = "babyhubert_stc_best.pt"

    WEIGHTS_LIST = [
        FACE_WEIGHTS_NAME,
        PERSON_WEIGHTS_NAME,
        AUDIO_WEIGHTS_NAME,
    ]

    # Local checkpoint paths
    FACE_MODEL_PATH = BasePaths.MODELS_DIR / FACE_WEIGHTS_NAME
    PERSON_MODEL_PATH = BasePaths.MODELS_DIR / PERSON_WEIGHTS_NAME
    AUDIO_MODEL_PATH = BasePaths.MODELS_DIR / AUDIO_WEIGHTS_NAME

    # Voice Type Classifier output
    VTC_RTTM_FILE = BasePaths.DATA_DIR / "diarization_all.rttm"

class PipelineLogs:
    """Progress tracking text files for multi-stage video extraction."""

    PERSON_LOG = BasePaths.LOGGING_DIR / "person_processed.txt"
    FACE_LOG = BasePaths.LOGGING_DIR / "face_processed.txt"
    SPEECH_LOG = BasePaths.LOGGING_DIR / "speech_processed.txt"
    BOOK_LOG = BasePaths.LOGGING_DIR / "book_processed.txt"


class Proximity:
    """Empirical bounding-box area anchors (in pixels) and aspect ratios."""

    REFERENCE_VALUES = {
        "child_ref_close": 458185,
        "child_ref_far": 308,
        "adult_ref_close": 442980,
        "adult_ref_far": 208,
        "child_ref_aspect_ratio": 0.965166908563135,
        "adult_ref_aspect_ratio": 0.6461352657004831,
    }
    
    # Maximum allowed deviation (|observed - reference|) before treating an anomalous aspect ratio as a close-range partial face cut-off
    ASPECT_RATIO_THRESHOLD = 1.0

class AnalysisPaths:
    """Database locations, tabular analysis artifacts, and visualization directories."""

    # Ground truth annotations for social interactions
    GT_SEGMENTS_CSV = BasePaths.DATA_DIR / "ground_truth_segments.csv"
    
    # Participant metadata and video registry
    METADATA_CSV = BasePaths.DATA_DIR / "metadata.csv"
    
    #Primary relational databases
    INFERENCE_DB = BasePaths.DATA_DIR / "inference.db"

    # Input video registry
    VIDEOS_LIST_TUNING = BasePaths.DATA_DIR / "video_list_tuning.txt"
    VIDEO_LIST_INFERENCE = BasePaths.DATA_DIR / "video_list_inference.txt"
    VIDEOS_DIR = BasePaths.DATA_DIR / "videos"

    # Root directory containing per-video subfolders of extracted frames
    # (e.g. data/videos_processed/<video_name>/frame_000010.jpg)
    FRAMES_DIR = BasePaths.DATA_DIR / "videos_processed"
    
    # Pipeline output directory
    OUTPUT_FOLDER = BasePaths.OUTPUT_DIR / "pipeline_outputs"
    HEATMAP_DIR = OUTPUT_FOLDER / "heatmaps"
    SEGMENT_TIMELINE_DIR = OUTPUT_FOLDER / "segment_timeline_plots"

    # Tabular outputs
    FRAME_INTERACTIONS_CSV = OUTPUT_FOLDER / "frame_level_social_interactions.csv"
    INTERACTION_SEGMENTS_CSV = OUTPUT_FOLDER / "interaction_segments.csv"

    KCS_SUMMARY_CSV = OUTPUT_FOLDER / "02_kcs_summary.csv"
    GLOBAL_KCS_SUMMARY_CSV = OUTPUT_FOLDER / "02a_global_kcs_summary.csv"
    CDS_SUMMARY_CSV = OUTPUT_FOLDER / "03_cds_summary.csv"
    GLOBAL_CDS_SUMMARY_CSV = OUTPUT_FOLDER / "03a_global_cds_summary.csv"
    TURN_TAKING_CSV = OUTPUT_FOLDER / "04_turn_taking_summary.csv"
    GLOBAL_TURN_TAKING_CSV = OUTPUT_FOLDER / "04a_global_turn_taking_summary.csv"
    TURN_DURATION_CSV = OUTPUT_FOLDER / "04b_turn_durations_summary.csv"
    TURN_GAPS_CSV = OUTPUT_FOLDER / "04c_turn_gaps_summary.csv"
    CUE_SPECIFICITY_CSV = OUTPUT_FOLDER / "05_cue_specificity.csv"

    # Diagnostic data for figures
    FACE_HEATMAP_PNG = OUTPUT_FOLDER / "face_heatmaps_per_state.png"
    FULL_MATCHED_FACES_CSV = HEATMAP_DIR / "full_matched_faces_detections.csv"
    GT_MATCHED_FACES_CSV = HEATMAP_DIR / "gt_matched_faces_detections.csv"