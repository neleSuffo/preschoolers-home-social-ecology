from collections import defaultdict
from dynaconf import Dynaconf
from pathlib import Path
import numpy as np

# Dynaconf settings
SETTINGS = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", ".secrets.toml"],
)

# General Preprocessing and Data Configuration
class DataConfig:
    """General configuration for data processing and labels."""
    VIDEO_FILE_EXTENSION = ".mp4"
    FRAME_STEP_INTERVAL = 10
    EXTRACTION_FPS = 1
    TRAIN_SPLIT_RATIO = 0.7
    RANDOM_SEED = 42
    FPS = 30
    FRAME_WIDTH = 2304
    FRAME_HEIGHT = 1296
    HORIZONTAL_FOV_DEG = 140.0 #wide lens
    VIDEO_BATCH_SIZE = 16
    VALID_EXTENSIONS = [".jpg", ".PNG"]
    CUT_VIDEO = ['quantex_at_home_id255237_2022_05_08_04']
    CUT_VIDEO_OFFSET = 7123 #add offset to frame number for this video
    SHIFTED_VIDEOS_OFFSETS = {'quantex_at_home_id257511_2021_07_13_01': (24, -6), # annotations are shifted by six frames from frame 30 on
                              'quantex_at_home_id257573_2021_04_02_01': (28, -2), #annotations are shifted by two frames from frame 28 on
                              'quantex_at_home_id257578_2021_05_12_04': (28, -2), #annotations are shifted by two frames from frame 28 on
                              'quantex_at_home_id257578_2021_05_12_05': (28, -2),
                              'quantex_at_home_id257578_2021_05_12_06': (28, -2),
                              'quantex_at_home_id257578_2021_05_16_01': (28, -2)} #annotations are shifted by two frames from frame 28 on
    NON_STANDARD_FRAME_STEPS = {'quantex_at_home_id258704_2022_05_15_01': 34} # Interval is 34 frames, not 30
    

class LabelMapping:
    """Mappings for labels, IDs, and supercategories."""
    LABEL_TO_ID_MAPPING = defaultdict(
        lambda: 99,
        {
            "person": 1,
            "reflection": 2,
            "book": 3,
            "animal": 4,
            "toy": 5,
            "kitchenware": 6,
            "screen": 7,
            "food": 8,
            "object": 9,
            "other_object": 12,
            "face": 10,
            'child_body_parts': 11,
            "voice": 20,
            "noise": -1,
        },
    )

    ID_TO_SUPERCATEGORY_MAPPING = defaultdict(
        lambda: "unknown",
        {
            1: "person",
            2: "reflection",
            3: "object",
            4: "object",
            5: "object",
            6: "object",
            7: "object",
            8: "object",
            9: "object",
            12: "object",
            10: "face",
            20: "voice",
            -1: "noise",
            99: "unknown",
        },
    )
    unknown_label_id = -1
    unknown_supercategory = "unknown"

class BookConfig:
    """Configuration for book detection."""
    MODEL_NAME = "yolo26x"
    DATABASE_CATEGORY_IDS = [73]  # COCO category ID for 'book'
    MODEL_ID = 5
    
# Specific Task Configurations
class PersonConfig:
    """Configuration for person detection and classification."""
    MODEL_SIZE = 'l'
    MODEL_NAME = f"yolo12{MODEL_SIZE}"
    # Ratio of training data to use for training
    TRAIN_SPLIT_RATIO = 0.6
    # Ratio of class-to-class samples in each dataset split
    MAX_CLASS_RATIO_THRESHOLD = 0.9
    AGE_GROUP_TO_CLASS_ID_PERSON_ONLY = {
        'Inf': 0,
        'Child': 0,
        'Teen': 0,
        'Adult': 0,
        'Dk': 0,
    }
    AGE_GROUP_TO_CLASS_ID_AGE_BINARY = {
        'Inf': 0,
        'Child': 0,
        'Teen': 1,
        'Adult': 1,
    }
    MODEL_CLASS_ID_TO_LABEL_AGE_BINARY = {
        0: "child",
        1: "adult"
    }
    MODEL_CLASS_ID_TO_LABEL_PERSON_ONLY = {0: "person"}
    DATABASE_CATEGORY_IDS = [1, 2]
    TARGET_LABELS_AGE_BINARY = ['child', 'adult']
    TARGET_LABELS_PERSON_ONLY = ['person']
    TARGET_LABELS_CLS = ['0_no_person', '1_person']
    NEGATIVE_SAMPLING_RATIO = 1 #100% negative samples compared to positive samples in train and val splits

    MIN_IDS_PER_SPLIT = 2

    NUM_EPOCHS = 300
    BATCH_SIZE = 64
    IMG_SIZE = 640
    PATIENCE = 25
    NUM_WORKERS = 24
    DEGREES = 15

    # number of videos per batch
    LR = 1e-3
    FREEZE_CNN = True
    SEQUENCE_LENGTH = 60
    DROPOUT = 0.5
    WEIGHT_DECAY = 1e-5
    FEAT_DIM = 512
    RNN_HIDDEN = 256
    RNN_LAYERS = 2
    BIDIRECTIONAL = True
    NUM_OUTPUTS = 1
    BACKBONE = 'efficientnet_b0'
    BATCH_SIZE_INFERENCE = 64
    WINDOW_SIZE = 6 # frames (6 extracted frames = 60 original frames = 2 seconds)
    STRIDE = 1 # frames (every 1st extracted frame = every 10 original frames)
    CHILD_POS_WEIGHT = 3.08  
    ADULT_POS_WEIGHT = 2.60
    MODEL_ID = 2
    CONFIDENCE_THRESHOLD = 0.170
    
class FaceConfig:
    """Configuration for face detection and classification."""
    MODEL_SIZE = 'l'
    MODEL_NAME = f"yolo12{MODEL_SIZE}"
    AGE_GROUP_TO_CLASS_ID_AGE_BINARY = {
        'infant': 0,
        'child': 0,
        'teen': 1,
        'adult': 1,
    }
    AGE_GROUP_TO_CLASS_ID_FACE_ONLY = {
        'infant': 0,
        'child': 0,
        'teen': 0,
        'adult': 0,
    }
    MODEL_CLASS_ID_TO_LABEL_AGE_BINARY = {
        0: "child",
        1: "adult"
    }
    MODEL_CLASS_ID_TO_LABEL_FACE_ONLY = {0: "face"}
    DATABASE_CATEGORY_IDS = [10]
    TARGET_LABELS_AGE_BINARY = ['child', 'adult']
    TARGET_LABELS_FACE_ONLY = ['face']
    NEGATIVE_SAMPLING_RATIO = 1 #100% negative samples compared to positive samples in train and val splits

    TRAIN_SPLIT_RATIO = 0.6
    MIN_IDS_PER_SPLIT = 2
    
    NUM_EPOCHS = 300
    BATCH_SIZE = 64
    IMG_SIZE = 640
    PATIENCE = 30
    MODEL_ID = 1
    NUM_WORKERS = 12
    DEGREES = 15
        
    PROXIMITY_MIN_DISTANCE = 0.30 # proximity of 1 corresponds to 0.3m distance
    PROXIMITY_MAX_DISTANCE = 5.00 # proximity of 0 corresponds to 5m distance
    PLOT_HORIZON_METERS = 3.0 # maximum distance to plot in proximity plots (in meters)

class AudioConfig:
    """Configuration for audio classification."""
    VALID_RTTM_CLASSES = ['OHS', 'KCDS', 'KCHI']
    VALID_EVENT_IDS = {"child_talking", "other_person_talking", "overheard_speech", "singing/humming"}
    SR = 16000
    N_MELS = 256
    HOP_LENGTH = 256
    #WINDOW_DURATION = 3.0
    #WINDOW_STEP = 1.0
    MAX_SEGMENT_DURATION = 20 #in seconds
    FIXED_TIME_STEPS = int(np.ceil(MAX_SEGMENT_DURATION * SR / HOP_LENGTH))

    NUM_EPOCHS = 200
    PATIENCE = 30
    MODEL_ID = 3

class KchiVoc_Config:
    """Configuration for KCHI vocalizations."""
    MODEL_ID = 4
    
class AnalysisConfig:
    """Consolidated configuration for active social analysis rules."""
    
    RANDOM_SAMPLING = True
    # -- General --
    NUM_FOLDS = 5
    EXCLUSION_SECONDS = 30           
    SAMPLE_RATE = 10                 

    
    TERTIARY_PARAMETERS = {
        "SOCIAL_STATE_MAPPING": {1: 'Interacting', 2: 'Available', 3: 'Alone'},
        "GAP_DEFAULT_LABEL": "Alone",
        "MAX_TURN_TAKING_GAP_SEC": 5.0,
        "MAX_SAME_SPEAKER_GAP_SEC": 1.0,
        "AUDIO_VISUAL_GATING_FLOOR": 0.35,
        "PRESENCE_SCORE_FRAMES_PER_SEC": 3,
        "INSTANT_CONFIDENCE_THRESHOLD": 0.15,
        "PROXIMITY_THRESHOLD": 0.60,
        "HIGH_CONFIDENCE_PROXIMITY_THRESHOLD": 0.65,
        "HIGH_CONFIDENCE_FACE_CONFIDENCE": 0.85,
        "SUSTAINED_KCDS_WINDOW_SEC": 10.0,
        "SUSTAINED_KCDS_THRESHOLD": 0.95,
        "SUSTAINED_OHS_WINDOW_SEC": 10.0,
        "SUSTAINED_OHS_THRESHOLD": 0.65,
        "VISUAL_PERSISTENCE_SEC": 3.0,
        "SHORT_TERM_VISUAL_MEMORY_SEC": 3.5,
        "CPD_PENALTY": 5.0,
        "CPD_INTERACTING_THRESHOLD": 0.40,
        "CPD_TOTAL_PRESENCE_FLOOR": 0.45,
        "SAME_SEGMENT_MERGE_THRESHOLD": 3.0,
        "GAP_STRETCH_THRESHOLD": 2.0
    }
    # TERTIARY_PARAMETERS = {
    #     "SOCIAL_STATE_MAPPING": {1: 'Interacting', 2: 'Available', 3: 'Alone'},
    #     "GAP_DEFAULT_LABEL": "Alone",
    #     "MAX_TURN_TAKING_GAP_SEC": 5.0,
    #     "MAX_SAME_SPEAKER_GAP_SEC": 1.0,
    #     "AUDIO_VISUAL_GATING_FLOOR": 0.35,
    #     "INSTANT_CONFIDENCE_THRESHOLD": 0.15,
    #     "PROXIMITY_THRESHOLD": 0.6,
    #     "HIGH_CONFIDENCE_PROXIMITY_THRESHOLD": 0.65,
    #     "HIGH_CONFIDENCE_FACE_CONFIDENCE": 0.85,
    #     "SUSTAINED_KCDS_WINDOW_SEC": 8.0,
    #     "SUSTAINED_KCDS_THRESHOLD": 0.95,
    #     "SUSTAINED_OHS_WINDOW_SEC": 10.0,
    #     "SUSTAINED_OHS_THRESHOLD": 0.6,
    #     "VISUAL_PERSISTENCE_SEC": 2.5,
    #     "SHORT_TERM_VISUAL_MEMORY_SEC": 3.5,
    #     "CPD_PENALTY": 5.0,
    #     "CPD_INTERACTING_THRESHOLD": 0.4,
    #     "CPD_TOTAL_PRESENCE_FLOOR": 0.45,
    #     "SAME_SEGMENT_MERGE_THRESHOLD": 3.0,
    #     "GAP_STRETCH_THRESHOLD": 2.5
    #     }
    
    # Define the "Binary Specials"
    BINARY_PARAMETERS = {
        "SOCIAL_STATE_MAPPING": {1: 'Interacting', 2: 'Not_Interacting'},
        "GAP_DEFAULT_LABEL": "Not_Interacting",
        "AUDIO_VISUAL_GATING_FLOOR": 0.45,
        "INSTANT_CONFIDENCE_THRESHOLD": 0.6,
        "MAX_TURN_TAKING_GAP_SEC": 9.0,
        "MAX_SAME_SPEAKER_GAP_SEC": 2.5,
        "PROXIMITY_THRESHOLD": 0.65,
        "SUSTAINED_KCDS_WINDOW_SEC": 18.0,
        "SUSTAINED_KCDS_THRESHOLD": 0.98,
        "HIGH_CONFIDENCE_PROXIMITY_THRESHOLD": 0.9,
        "HIGH_CONFIDENCE_FACE_CONFIDENCE": 0.8,
        "VISUAL_PERSISTENCE_SEC": 1.0,
        "SHORT_TERM_VISUAL_MEMORY_SEC": 2.5,
        "CPD_PENALTY": 3.0,
        "CPD_INTERACTING_THRESHOLD": 0.95,
        "CPD_TOTAL_PRESENCE_FLOOR": 0.3,
        "SAME_SEGMENT_MERGE_THRESHOLD": 1.5,
        "GAP_STRETCH_THRESHOLD": 1.5
    }

    @classmethod
    def apply_mode(cls, mode="tertiary"):
        """Hot-swaps all parameters based on mode."""
        params = cls.BINARY_PARAMETERS if mode == "binary" else cls.TERTIARY_PARAMETERS
        for key, value in params.items():
            setattr(cls, key, value)

class HyperparameterConfig:
    """
    Complete Search Grid for Social Interaction Analysis.
    Includes all 15 active parameters from the frame-level and video-level engines.
    """
    HYPERPARAMETER_RANGES = {
    # --- 1. Target Turn-Taking & Speaker Gaps ---
    "MAX_TURN_TAKING_GAP_SEC": [5.0],            # Literature baseline; unanimous consensus
    "MAX_SAME_SPEAKER_GAP_SEC": [1.0],      # Both values represented equally in top setups

    # --- 2. Audio-Visual Gating & Visual Confidence ---
    "AUDIO_VISUAL_GATING_FLOOR": [0.35, 0.1, 0.2, 0.3], # Retain: top setups spanned 0.30 to 0.45
    "PRESENCE_SCORE_FRAMES_PER_SEC": [3, 6, 9],
    "INSTANT_CONFIDENCE_THRESHOLD": [0.15],          # Pin: 0.15 swept 100% of top setups

    # --- 3. Physical Proximity & Face Anchors ---
    "PROXIMITY_THRESHOLD": [0.60],       # Evenly distributed across 0.60, 0.65, 0.70
    "HIGH_CONFIDENCE_PROXIMITY_THRESHOLD": [0.65],
    "HIGH_CONFIDENCE_FACE_CONFIDENCE": [0.85], # Split 50/50 between 0.85 and 0.90

    # --- 4. Speech Stream Gating (KCDS & OHS) ---
    "SUSTAINED_KCDS_WINDOW_SEC": [10.0],   # Spanned all three values
    "SUSTAINED_KCDS_THRESHOLD": [0.95],        # Drop 0.98; focus on the two active modes
    "SUSTAINED_OHS_WINDOW_SEC": [10.0],   # Spanned 8.0 to 12.0
    "SUSTAINED_OHS_THRESHOLD": [0.65],

    # --- 5. Visual Persistence & Short-Term Memory ---
    "VISUAL_PERSISTENCE_SEC": [3.0],       # Perfectly distributed (2.0, 2.5, 3.0)
    "SHORT_TERM_VISUAL_MEMORY_SEC": [3.5],      # 3.5 was unanimous floor/ceiling; test 3.5 & test upward to 4.0

    # --- 6. PELT Change-Point Detection & State Assignment ---
    "CPD_PENALTY": [5.0],                       # Drop 4.5 and 6.0 (100% of top setups picked 5.0 or 5.5)
    "CPD_INTERACTING_THRESHOLD": [0.40],             # Pin: 0.40 dominated 3 out of 4 top setups
    "CPD_TOTAL_PRESENCE_FLOOR": [0.45],        # Drop 0.55 (no top setups used it)

    # --- 7. Temporal Smoothing & Post-Processing ---
    "SAME_SEGMENT_MERGE_THRESHOLD": [3.0], # Top setups were 3.0, 3.5, and 4.0
    "GAP_STRETCH_THRESHOLD": [2.0]         # Shift downward: 2.0 was ceiling/floor in 3/4 setups
    }
    
    # HYPERPARAMETER_RANGES = {
    #     # --- 1. Target Turn-Taking & Speaker Gaps ---
    #     "MAX_TURN_TAKING_GAP_SEC": [5.0],            # Literature baseline; unanimous consensus
    #     "MAX_SAME_SPEAKER_GAP_SEC": [0.8, 1.0],      # Both values represented equally in top setups
    
    #     # --- 2. Audio-Visual Gating & Visual Confidence ---
    #     "AUDIO_VISUAL_GATING_FLOOR": [0.30, 0.35, 0.45], # Retain: top setups spanned 0.30 to 0.45
    #     "INSTANT_CONFIDENCE_THRESHOLD": [0.15],          # Pin: 0.15 swept 100% of top setups
    
    #     # --- 3. Physical Proximity & Face Anchors ---
    #     "PROXIMITY_THRESHOLD": [0.60, 0.65, 0.70],       # Evenly distributed across 0.60, 0.65, 0.70
    #     "HIGH_CONFIDENCE_PROXIMITY_THRESHOLD": [0.65, 0.70, 0.75],
    #     "HIGH_CONFIDENCE_FACE_CONFIDENCE": [0.85, 0.90], # Split 50/50 between 0.85 and 0.90
    
    #     # --- 4. Speech Stream Gating (KCDS & OHS) ---
    #     "SUSTAINED_KCDS_WINDOW_SEC": [6.0, 8.0, 10.0],   # Spanned all three values
    #     "SUSTAINED_KCDS_THRESHOLD": [0.95, 1.00],        # Drop 0.98; focus on the two active modes
    #     "SUSTAINED_OHS_WINDOW_SEC": [8.0, 10.0, 12.0],   # Spanned 8.0 to 12.0
    #     "SUSTAINED_OHS_THRESHOLD": [0.60, 0.65],
    
    #     # --- 5. Visual Persistence & Short-Term Memory ---
    #     "VISUAL_PERSISTENCE_SEC": [2.0, 2.5, 3.0],       # Perfectly distributed (2.0, 2.5, 3.0)
    #     "SHORT_TERM_VISUAL_MEMORY_SEC": [3.5, 4.0],      # 3.5 was unanimous floor/ceiling; test 3.5 & test upward to 4.0
    
    #     # --- 6. PELT Change-Point Detection & State Assignment ---
    #     "CPD_PENALTY": [5.0, 5.5],                       # Drop 4.5 and 6.0 (100% of top setups picked 5.0 or 5.5)
    #     "CPD_INTERACTING_THRESHOLD": [0.40],             # Pin: 0.40 dominated 3 out of 4 top setups
    #     "CPD_TOTAL_PRESENCE_FLOOR": [0.45, 0.50],        # Drop 0.55 (no top setups used it)
    
    #     # --- 7. Temporal Smoothing & Post-Processing ---
    #     "SAME_SEGMENT_MERGE_THRESHOLD": [3.0, 3.5, 4.0], # Top setups were 3.0, 3.5, and 4.0
    #     "GAP_STRETCH_THRESHOLD": [1.5, 2.0, 2.5]         # Shift downward: 2.0 was ceiling/floor in 3/4 setups
    # }