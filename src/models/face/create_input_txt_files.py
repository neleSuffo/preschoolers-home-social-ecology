import json
import logging
import sqlite3
import cv2
import re
import json
import shutil
import random
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from src.constants import DataPaths, BasePaths, FaceDetection, FaceClassification
from src.config import FaceConfig, DataConfig

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =======================================================
# Helper Functions
# =======================================================
def get_child_id_from_filename(filename: str) -> str:
    """
    Get child ID from filename using regex. Assumes filename contains a pattern like 'id123456'.

    Parameters
    ----------
    filename : str
        The name of the file from which to extract the child ID.

    Returns
    -------
    str
        The extracted child ID or None if not found.
    """
    match = re.search(r'id(\d+)_', filename)
    return 'id' + match.group(1) if match else None

def create_neg_df(sampled_neg_list: List[Tuple[str, str]], child_id_getter: callable, class_cols: List[str]) -> pd.DataFrame:
    """
    Creates a DataFrame for negative samples with the specified class columns set to 0.

    Parameters
    ----------
    sampled_neg_list : List[Tuple[str, str]]
        List of tuples containing image paths and IDs.
    child_id_getter : callable
        Function to extract child ID from filename.
    class_cols : List[str]
        List of class column names.

    Returns
    -------
    pd.DataFrame
        DataFrame containing negative samples.
    """
    entries = []
    for image_path, image_id in sampled_neg_list:
        filename = Path(image_path).stem
        entries.append({
            "filename": filename,
            "id": image_id,
            "has_annotation": False,
            "child_id": child_id_getter(filename),
            **{col: 0 for col in class_cols}
        })
    return pd.DataFrame(entries)

def get_data_constants(data_type: str) -> type:
    """
    Returns the correct constants object based on the data_type flag.

    Parameters
    ----------
    data_type : str
        Type of data processing ("detection" or "classification")

    Returns
    -------
    Type
        Corresponding constants object for the specified data type.
    """
    if data_type == "detection":
        return FaceDetection
    elif data_type == "classification":
        return FaceClassification
    else:
        raise ValueError(f"Unknown data type: {data_type}. Must be 'detection' or 'classification'.")
    
# ==============================
# Database and Query
# ==============================
def fetch_all_annotations(category_ids: List[int]) -> List[Tuple]:
    """
    Fetch annotations for given category IDs from the SQLite database.
    
    Parameters
    ----------
    category_ids : List[int]
        List of category IDs to filter annotations.
        
    Returns
    -------
    List[Tuple]       
        List of tuples containing annotation data.
    """
    logging.info(f"Fetching annotations for category IDs: {category_ids}")
    placeholders = ", ".join("?" * len(category_ids))

    query = f"""
    SELECT DISTINCT 
        a.category_id, a.bbox, v.file_name as video_file, a.image_id as raw_frame, a.person_age, v.id as video_id
    FROM annotations a
    JOIN videos v ON a.video_id = v.id
    WHERE a.category_id IN ({placeholders})
      AND a.outside = 0
    ORDER BY a.video_id, a.image_id
    """

    # set up correct mapping for shifted videos
    corrected_results = []
    exception_map = DataConfig.SHIFTED_VIDEOS_OFFSETS
    
    with sqlite3.connect(DataPaths.ANNO_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, category_ids)
        rows = cursor.fetchall()
        
    for cat_id, bbox, video_file, raw_frame, age, v_id in rows:
        video_name = video_file.replace('.mp4', '')
        
        # 2. Apply logical correction
        adjusted_frame = raw_frame
        if video_name in exception_map:
            start_frame, shift = exception_map[video_name]
            # If annotation is at or after the shift point, reverse the shift
            if raw_frame >= start_frame:
                # subtract shift because DB has "wrong" frame 
                # and we need "real" frame on disk.
                # Example: raw_frame 28, shift -2 -> 28 - (-2) = 30.
                adjusted_frame = raw_frame - shift 
        
        # 3. Reconstruct the image filename to match the disk
        corrected_img_name = f"{video_name}_{adjusted_frame:06d}"
        
        corrected_results.append((cat_id, bbox, corrected_img_name, age))

    logging.info(f"Found and corrected {len(corrected_results)} annotations.")   
    return corrected_results

# ==============================
# Image and Bounding Box Utilities
# ==============================
def convert_to_yolo_format(width: int, height: int, bbox: List[float]) -> Tuple[float, float, float, float]:
    """
    Convert [xtl, ytl, xbr, ybr] to YOLO (x_center, y_center, width, height).
    
    Parameters
    ----------
    width : int
        Image width in pixels
    height : int 
        Image height in pixels
    bbox : List[float]
        Bounding box in format [xtl, ytl, xbr, ybr] (absolute coordinates)
        
    Returns
    -------
    Tuple[float, float, float, float]
        YOLO format: (x_center_norm, y_center_norm, width_norm, height_norm)
        All values normalized to [0, 1]
    """
    xtl, ytl, xbr, ybr = bbox
    
    # Calculate center coordinates (absolute)
    x_center = (xtl + xbr) / 2.0
    y_center = (ytl + ybr) / 2.0
    
    # Calculate width and height (absolute)
    bbox_width = xbr - xtl
    bbox_height = ybr - ytl
    
    # Normalize to [0, 1] by dividing by image dimensions
    x_center_norm = x_center / width
    y_center_norm = y_center / height
    width_norm = bbox_width / width
    height_norm = bbox_height / height
    
    return (x_center_norm, y_center_norm, width_norm, height_norm)

# ==============================
# Annotation Writing
# ==============================
def write_annotations(file_path: Path, lines: List[str]) -> None:
    file_path.write_text("".join(lines))

def save_annotations(annotations: List[Tuple], output_dir: Path = None, mode: str = "face-only", data_type: str = "detection") -> None:
    """
    Save annotations to text files in the specified directory.
    
    Parameters
    ----------
    annotations : List[Tuple]
        List of tuples containing (category_id, bbox_json, img_name, age)
    output_dir : Path, optional
        Directory to save annotation files
    mode : str
        Detection mode to use for saving annotations (default: "face-only")
    data_type : str
        Type of data processing ("detection" or "classification")
    """
    CONSTANTS = get_data_constants(data_type)
    
    if output_dir is None:
        output_dir = CONSTANTS.LABELS_INPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = defaultdict(list)
    processed, skipped = 0, 0

    if mode == "face-only":
        mode_config = FaceConfig.AGE_GROUP_TO_CLASS_ID_FACE_ONLY
    elif mode == "age-binary":
        mode_config = FaceConfig.AGE_GROUP_TO_CLASS_ID_AGE_BINARY
    else:
        logging.error(f"Unknown mode: {mode}")
        return
    
    for _, bbox_json, img_name, age in annotations:
        
        class_id = mode_config.get(age, 99)
        
        if class_id == 99:
            logging.warning(f"Unknown age group '{age}' for {img_name}")
            skipped += 1
            continue
        
        # --- CLASSIFICATION (Single class ID line) ---
        if data_type == "classification" and mode == "age-binary":
            # For classification, we only care if a face is present, and its class (child/adult).
            # The classification label is only the class ID, and we assume the image is positive.
            files[img_name].append(f"{class_id}\n")
            processed += 1
            continue
        
        # --- DETECTION (YOLO) MODE LOGIC ---
        # img_name comes from database and should be the image filename without extension
        # Example: quantex_at_home_id255237_2022_05_08_04_000240 -> quantex_at_home_id255237_2022_05_08_04
        img_name_parts = img_name.split("_")
        if len(img_name_parts) < 9:
            logging.warning(f"Invalid image name format: {img_name} (expected at least 9 parts)")
            skipped += 1
            continue

        video_folder_name = "_".join(img_name.split("_")[:-1])
        
        img_path = None
        for ext in DataConfig.VALID_EXTENSIONS:
            potential_path = FaceDetection.IMAGES_INPUT_DIR / video_folder_name / f"{img_name}{ext}"
            if potential_path.exists():
                img_path = potential_path
                break
        
        if img_path is None:
            logging.warning(f"Image not found: {img_name} in folder {video_folder_name}")
            logging.debug(f"Searched paths: {[FaceDetection.IMAGES_INPUT_DIR / video_folder_name / f'{img_name}{ext}' for ext in DataConfig.VALID_EXTENSIONS]}")
            skipped += 1
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logging.warning(f"Failed to load image: {img_path}")
                skipped += 1
                continue
            
            height, width = img.shape[:2]
            
            bbox = json.loads(bbox_json)
            yolo_bbox = convert_to_yolo_format(width, height, bbox)
            
            files[img_name].append(f"{class_id} " + " ".join(map(str, yolo_bbox)) + "\n")
            processed += 1
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
            skipped += 1

    # Save annotation files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        for img_name, lines in files.items():
            output_file = output_dir / f"{img_name}.txt"
            executor.submit(write_annotations, output_file, lines)

    logging.info(f"Processed {processed}, skipped {skipped}")

def fetch_noisy_frames() -> List[str]:
    """
    Fetch file_names of frames that contain *only* the category_id = -1
    (i.e., frames that were flagged as 'bad' or 'no face' but have no other faces).
    
    Returns
    -------
    List[str]
        List of file_names (stems) to be excluded from the negative sample pool.
    """
    query = """
    SELECT
        i.file_name
    FROM images i
    JOIN annotations a ON i.frame_id = a.image_id AND i.video_id = a.video_id
    WHERE a.category_id = -1
    GROUP BY i.file_name
    HAVING COUNT(CASE WHEN a.category_id != -1 THEN 1 END) = 0
    """

    with sqlite3.connect(DataPaths.ANNO_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        # Fetch results and flatten the list of tuples (file_name,) into a list of strings
        results = [row[0] for row in cursor.fetchall()]

    return results

def get_total_number_of_annotated_frames(label_path: Path, image_folder: Path = FaceDetection.IMAGES_INPUT_DIR) -> Tuple[List[Tuple[str, str]], Dict[str, List[Tuple[str, str]]]]:
    """
    Returns ALL annotated (positive) frames and ALL potential clean negative frame candidates (sampled by frame offset).
    
    Parameters
    ----------
    label_path : Path
        Path to the directory containing label files.
    image_folder : Path
        Path to the image folder containing video subfolders.
    
    Returns
    -------
    Tuple[List[Tuple[str, str]], Dict[str, List[Tuple[str, str]]]]
        A tuple containing:
        - List of tuples (image_path, image_id) for all positive annotated frames.
        - Dictionary mapping video names to lists of tuples (image_path, image_id) for potential negative frame candidates.
    """
    DEFAULT_OFFSET = 0

    # 1. Fetch frames to be explicitly excluded from negative samples (only -1 annotation)
    frames_to_exclude = fetch_noisy_frames()
    excluded_frames_set = set(frames_to_exclude)
    
    video_names = set()
    positive_images = []
    positive_frame_stems = set()
    
    # 2. Find all videos that have any valid annotations
    for annotation_file in label_path.glob("*.txt"):
        if annotation_file.stat().st_size > 0:
            try:
                # Extract video name from annotation file stem
                # match example: quantex_at_home_id255237_2022_05_08_04_000240.txt -> positive_frame_stems: quantex_at_home_id255237_2022_05_08_04_000240
                # video_name: quantex_at_home_id255237_2022_05_08_04_000240
                stem = annotation_file.stem
                match = re.match(r"(.+)_\d{6}$", stem)
                if match:
                    video_name = match.group(1)
                    video_names.add(video_name)
                    positive_frame_stems.add(stem)
            except Exception as e:
                logging.warning(f"Error reading annotation file {annotation_file}: {e}")

    logging.info(f"Found {len(video_names)} unique video names with annotations.")

    # 3. Get annotated frames and collect ALL valid potential negative frames (sampled frames only)
    video_negative_candidates = defaultdict(list) # Use defaultdict for cleaner grouping
    
    for video_name in video_names:
        video_path = image_folder / video_name
        if video_path.exists() and video_path.is_dir():
            
            exception_map = DataConfig.SHIFTED_VIDEOS_OFFSETS
            # Get the frame interval for this video (default is DataConfig.FPS)
            frame_interval = DataConfig.NON_STANDARD_FRAME_STEPS.get(video_name, DataConfig.FPS)
            
            # Iterate through all frames in the video folder
            for frame in video_path.iterdir():
                if frame.is_file():
                    stem = frame.stem
                    parts = stem.split("_")
                    
                    # --- Frame Number Check ---
                    frame_number = -1
                    if len(parts) >= 9:
                        try:
                            frame_number = int(parts[-1])
                        except ValueError:
                            continue # Skip if frame number is not an integer
                    
                    # Determine if the frame is a sampled frame
                    is_sampled_frame = False
                    
                    if video_name in exception_map:
                        # Logic for exception videos with frame shift
                        start_frame, shift = exception_map[video_name]
                        
                        if frame_number < start_frame:
                            # Rule 1: Before the exception start, use standard sampling
                            if frame_number % frame_interval == DEFAULT_OFFSET:
                                is_sampled_frame = True
                        else:
                            # Rule 2: From the exception start onward, use the shifted modulo rule
                            # Sampling is shifted to start at start_frame and then every frame_interval frames
                            if (frame_number - start_frame) % frame_interval == DEFAULT_OFFSET:
                                is_sampled_frame = True
                            
                    else:
                        # Default rule for all other videos: multiples of frame_interval
                        # This assumes DEFAULT_OFFSET = 0 and the step size is constant.
                        if frame_number % frame_interval == DEFAULT_OFFSET:
                            is_sampled_frame = True
                    
                    # A. If this frame has a valid annotation, include it as a positive
                    if stem in positive_frame_stems:
                        positive_images.append((str(frame.resolve()), stem))
                        continue

                    # B. If it's a sampled frame AND a clean negative, add to candidates
                    if is_sampled_frame and stem not in excluded_frames_set:
                        video_negative_candidates[video_name].append((str(frame.resolve()), stem))
    
    logging.info(f"Total positive (annotated) frames found: {len(positive_images)}")
    logging.info(f"Total potential negative candidates found: {sum(len(c) for c in video_negative_candidates.values())}")
    
    # Return positive images list and the dict of negative candidates
    return positive_images, dict(video_negative_candidates)

def get_class_distribution(total_images: list, annotation_folder: Path, mode: str) -> pd.DataFrame:
    """
    Reads label files and groups images based on their class distribution for the given detection mode.

    Parameters:
    ----------
    total_images: list
        List of tuples containing image paths and IDs.
    annotation_folder: Path
        Path to the directory containing label files.
    mode: str
        The mode for class mapping ('face-only' or 'age-binary').

    Returns:
    -------
    pd.DataFrame
        DataFrame containing image filenames, IDs, and their corresponding one-hot encoded class labels.
    """   
    if not total_images:
        logging.error("No images provided to get_class_distribution")
        return pd.DataFrame()
        
    image_class_mapping = []

    if mode == "face-only":
        id_to_label_map = FaceConfig.MODEL_CLASS_ID_TO_LABEL_FACE_ONLY
        target_class_names = FaceConfig.TARGET_LABELS_FACE_ONLY
    elif mode == "age-binary":
        id_to_label_map = FaceConfig.MODEL_CLASS_ID_TO_LABEL_AGE_BINARY
        target_class_names = FaceConfig.TARGET_LABELS_AGE_BINARY
    else:
        logging.error(f"Unknown detection mode: {mode}")
        return pd.DataFrame()

    # Step 1: Read each image and its corresponding annotation file
    for i, (image_path, image_id) in enumerate(total_images):
        image_file = Path(image_path)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name

        mode_labels = set()
        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            try:
                with open(annotation_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        lines = content.split('\n')
                        class_ids = {int(line.split()[0]) for line in lines if line.strip()}
                        
                        original_labels = [id_to_label_map[cid] for cid in class_ids if cid in id_to_label_map]
                        mode_labels.update(original_labels)
                            
            except Exception as e:
                logging.warning(f"Error reading annotation file {annotation_file}: {e}")

        # Create one-hot encoded dictionary for the image
        try:
            mapping_entry = {
                "filename": image_file.stem,
                "id": image_id,
                "has_annotation": bool(mode_labels),
            }

            for class_name in target_class_names:
                mapping_entry[class_name] = 1 if class_name in mode_labels else 0
                
            image_class_mapping.append(mapping_entry)
            
        except Exception as e:
            logging.error(f"Error creating mapping entry for {image_file.name}: {e}")
            continue
        
    df = pd.DataFrame(image_class_mapping)
    
    return df

def get_sampled_frames_for_children(child_ids: List[str], image_folder: Path) -> List[str]:
    """
    Get all sampled frames (positive and negative) for specified child IDs from all their videos,
    respecting the SHIFTED_VIDEOS_OFFSETS configuration.
    
    Parameters
    ----------
    child_ids : List[str]
        List of child IDs to get sampled frames for
    image_folder : Path
        Path to the image folder
        
    Returns
    -------
    List[str]
        List of image file stems (filenames without extension) that are sampled.
    """
    DEFAULT_OFFSET = 0
    sampled_frames = []
    
    for child_id in child_ids:
        for video_folder in image_folder.iterdir():
            if video_folder.is_dir() and child_id in video_folder.name:
                video_name = video_folder.name
                exception_map = DataConfig.SHIFTED_VIDEOS_OFFSETS
                
                for frame in video_folder.iterdir():
                    if frame.is_file() and frame.suffix.lower() in DataConfig.VALID_EXTENSIONS:
                        stem = frame.stem
                        parts = stem.split("_")
                        
                        frame_number = -1
                        if len(parts) >= 9:
                            try:
                                frame_number = int(parts[-1])
                            except ValueError:
                                continue 

                        is_sampled_frame = False
                        
                        if video_name in exception_map:
                            start_frame, shift = exception_map[video_name]
                            
                            if frame_number < start_frame:
                                if frame_number % DataConfig.FPS == DEFAULT_OFFSET:
                                    is_sampled_frame = True
                            else:
                                # Rule 2: From the exception start onward, use the shifted modulo rule
                                if (frame_number - start_frame) % DataConfig.FPS == DEFAULT_OFFSET:
                                    is_sampled_frame = True
                                
                        else:
                            # Default rule for all other videos: multiples of DataConfig.FPS
                            if frame_number % DataConfig.FPS == DEFAULT_OFFSET:
                                is_sampled_frame = True
                        
                        if is_sampled_frame:
                            sampled_frames.append(stem)
    
    return sampled_frames
    
def split_by_child_id(df: pd.DataFrame, negative_candidates: Dict[str, List[Tuple[str, str]]], train_ratio: float = FaceConfig.TRAIN_SPLIT_RATIO, labels_input_dir: Path = None, mode: str = "face-only", data_type: str = "detection") -> Tuple[List[str], List[str], List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training, validation, and test sets using child IDs as the unit, while balancing class distributions.

    For test set: Keeps ALL frames from test children (not just annotated ones) to reflect real-world scenarios.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with image data and annotations
    negative_candidates : Dict[str, List[Tuple[str, str]]]
        Dictionary of potential negative frame candidates per video
    train_ratio : float
        Ratio for training split
    labels_input_dir : Path
        Path to labels directory (if None, uses default FaceDetection.LABELS_INPUT_DIR)
    mode : str
        Detection mode to use for splitting ('face-only' or 'age-binary')
    data_type : str
        Type of data processing ("detection" or "classification")
    """
    # Define minimum number of face images required in the test set as fixed percentage of positive images
    if labels_input_dir is None:
        labels_input_dir = FaceDetection.LABELS_INPUT_DIR
        
    if 'filename' not in df.columns:
        logging.error(f"'filename' column not found in DataFrame. Available columns: {df.columns.tolist()}")
        return [], [], [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Prepare IDS and Columns
    df['child_id'] = df['filename'].apply(get_child_id_from_filename)
    df.dropna(subset=['child_id'], inplace=True)
    class_columns = [col for col in df.columns if col not in ['filename', 'id', 'has_annotation', 'child_id']]
    sorted_child_ids = df['child_id'].unique().tolist()

    # --- 1. ID SELECTION ---
    train_ids, val_ids, test_ids = [], [], []    
    image_dir = FaceDetection.IMAGES_INPUT_DIR if data_type == "detection" else FaceClassification.IMAGES_INPUT_DIR
    
    # Group children by video count
    child_video_map = defaultdict(list)
    for folder in image_dir.iterdir():
        if folder.is_dir():
            cid = get_child_id_from_filename(folder.name)
            if cid in sorted_child_ids:
                child_video_map[cid].append(folder.name)

    # Buckets: Single vs Multi video children
    single_video_pool = [cid for cid, vids in child_video_map.items() if len(vids) == 1]
    multi_video_pool = [cid for cid, vids in child_video_map.items() if len(vids) > 1]

    # Shuffle to ensure randomness within the categories
    random.seed(DataConfig.RANDOM_SEED)
    random.shuffle(single_video_pool)
    random.shuffle(multi_video_pool)
    
    # Target: 3 children for Val, 3 for Test (using single-video children first)
    for _ in range(3):
        if single_video_pool:
            val_ids.append(single_video_pool.pop())
        else:
            val_ids.append(multi_video_pool.pop())
            
        if single_video_pool:
            test_ids.append(single_video_pool.pop())
        else:
            test_ids.append(multi_video_pool.pop())

    # Everyone else goes to Train
    train_ids = single_video_pool + multi_video_pool
    logging.info(f"Split Summary: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # --- 2. COMPILE POOLS ---
    # Positives (from the annotated dataframe)
    train_df_pos = df[df["child_id"].isin(train_ids)].copy()
    val_df_pos = df[df["child_id"].isin(val_ids)].copy()
    test_df_pos = df[df["child_id"].isin(test_ids)].copy()

    # Negatives (from candidate lists)
    train_neg_candidates, val_neg_candidates, test_neg_candidates = [], [], []
    for video_name, candidates in negative_candidates.items():
        cid = get_child_id_from_filename(video_name)
        if cid in train_ids:
            train_neg_candidates.extend(candidates)
        elif cid in val_ids:
            val_neg_candidates.extend(candidates)
        elif cid in test_ids:
            test_neg_candidates.extend(candidates)
            
    # --- 3. APPLY SAMPLING (Balance Train, Keep Val/Test Natural) ---
    
    # 3a. TRAIN: Balanced
    target_train_neg = int(len(train_df_pos) * FaceConfig.NEGATIVE_SAMPLING_RATIO)
    if len(train_neg_candidates) >= target_train_neg:
        sampled_train_neg = random.sample(train_neg_candidates, target_train_neg)
    else:
        sampled_train_neg = train_neg_candidates
        logging.warning(f"Train set under-sampled: {len(sampled_train_neg)}/{target_train_neg}")
    
    train_neg_df = create_neg_df(sampled_train_neg, get_child_id_from_filename, class_columns)
    train_df = pd.concat([train_df_pos, train_neg_df], ignore_index=True)
    
    # 3b. VALIDATION: Naturally Imbalanced (Take ALL available)
    val_neg_df = create_neg_df(val_neg_candidates, get_child_id_from_filename, class_columns)
    val_df = pd.concat([val_df_pos, val_neg_df], ignore_index=True)
    
    # 3c. TEST: Naturally Imbalanced (Take ALL available)
    test_neg_df = create_neg_df(test_neg_candidates, get_child_id_from_filename, class_columns)
    test_df = pd.concat([test_df_pos, test_neg_df], ignore_index=True)

    # --- 4. RETURN ---
    return (
        train_df['filename'].tolist(), val_df['filename'].tolist(), test_df['filename'].tolist(),
        train_df, val_df, test_df)

def move_images(image_names: list,
                split_type: str,
                label_path: Path,
                n_workers: int = 4,
                data_type: str = "detection",
                mode: str = "face-only",
                df_split: pd.DataFrame = None) -> Tuple[int, int]:
    """
    Move images and their corresponding labels to the specified split directory for face detection.
    Uses multithreading for faster processing.
    
    Parameters
    ----------
    image_names: list
        List of image names to process
    split_type: str
        Split type (train, val, or test)
    label_path: Path
        Path to label directory
    n_workers: int
        Number of worker threads for parallel processing
    data_type: str
        Type of data ('detection' or 'classification')
    mode: str
        Mode for classification ('age-binary' supported)
    df_split: pd.DataFrame
        DataFrame for the current split (required for classification)
        
    Returns
    -------
    Tuple[int, int]
        Number of successful and failed moves
    """
    CONSTANTS = get_data_constants(data_type)
    
    if not image_names:
        logging.info(f"No images to move for face detection {split_type}")
        return (0, 0)
    
    image_src_root = CONSTANTS.IMAGES_INPUT_DIR

    # --- Classification Mode Setup ---
    if data_type == "classification":
        if mode != "age-binary":
            logging.error("Classification is only supported in 'age-binary' mode for this script logic.")
            return (0, len(image_names))
            
        # Check if we have the DataFrame needed to map filenames to classes
        if df_split is None or df_split.empty:
             logging.error("DataFrame is required for classification to determine image class!")
             return (0, len(image_names))

        # Map filenames to class folders based on df_split
        class_map = {}
        target_labels = FaceConfig.TARGET_LABELS_AGE_BINARY
        # Map class name to folder name (e.g., 'child' -> 'child' folder)
        for class_name in target_labels:
            class_map[class_name] = class_name
                    
        file_to_class_folder = {}
        for _, row in df_split.iterrows():
            # Positive case (should only be one class flag true for classification)
            if row[target_labels[0]] == 1:
                file_to_class_folder[row['filename']] = target_labels[0]
            elif row[target_labels[1]] == 1:
                file_to_class_folder[row['filename']] = class_map[target_labels[1]]
            else:
                logging.warning(f"Positive image {row['filename']} has no class label set. Skipping.")
                continue
                
    # --- Setup Destination Directories ---
    image_dst_base_dir = CONSTANTS.INPUT_DIR / "images" / split_type
    
    if data_type == "detection":
        label_dst_dir = CONSTANTS.INPUT_DIR / "labels" / split_type
        label_dst_dir.mkdir(parents=True, exist_ok=True)
    else:
        # For classification, the final destination is within a class folder inside 'images/split_type'
        image_dst_base_dir.mkdir(parents=True, exist_ok=True) # Ensure split dir exists

    def process_single_image(image_name: str) -> bool:
        """Process a single image and move it to the final location."""
        try:
            image_parts = image_name.split("_")
            if len(image_parts) < 9:
                return False

            image_folder = image_name.rsplit('_', 1)[0]
            image_src = None

            for ext in DataConfig.VALID_EXTENSIONS:
                potential_path = image_src_root / image_folder / f"{image_name}{ext}"
                if potential_path.exists():
                    image_src = potential_path
                    break

            if image_src is None:
                return False

            # --- 1. Determine Image Destination ---
            if data_type == "classification":
                class_folder_name = file_to_class_folder.get(image_name)
                if not class_folder_name:
                    # Already logged a warning in file_to_class_folder creation, just return False
                    return False

                # Destination is [base_dir]/[split]/[class_folder]/[image_file]
                final_image_dst_dir = image_dst_base_dir / class_folder_name
                final_image_dst_dir.mkdir(parents=True, exist_ok=True)
                image_dst = final_image_dst_dir / f"{image_name}{image_src.suffix}"

            else: # Detection Mode
                # Destination is [base_dir]/[split]/[image_file]
                image_dst = image_dst_base_dir / f"{image_name}{image_src.suffix}"
                image_dst_base_dir.mkdir(parents=True, exist_ok=True) # Ensure split dir exists

                # --- 2. Handle Label File (Detection Only) ---
                label_src = label_path / f"{image_name}.txt"
                label_dst = label_dst_dir / f"{image_name}.txt"

                if not label_src.exists():
                    label_dst.touch()
                else:
                    shutil.copy2(label_src, label_dst)

            # --- 3. Move Image ---
            shutil.copy2(image_src, image_dst)
            return True

        except Exception as e:
            logging.error(f"Error processing {image_name}: {str(e)}")
            return False

    # Process images in parallel
    successful = failed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_single_image, img) for img in image_names]

        from concurrent.futures import as_completed
        from tqdm import tqdm
        with tqdm(total=len(image_names), desc=f"Moving {split_type} {data_type} files") as pbar:
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)

    logging.info(f"\nCompleted moving {split_type} images:")
    return successful, failed
    
def generate_statistics_file(df: pd.DataFrame, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, train_ids: List, val_ids: List, test_ids: List, mode: str = "face-only"):
    """
    Generates a statistics file with dataset split information, including percentages.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Original DataFrame with all images and annotations.
    df_train : pd.DataFrame
        Training set DataFrame.
    df_val : pd.DataFrame
        Validation set DataFrame.
    df_test : pd.DataFrame
        Test set DataFrame.
    train_ids : List
        List of training child IDs.
    val_ids : List
        List of validation child IDs.
    test_ids : List
        List of test child IDs.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = BasePaths.LOGGING_DIR / f"split_distribution_face_det_{timestamp}.txt"
    
    class_columns = [col for col in df.columns if col not in ['filename', 'id', 'has_annotation', 'child_id']]
    
    # Calculate totals based on original balanced dataset (before test enhancement)
    original_total = len(df_train) + len(df_val) + len(df_test[df_test['has_annotation'] == True])
    
    with open(file_path, "w") as f:
        f.write(f"Dataset Split Information - {timestamp}\n")
        f.write(f"*** DETECTION MODE: {mode.upper()} ***\n")
        f.write("\n")
        
        # Original distribution (before test enhancement)
        f.write(f"Original Balanced Distribution (before test enhancement):\n")
        f.write(f"Total Images: {original_total}\n")
        
        if class_columns:
            # If age-binary, include class id numbers using FaceConfig mapping if available
            try:
                label_to_id = {v: k for k, v in FaceConfig.MODEL_CLASS_ID_TO_LABEL.items()}
            except Exception:
                label_to_id = {}

            for col in class_columns:
                original_count = df_train[col].sum() + df_val[col].sum() + df_test[df_test['has_annotation'] == True][col].sum()
                if original_total > 0:
                    pct = original_count / original_total
                else:
                    pct = 0

                if col in label_to_id:
                    f.write(f"Class {label_to_id[col]} {col}: {original_count} images ({pct:.2%})\n")
                else:
                    f.write(f"Class {col}: {original_count} images ({pct:.2%})\n")
        f.write("\n")

        f.write("Split Distribution (within each split):\n")
        f.write("--------------------------------------------------\n\n")

        def write_split_info(split_name, split_df):
            total_split = len(split_df)
            
            # Calculate face coverage (frames with any faces)
            frames_with_faces = split_df[split_df['has_annotation'] == True]
            face_coverage = len(frames_with_faces) / total_split if total_split > 0 else 0
            
            f.write(f"{split_name} Set:\n")
            f.write(f"Total Images: {total_split}\n")
            f.write(f"Face Coverage: {len(frames_with_faces)} ({face_coverage:.2%}) - frames with any faces\n")
            f.write(f"No Face: {total_split - len(frames_with_faces)} ({1-face_coverage:.2%}) - frames without faces\n")
            
            for col in class_columns:
                count = split_df[col].sum()
                ratio = count / total_split if total_split > 0 else 0
                # Print label names in lowercase (e.g., 'child', 'adult') for readability
                f.write(f"{col}: {count} ({ratio:.2%}) - within split\n")
            f.write("\n")

        write_split_info("Train", df_train)
        write_split_info("Validation", df_val)
        write_split_info("Test", df_test)

        f.write("ID Distribution:\n")
        f.write(f"Training IDs: {len(train_ids)}, {train_ids}\n")
        f.write(f"Validation IDs: {len(val_ids)}, {val_ids}\n")
        f.write(f"Test IDs: {len(test_ids)}, {test_ids}\n\n")

        f.write("ID Overlap Check:\n")
        train_val_overlap = set(train_ids).intersection(val_ids)
        train_test_overlap = set(train_ids).intersection(test_ids)
        val_test_overlap = set(val_ids).intersection(test_ids)
        if train_val_overlap or train_test_overlap or val_test_overlap:
            f.write("Overlap found: Yes\n")
        else:
            f.write("Overlap found: No\n")
        
        logging.info(f"Statistics file generated at: {file_path}")
    
def split_data(annotation_folder: Path, mode: str = "face-only", data_type: str = "detection"):
    """
    This function prepares the dataset for face detection YOLO training by splitting the images into train, val, and test sets.
    
    Parameters:
    ----------
    annotation_folder: Path
        Path to the directory containing label files.
    mode: str
        The mode for class mapping ('face-only' or 'age-binary').
    data_type: str
        Type of data processing ('detection' or 'classification').
    """
    logging.info(f"Starting dataset preparation for face detection in mode: {mode}")
    
    try:
        # --- 2. Get All Positive Images and Negative Candidates ---
        positive_images, negative_candidates = get_total_number_of_annotated_frames(annotation_folder)
        
        # save negative candidates for debugging
        neg_cand_file = BasePaths.LOGGING_DIR / f"negative_candidates_face_det_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(neg_cand_file, 'w') as f:
            for child_id, frames in negative_candidates.items():
                f.write(f"{child_id}: {frames}\n")
        
        if not positive_images:
            logging.error("No annotated images found.")
            return
        
        # --- 3. Get initial DataFrame from POSITIVE images ---
        df = get_class_distribution(positive_images, annotation_folder, mode)
        
        if df.empty:
            logging.error("DataFrame is empty. Check class distribution function.")
            return
        
        splits_to_move = []
        train, val, test, df_train, df_val, df_test = split_by_child_id(df, negative_candidates, len(positive_images), annotation_folder, mode, data_type)
        splits_to_move = [("train", train), ("val", val), ("test", test)]

        # Get the IDs for logging
        train_ids = df_train['child_id'].unique().tolist() if 'child_id' in df_train.columns else []
        val_ids = df_val['child_id'].unique().tolist() if 'child_id' in df_val.columns else []
        test_ids = df_test['child_id'].unique().tolist() if 'child_id' in df_test.columns else []

        # --- 5. Generate Statistics and Move Files ---
        generate_statistics_file(df, df_train, df_val, df_test, train_ids, val_ids, test_ids, mode=mode)
        
        for split_name, split_set in splits_to_move:
            if split_set:
                successful, failed = move_images(
                    image_names=split_set,
                    split_type=split_name,
                    label_path=annotation_folder,
                    n_workers=4,
                    data_type=data_type,
                    mode=mode,
                    df_split={'train': df_train, 'val': df_val, 'test': df_test}.get(split_name) if data_type == "classification" else None
                )
                logging.info(f"{split_name}: Moved {successful}, Failed {failed}")
            else:
                logging.warning(f"No images for {split_name} split")
    
    except Exception as e:
        logging.error(f"Error processing face detection: {str(e)}")
        raise
    
    logging.info(f"Completed dataset preparation for face detection in mode: {mode}")

def main():
    parser = argparse.ArgumentParser(description='Create input files for face detection YOLO training')
    parser.add_argument('--mode', choices=["face-only", "age-binary"], default="face-only",
                        help='Select the detection mode')
    parser.add_argument('--type', choices=["detection", "classification"], default="detection",
                        help='Select the output data format type (detection for YOLO, classification for single class ID)') # ADDED argument
    parser.add_argument('--fetch-annotations', action='store_true',
                        help='Fetch and save annotations from database (default: False)')
    args = parser.parse_args()
    
    # The classification logic is specifically tied to 'age-binary' mode, which is 2+1 classes (adult, child, no_face)
    if args.type == "classification" and args.mode == "face-only":
        logging.error("Classification mode is only supported with '--mode age-binary'. Falling back to detection.")
        args.type = "detection"
    
    CONSTANTS = get_data_constants(args.type)
    
    try:            
        if args.fetch_annotations:
            # Output annotations to the correct location (standard or retrain folder)
            labels_output_dir = CONSTANTS.LABELS_INPUT_DIR
            anns = fetch_all_annotations(FaceConfig.DATABASE_CATEGORY_IDS)
            save_annotations(anns, output_dir=labels_output_dir, mode=args.mode, data_type=args.type)
        
        split_data(CONSTANTS.LABELS_INPUT_DIR,
                   mode=args.mode,
                   data_type=args.type)
                                   
    except Exception as e:
        logging.error(f"Failed: {e}")
        raise
                                   
if __name__ == "__main__":
    main()