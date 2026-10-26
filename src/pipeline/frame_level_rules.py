"""
src/pipeline/frame_level_rules.py

Frame-Level Multimodal Social Ecology Classification Pipeline.

This module integrates multimodal visual cues (YOLO face detections, person
detections, face proximity estimates) and acoustic cues (Key Child Speech [KCS],
Key Child-Directed Speech [KCDS], and Overheard Speech [OHS]) from SQLite to classify
each sampled video frame into a provisional social state:
- Interacting (1)
- Available (2)
- Alone (3)

Applies vectorized Pandas and NumPy operations for rolling window computations,
temporal gating, and cross-modal rule fusion.
"""

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import AnalysisConfig, VideoSpecs
from src.constants import AnalysisPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_persistent_tables(conn: sqlite3.Connection) -> None:
    """
    Create persistent indexed aggregate tables in SQLite to accelerate queries.

    Pre-aggregates frame detections to avoid re-evaluating exclusions (e.g.,
    faces or people depicted inside picture books) and expensive SQL MAX()
    aggregations across repeated pipeline runs.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection to the multimodal SQLite database.
    """
    sample_rate = AnalysisConfig.SAMPLE_RATE

    # 1. Exclusion table: detects faces/bodies illustrated inside books
    conn.execute("""
        CREATE TABLE IF NOT EXISTS PersistentExclusions AS
        SELECT fd.detection_id, 'face' as type FROM FaceDetections fd
        JOIN BookDetections bd ON fd.frame_number = bd.frame_number AND fd.video_id = bd.video_id
        WHERE fd.x_min >= bd.x_min AND fd.y_min >= bd.y_min AND fd.x_max <= bd.x_max AND fd.y_max <= bd.y_max
        UNION ALL
        SELECT pc.detection_id, 'person' as type FROM PersonDetections pc
        JOIN BookDetections bd ON pc.frame_number = bd.frame_number AND pc.video_id = bd.video_id
        WHERE pc.x_min >= bd.x_min AND pc.y_min >= bd.y_min AND pc.x_max <= bd.x_max AND pc.y_max <= bd.y_max;
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pers_excl ON PersistentExclusions(detection_id);")

    # 2. Aggregated Face table: filtered by book exclusions and subsampled at sample_rate
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS CachedFaceAgg AS
        SELECT frame_number, video_id, MAX(proximity) AS proximity, MAX(confidence_score) AS face_conf
        FROM FaceDetections
        WHERE detection_id NOT IN (SELECT detection_id FROM PersistentExclusions WHERE type='face')
        AND frame_number % {sample_rate} = 0
        GROUP BY video_id, frame_number;
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_face_agg ON CachedFaceAgg(video_id, frame_number);")

    # 3. Aggregated Person table: filtered by book exclusions and subsampled at sample_rate
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS CachedPersonAgg AS
        SELECT frame_number, video_id, MAX(confidence_score) AS person_conf
        FROM PersonDetections
        WHERE detection_id NOT IN (SELECT detection_id FROM PersistentExclusions WHERE type='person')
        AND frame_number % {sample_rate} = 0
        GROUP BY video_id, frame_number;
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_person_agg ON CachedPersonAgg(video_id, frame_number);")


def get_all_analysis_data(
    conn: sqlite3.Connection,
    video_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fetch and align multimodal frame-level features into a single DataFrame.

    Constructs a uniform temporal grid along each video at step size `SAMPLE_RATE`
    and merges visual and acoustic features via LEFT JOINs.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection to the multimodal SQLite database.
    video_list : Optional[List[str]], default=None
        Specific video file names or stems to process. If None or empty, all videos in the
        database will be processed.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per sampled frame, sorted by `video_id` and `frame_number`.
    """
    sample_rate = AnalysisConfig.SAMPLE_RATE

    if video_list:
        clean_stems = [Path(v).stem for v in video_list]
        placeholders = ",".join("?" for _ in clean_stems)
        where_clause = f"WHERE video_name IN ({placeholders}) OR REPLACE(video_name, '.mp4', '') IN ({placeholders})"
        params = tuple(clean_stems + clean_stems)
    else:
        where_clause = ""
        params = ()

    query = f"""
    WITH RECURSIVE FrameGrid AS (
        SELECT video_id, video_name, 0 AS frame_number, max_frame FROM Videos
        {where_clause}
        UNION ALL
        SELECT video_id, video_name, frame_number + {sample_rate}, max_frame FROM FrameGrid
        WHERE frame_number + {sample_rate} <= max_frame
    )
    SELECT
        fg.frame_number, fg.video_id, fg.video_name,
        COALESCE(fa.proximity, 0.0) AS proximity,
        COALESCE(fa.face_conf, 0.0) AS face_conf,
        COALESCE(pa.person_conf, 0.0) AS person_conf,
        COALESCE(af.has_kchi, 0) AS has_kchi,
        COALESCE(af.has_ohs, 0) AS has_ohs,
        COALESCE(af.has_cds, 0) AS has_cds,
        MAX(COALESCE(pa.person_conf, 0.0), COALESCE(fa.face_conf, 0.0)) AS instant_presence_conf
    FROM FrameGrid fg
    LEFT JOIN CachedFaceAgg fa ON fg.frame_number = fa.frame_number AND fg.video_id = fa.video_id
    LEFT JOIN CachedPersonAgg pa ON fg.frame_number = pa.frame_number AND fg.video_id = pa.video_id
    LEFT JOIN AudioClassifications af ON fg.frame_number = af.frame_number AND fg.video_id = af.video_id
    ORDER BY fg.video_id, fg.frame_number
    """
    return pd.read_sql(query, conn, params=params)


def calculate_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute vectorized rolling window features for temporal smoothing and gating.

    Calculates:
    1. Centered rolling presence score over 1 second.
    2. Long-term social anchor persistence from high-confidence detections.
    3. Short-term visual memory buffer bridging brief visual occlusions.
    4. Sustained backward-looking speech fractions for KCDS and OHS.

    Parameters
    ----------
    df : pd.DataFrame
        Frame-level DataFrame produced by `get_all_analysis_data`.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with windowed acoustic and visual feature columns.
    """
    fps = VideoSpecs.FPS
    sr = AnalysisConfig.SAMPLE_RATE
    samples_per_sec = fps / sr

    # 1. Visual Presence Score: Centered 1-second rolling average (N = 3 frames at 3 Hz)
    df["presence_score"] = df.groupby("video_id")["instant_presence_conf"].transform(
        lambda s: s.rolling(window=int(AnalysisConfig.PRESENCE_SCORE_FRAMES_PER_SEC), min_periods=1, center=True).mean()
    ).fillna(0.0)

    # 2. Rule 4 Component: High-Confidence Social Anchor
    df["is_high_conf_anchor"] = (
        (df["proximity"] > AnalysisConfig.HIGH_CONFIDENCE_PROXIMITY_THRESHOLD)
        | (df["face_conf"] > AnalysisConfig.HIGH_CONFIDENCE_FACE_CONFIDENCE)
    ).astype(int)

    # Sustained anchor buffer: Centered rolling maximum across VISUAL_PERSISTENCE_SEC
    long_win = max(1, int(AnalysisConfig.VISUAL_PERSISTENCE_SEC * samples_per_sec))
    long_mem = df.groupby("video_id")["is_high_conf_anchor"].transform(
        lambda s: s.rolling(window=long_win, min_periods=1, center=True).max()
    ).fillna(0)

    # 3. Rule 4 Component: Short-Term Visual Memory
    short_win = max(1, int(AnalysisConfig.SHORT_TERM_VISUAL_MEMORY_SEC * samples_per_sec))
    short_mem = df.groupby("video_id")["instant_presence_conf"].transform(
        lambda s: s.rolling(window=short_win, min_periods=1, center=True).max()
    ).fillna(0.0) >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD

    # Combined Visual Persistence (R4 trigger): person verified by either anchor or memory
    df["person_seen_recently"] = (long_mem == 1) | short_mem

    # 4. Sustained Vocalization Windows (Backward-looking fractions)
    sustained_kcds_win = max(1, int(AnalysisConfig.SUSTAINED_KCDS_WINDOW_SEC * samples_per_sec))
    df["is_sustained_kcds"] = df.groupby("video_id")["has_cds"].transform(
        lambda s: s.rolling(window=sustained_kcds_win, min_periods=1).mean()
    ) >= AnalysisConfig.SUSTAINED_KCDS_THRESHOLD

    sustained_ohs_win = max(1, int(AnalysisConfig.SUSTAINED_OHS_WINDOW_SEC * samples_per_sec))
    df["is_sustained_ohs"] = df.groupby("video_id")["has_ohs"].transform(
        lambda s: s.rolling(window=sustained_ohs_win, min_periods=1).mean()
    ) >= AnalysisConfig.SUSTAINED_OHS_THRESHOLD

    return df


def find_segments(video_df: pd.DataFrame, column_name: str) -> List[Dict]:
    """
    Extract continuous uninterrupted temporal intervals for an active feature flag.
    
    Parameters
    ----------
    video_df : pd.DataFrame
        DataFrame containing frame-level features for a single video.
    column_name : str   
        Name of the binary column indicating active frames (e.g., 'has_kchi', 'has_cds').
        
    Returns
    -------
    List[Dict]
        List of dictionaries representing the start and end frame numbers of each segment.
    """
    segments = []

    if column_name not in video_df.columns:
        return segments

    speech_frames = video_df[video_df[column_name] == 1]
    if speech_frames.empty:
        return segments

    if "frame_number" in speech_frames.columns:
        frame_numbers = speech_frames["frame_number"].values.astype(int)
    else:
        frame_numbers = speech_frames.index.values.astype(int)

    if len(frame_numbers) < 2:
        return [{
            "start": int(frame_numbers[0]),
            "end": int(frame_numbers[-1]),
            "type": column_name.split("_")[-1],
        }]

    gaps = np.diff(frame_numbers)
    break_indices = np.where(gaps > AnalysisConfig.SAMPLE_RATE)[0]

    current_start = frame_numbers[0]
    for i in break_indices:
        segments.append({
            "start": int(current_start),
            "end": int(frame_numbers[i]),
            "type": column_name.split("_")[-1],
        })
        current_start = frame_numbers[i + 1]

    segments.append({
        "start": int(current_start),
        "end": int(frame_numbers[-1]),
        "type": column_name.split("_")[-1],
    })

    return segments


def check_audio_interaction_turn_taking(df: pd.DataFrame, fps: int = VideoSpecs.FPS) -> pd.Series:
    """
    Identify conversational turn-taking bouts (Rule 1) across speaker turns.

    Chains consecutive vocalizations into communicative bouts based on speaker-specific
    gap ceilings:
    - Alternating speaker gap (KCS <-> KCDS) <= MAX_TURN_TAKING_GAP_SEC (5.0s).
    - Same speaker gap <= MAX_SAME_SPEAKER_GAP_SEC (1.0s).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ['video_id', 'frame_number', 'has_kchi', 'has_cds'].
    fps : int, default=30
        Video recording frame rate.

    Returns
    -------
    pd.Series
        Boolean Series indicating whether each sampled frame falls within a turn-taking bout.
    """
    if df is None or df.empty:
        return pd.Series(False, index=df.index if df is not None else [], name="is_audio_interaction")

    orig_index = df.index
    max_gap_frames = AnalysisConfig.MAX_TURN_TAKING_GAP_SEC * fps
    max_same_speaker_gap_frames = AnalysisConfig.MAX_SAME_SPEAKER_GAP_SEC * fps
    all_results = []

    for _, video_df in df.groupby("video_id"):
        video_df = video_df.copy()
        video_df["orig_idx"] = video_df.index
        video_df.set_index("frame_number", inplace=True)
        video_df["is_audio_interaction"] = False

        kchi_segments = find_segments(video_df, "has_kchi")
        kcds_segments = find_segments(video_df, "has_cds")
        all_segments = sorted(kchi_segments + kcds_segments, key=lambda x: x["start"])

        if not all_segments:
            all_results.append(video_df[["orig_idx", "is_audio_interaction"]])
            continue

        interaction_windows = []
        current_window = {
            "start": all_segments[0]["start"],
            "end": all_segments[0]["end"],
            "types": {all_segments[0]["type"]},
        }

        for seg in all_segments[1:]:
            is_same_type = seg["type"] in current_window["types"]
            gap = seg["start"] - current_window["end"]

            if is_same_type:
                if gap > max_same_speaker_gap_frames:
                    if "kchi" in current_window["types"] and "cds" in current_window["types"]:
                        interaction_windows.append(current_window)
                    current_window = {"start": seg["start"], "end": seg["end"], "types": {seg["type"]}}
                else:
                    current_window["end"] = seg["end"]
                    current_window["types"].add(seg["type"])
            else:
                if gap <= max_gap_frames:
                    current_window["end"] = seg["end"]
                    current_window["types"].add(seg["type"])
                else:
                    if "kchi" in current_window["types"] and "cds" in current_window["types"]:
                        interaction_windows.append(current_window)
                    current_window = {"start": seg["start"], "end": seg["end"], "types": {seg["type"]}}

        if "kchi" in current_window["types"] and "cds" in current_window["types"]:
            interaction_windows.append(current_window)

        for window in interaction_windows:
            video_df.loc[window["start"]:window["end"], "is_audio_interaction"] = True

        all_results.append(video_df[["orig_idx", "is_audio_interaction"]])

    result_df = pd.concat(all_results, ignore_index=True).set_index("orig_idx")
    return result_df.reindex(orig_index)["is_audio_interaction"].fillna(False)


def classify_frames(
    df: pd.DataFrame,
    social_state_mode: str = "tertiary",
) -> pd.DataFrame:
    """
    Classify each sampled frame into a social state via hierarchical priority logic.

    Evaluates Rules 1–5 and assigns mutual provisional states:
        Interacting (Priority 1) > Available (Priority 2) > Alone (Default 3)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with integrated features and rolling window variables.
    social_state_mode : str, default="tertiary"
        Classification granularity: 'tertiary' (1, 2, 3) or 'binary' (1, 2).

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with rule flags and integer 'social_state'.
    """
    # 1. Base Feature Flags
    is_visual_anchor = df["presence_score"] >= AnalysisConfig.AUDIO_VISUAL_GATING_FLOOR
    is_visual_confident = df["instant_presence_conf"] >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD
    is_close = df["proximity"] >= AnalysisConfig.PROXIMITY_THRESHOLD

    # 2. Rule Evaluation
    rule1_tt = df["is_audio_interaction"].astype(bool) if "is_audio_interaction" in df else False
    rule2_prox = is_visual_confident & is_close
    rule3_kcds = df["is_sustained_kcds"].astype(bool) & is_visual_anchor
    rule4_persistence = df["person_seen_recently"].astype(bool)
    rule5_ohs = df["is_sustained_ohs"].astype(bool) & is_visual_anchor

    # Save canonical rule flags and aliases for downstream scripts
    df["rule1_turn_taking"] = rule1_tt
    df["rule2_close_proximity"] = rule2_prox
    df["rule3_sustained_kcds"] = rule3_kcds
    df["rule3_kcds_speaking"] = rule3_kcds  # Downstream compatibility alias
    df["rule4_visual_persistence"] = rule4_persistence
    df["rule5_intermittent_ohs"] = rule5_ohs
    df["rule5_sustained_ohs"] = rule5_ohs  # Downstream compatibility alias

    # 3. Top-Down Priority Assignment
    df["social_state"] = 3  # Default: Alone

    # Priority 2: Available
    available_mask = rule4_persistence | rule5_ohs
    df.loc[available_mask, "social_state"] = 2

    # Priority 1: Interacting
    interacting_mask = rule1_tt | rule2_prox | rule3_kcds
    df.loc[interacting_mask, "social_state"] = 1

    # 4. Binary Mode Mapping
    if social_state_mode == "binary":
        df.loc[df["social_state"] != 1, "social_state"] = 2

    df["has_face"] = (df["face_conf"] >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD).astype(int)
    df["has_person"] = (df["person_conf"] >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD).astype(int)
    df["person_or_face_present"] = ((df["has_face"] == 1) | (df["has_person"] == 1)).astype(int)

    return df


def main(
    db_path: Path = AnalysisPaths.INFERENCE_DB,
    output_dir: Optional[Path] = None,
    social_state_mode: str = "tertiary",
    video_list: Optional[List[str]] = None,
    hyperparameter_tuning: bool = False,
) -> Path:
    """
    Execute the complete frame-level multimodal classification pipeline.

    Parameters
    ----------
    db_path : Path, default=AnalysisPaths.INFERENCE_DB
        Path to SQLite database containing detections and audio tables.
    output_dir : Optional[Path], default=None
        Directory where frame_level_social_interactions.csv will be saved.
    social_state_mode : str, default='tertiary'
        Granularity: 'tertiary' or 'binary'.
    video_list : Optional[List[str]], default=None
        Subset of video filenames to analyze. If None, analyzes all videos.
    hyperparameter_tuning : bool, default=False
        If True, prevents resetting active configuration attributes.

    Returns
    -------
    Path
        Path to the saved frame-level interactions CSV.
    """
    if not hyperparameter_tuning:
        AnalysisConfig.apply_mode(social_state_mode)

    dest_dir = output_dir or (AnalysisPaths.OUTPUT_FOLDER / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        prepare_persistent_tables(conn)
        all_data = get_all_analysis_data(conn, video_list)

        if all_data.empty:
            logger.warning("No frame data returned from database query.")
            output_path = dest_dir / AnalysisPaths.FRAME_INTERACTIONS_CSV.name
            all_data.to_csv(output_path, index=False)
            return output_path

        # 1. Rule 1: Turn-taking
        all_data["is_audio_interaction"] = check_audio_interaction_turn_taking(all_data, VideoSpecs.FPS)

        # 2. Window features
        all_data = calculate_window_features(all_data)

        # 3. Rule classification
        all_data = classify_frames(all_data, social_state_mode=social_state_mode)

        output_path = dest_dir / AnalysisPaths.FRAME_INTERACTIONS_CSV.name
        all_data.to_csv(output_path, index=False)
        logger.info("Frame-level analysis saved to: %s", output_path)
        return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorized Frame-Level Social Analysis")
    parser.add_argument(
        "--social_state_mode",
        choices=["binary", "tertiary"],
        default="tertiary",
        help="Classification mode: 'binary' or 'tertiary'",
    )
    parser.add_argument(
        "--video_list",
        nargs="+",
        default=None,
        help="Optional list of specific video names to process",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Custom destination directory for output CSVs",
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default=AnalysisPaths.INFERENCE_DB,
        help="Path to SQLite inference database",
    )
    parser.add_argument(
        "--hyperparameter_tuning",
        action="store_true",
        help="Flag indicating execution within automated hyperparameter search",
    )
    args = parser.parse_args()

    main(
        db_path=args.db_path,
        output_dir=args.output_dir,
        social_state_mode=args.social_state_mode,
        video_list=args.video_list,
        hyperparameter_tuning=args.hyperparameter_tuning,
    )