"""
Frame-Level Social Interaction Analysis Pipeline (segma)

This module integrates multimodal visual cues (YOLO face detections, person
detections, face proximity estimates) and acoustic cues (Key Child Speech [KCS],
Key Child-Directed Speech [KCDS], and Overheard Speech [OHS]) from a SQLite 
database to classify each sampled video frame into a provisional social state:
- Interacting (1)
- Available (2)
- Alone (3)

Optimization:
    Employs fully vectorized Pandas and NumPy operations for rolling window 
    computations and boolean mask filtering, replacing iterative row-wise logic.
"""

import logging
import sqlite3
import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Path configuration
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import DataPaths, Analysis
from config import AnalysisConfig, DataConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_persistent_tables(conn: sqlite3.Connection):
    """
    Creates persistent indexed aggregate tables in SQLite to accelerate queries.

    Pre-aggregates frame detections to avoid re-evaluating exclusions (e.g.,
    faces or people depicted inside picture books) and expensive SQL MAX()
    aggregations across repeated pipeline runs.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection to the multimodal SQLite database.

    Notes
    -----
    Creates three indexed auxiliary tables:
    1. PersistentExclusions: IDs of face/person bounding boxes fully enclosed
       within book bounding boxes.
    2. CachedFaceAgg: Maximum face confidence and maximum proximity per sampled frame.
    3. CachedPersonAgg: Maximum person confidence per sampled frame.
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

def get_all_analysis_data(conn: sqlite3.Connection, 
                          video_list: list) -> pd.DataFrame:
    """
    Fetches and aligns multimodal frame-level features into a single DataFrame.

    Constructs a uniform temporal grid along each video at step size `r = SAMPLE_RATE`
    and merges visual and acoustic features via LEFT JOINs.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection to the multimodal SQLite database.
    video_list : list of str, optional
        Specific video file names to process. If None or empty, all videos in the
        database will be processed.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per sampled frame ($t$), sorted by `video_id` and
        `frame_number`, containing:
        - 'frame_number': Raw video frame index.
        - 'video_id': Unique integer identifier for video recording $v$.
        - 'video_name': String filename of recording $v$.
        - 'proximity': Maximum estimated face proximity score ($[0, 1]$).
        - 'face_conf': Maximum YOLO face detection confidence ($[0, 1]$).
        - 'person_conf': Maximum YOLO person detection confidence ($[0, 1]$).
        - 'has_kchi': Binary indicator (0/1) for Key Child Speech (KCS).
        - 'has_cds': Binary indicator (0/1) for Key Child-Directed Speech (KCDS).
        - 'has_ohs': Binary indicator (0/1) for Overheard Speech (OHS).
        - 'instant_presence_conf': $C^{\\text{inst}}_{v,t} = \\max(C^{\\text{face}}, C^{\\text{person}})$.
    """
    sample_rate = AnalysisConfig.SAMPLE_RATE
    placeholders = ','.join('?' for _ in video_list)
    
    query = f"""
    WITH RECURSIVE FrameGrid AS (
        SELECT video_id, video_name, 0 AS frame_number, max_frame FROM Videos
        WHERE video_name IN ({placeholders})
        UNION ALL
        SELECT video_id, video_name, frame_number + {sample_rate}, max_frame FROM FrameGrid
        WHERE frame_number + {sample_rate} <= max_frame
    )
    SELECT
        fg.frame_number, fg.video_id, fg.video_name,
        COALESCE(fa.proximity, 0) AS proximity,
        COALESCE(fa.face_conf, 0) AS face_conf,
        COALESCE(pa.person_conf, 0) AS person_conf,
        COALESCE(af.has_kchi, 0) AS has_kchi,
        COALESCE(af.has_ohs, 0) AS has_ohs,
        COALESCE(af.has_cds, 0) AS has_cds,
        MAX(COALESCE(pa.person_conf, 0), COALESCE(fa.face_conf, 0)) AS instant_presence_conf
    FROM FrameGrid fg
    LEFT JOIN CachedFaceAgg fa ON fg.frame_number = fa.frame_number AND fg.video_id = fa.video_id
    LEFT JOIN CachedPersonAgg pa ON fg.frame_number = pa.frame_number AND fg.video_id = pa.video_id
    LEFT JOIN AudioClassifications af ON fg.frame_number = af.frame_number AND fg.video_id = af.video_id
    ORDER BY fg.video_id, fg.frame_number
    """
    return pd.read_sql(query, conn, params=tuple(video_list))

def calculate_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes vectorized rolling window features for temporal smoothing and gating.

    Calculates:
    1. Centered rolling presence score ($S^{\\text{presence}}_{v,t}$) over 1 second.
    2. Long-term social anchor persistence ($A_{v,t}$) from high-confidence detections.
    3. Short-term visual memory buffer ($M_{v,t}$) bridging brief visual occlusions.
    4. Sustained backward-looking speech fractions for KCDS ($P^{\\text{KCDS}}_{v,t}$)
       and OHS ($P^{\\text{OHS}}_{v,t}$).

    Parameters
    ----------
    df : pd.DataFrame
        Frame-level DataFrame produced by `get_all_analysis_data`.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with windowed acoustic and visual feature columns:
        - 'presence_score': Centered moving average of $C^{\\text{inst}}_{v,t}$ ($S^{\\text{presence}}_{v,t}$).
        - 'is_high_conf_anchor': Instantaneous anchor flag ($a_{v,t}$).
        - 'person_seen_recently': Combined visual persistence condition ($A_{v,t} \\lor M_{v,t}$).
        - 'is_sustained_kcds': Boolean flag indicating if $P^{\\text{KCDS}}_{v,t} \\ge \\text{threshold}$.
        - 'is_sustained_ohs': Boolean flag indicating if $P^{\\text{OHS}}_{v,t} \\ge \\text{threshold}$.
    """
    fps = DataConfig.FPS
    sr = AnalysisConfig.SAMPLE_RATE
    samples_per_sec = fps / sr
    
    # 1. Visual Presence Score: Centered 1-second rolling average (N = 3 frames)
    # Debounces momentary single-frame visual flicker and sensor motion blur
    df['presence_score'] = df.groupby('video_id')['instant_presence_conf'].transform(
        lambda s: s.rolling(window=int(AnalysisConfig.PRESENCE_SCORE_FRAMES_PER_SEC), min_periods=1, center=True).mean()
    ).fillna(0)

    # 2. Rule 4 Component: High-Confidence Social Anchor (a_{v,t})
    # Activates if face proximity or face detection confidence exceeds high-certainty thresholds
    df['is_high_conf_anchor'] = (
        (df['proximity'] > AnalysisConfig.HIGH_CONFIDENCE_PROXIMITY_THRESHOLD) | 
        (df['face_conf'] > AnalysisConfig.HIGH_CONFIDENCE_FACE_CONFIDENCE)
    ).astype(int)    
    
    # Sustained anchor buffer: Centered rolling maximum across VISUAL_PERSISTENCE_SEC
    long_win = int(AnalysisConfig.VISUAL_PERSISTENCE_SEC * samples_per_sec)
    long_mem = df.groupby('video_id')['is_high_conf_anchor'].transform(
        lambda s: s.rolling(window=long_win, min_periods=1, center=True).max()
    ).fillna(0)
    
    # 3. Rule 4 Component: Short-Term Visual Memory (M_{v,t})
    # Detects recent general human presence within a shorter centered window
    short_win = int(AnalysisConfig.SHORT_TERM_VISUAL_MEMORY_SEC * samples_per_sec)
    short_mem = df.groupby('video_id')['instant_presence_conf'].transform(
        lambda s: s.rolling(window=short_win, min_periods=1, center=True).max()
    ).fillna(0) >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD
    
    # Combined Visual Persistence (R_4 trigger): person verified by either anchor or memory
    df['person_seen_recently'] = (long_mem == 1) | short_mem

    # Combined Visual Persistence (R_4 trigger): person verified by either anchor or memory
    # Speech fractions represent the density of vocalizations in preceding seconds
    sustained_kcds_win = int(AnalysisConfig.SUSTAINED_KCDS_WINDOW_SEC * samples_per_sec)
    df['is_sustained_kcds'] = df.groupby('video_id')['has_cds'].transform(
        lambda s: s.rolling(window=sustained_kcds_win).mean()
    ) >= AnalysisConfig.SUSTAINED_KCDS_THRESHOLD

    sustained_ohs_win = int(AnalysisConfig.SUSTAINED_OHS_WINDOW_SEC * samples_per_sec)
    df['is_sustained_ohs'] = df.groupby('video_id')['has_ohs'].transform(
        lambda s: s.rolling(window=sustained_ohs_win).mean()
    ) >= AnalysisConfig.SUSTAINED_OHS_THRESHOLD

    return df

def classify_frames(df: pd.DataFrame, 
                    social_state_mode: str = "tertiary") -> pd.DataFrame:
    """
    Classifies each sampled frame into a social state via hierarchical priority logic.

    Evaluates Rules 1–5 and assigns mutual provisional states following:
        Interacting (Priority 1) > Available (Priority 2) > Alone (Default 3)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with integrated features and rolling window variables.
    social_state_mode : str, default="tertiary"
        Classification granularity mode:
        - "tertiary": 3 classes (1: Interacting, 2: Available, 3: Alone).
        - "binary": 2 classes (1: Interacting, 2: Not Interacting).

    Returns
    -------
    pd.DataFrame
        DataFrame with individual boolean rule indicators ('rule1_' through 'rule5_')
        and the final integer 'interaction_type' assignment per frame.
    """
    # ------------------------------------------------------------
    # 1. Base Feature Flags
    # ------------------------------------------------------------
    # Cross-modal visual gating floor: verifies social partner is physically present
    is_visual_anchor = df['presence_score'] >= AnalysisConfig.AUDIO_VISUAL_GATING_FLOOR
    # Instantaneous detector certainty floor: filters out low-confidence hallucinations
    is_visual_confident = df['instant_presence_conf'] >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD
    # Face proximity floor: identifies interactive physical closeness
    is_close = df['proximity'] >= AnalysisConfig.PROXIMITY_THRESHOLD
    
    # ------------------------------------------------------------
    # 2. Define Rules 1 to 5
    # ------------------------------------------------------------
    # Rule 1: Conversational Turn-Taking (Audio)
    rule1_tt = df['is_audio_interaction'].astype(bool) if 'is_audio_interaction' in df else False
    
    # Rule 2: Physical Proximity (Visual confident AND close)
    rule2_prox = (is_visual_confident & is_close)
    
    # Rule 3: Sustained KCDS (Audio sustained CDS gated by visual presence)
    rule3_kcds = (df['is_sustained_kcds'].astype(bool) & is_visual_anchor)
    
    # Rule 4: Visual Persistence (Long-term anchor OR short-term memory)
    rule4_persistence = df['person_seen_recently'].astype(bool)
    
    # Rule 5: Intermittent Speech (Audio sustained OHS gated by visual presence)
    rule5_ohs = (df['is_sustained_ohs'].astype(bool) & is_visual_anchor)
    
    # Save explicit rule flags into the DataFrame
    df['rule1_turn_taking'] = rule1_tt
    df['rule2_close_proximity'] = rule2_prox
    df['rule3_sustained_kcds'] = rule3_kcds
    df['rule4_visual_persistence'] = rule4_persistence
    df['rule5_intermittent_ohs'] = rule5_ohs
    
    # ------------------------------------------------------------
    # 3. Top-Down Priority Assignment: Interacting > Available > Alone
    # ------------------------------------------------------------
    # Default State: Alone (3)
    df['interaction_type'] = 3
    
    # Priority 2: Available (Rule 4 or Rule 5)
    available_mask = rule4_persistence | rule5_ohs
    df.loc[available_mask, 'interaction_type'] = 2
    
    # Priority 1: Interacting (Rule 1, Rule 2, or Rule 3)
    interacting_mask = rule1_tt | rule2_prox | rule3_kcds
    df.loc[interacting_mask, 'interaction_type'] = 1

    # ------------------------------------------------------------
    # 4. Optional Binary Mode Mapping
    # ------------------------------------------------------------
    if social_state_mode == "binary":
        # Interacting remains 1; Available (2) and Alone (3) become Not Interacting (2)
        df.loc[df['interaction_type'] != 1, 'interaction_type'] = 2
        
    df['has_face'] = (df['face_conf'] >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD).astype(int)
    df['has_person'] = (df['person_conf'] >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD).astype(int)
    df['person_or_face_present'] = ((df['has_face'] == 1) | (df['has_person'] == 1)).astype(int)
    
    return df

def find_segments(video_df: pd.DataFrame, 
                  column_name: str) -> List[Dict]:
    """Extracts continuous uninterrupted temporal intervals for an active feature flag.

    Parameters
    ----------
    video_df : pd.DataFrame
        Frame data for a single video.
    column_name : str
        Boolean or binary column name to segment (e.g., 'has_kchi', 'has_cds').

    Returns
    -------
    list of dict
        List of continuous segment dictionaries containing:
        - 'start': Frame number of segment onset.
        - 'end': Frame number of segment offset.
        - 'type': Type string extracted from column name.
    """
    segments = []

    if column_name not in video_df.columns:
        return segments

    speech_frames = video_df[video_df[column_name] == 1]
    if speech_frames.empty:
        return segments

    # Determine frame numbers: use index if frame_number isn't explicit
    if 'frame_number' in speech_frames.columns:
        frame_numbers = speech_frames['frame_number'].values.astype(int)
    else:
        frame_numbers = speech_frames.index.values.astype(int)

    if len(frame_numbers) < 2:
        # Handle the single-frame or single-segment case
        return [{
            'start': int(frame_numbers[0]),
            'end': int(frame_numbers[-1]),
            'type': column_name.split('_')[-1]
        }]

    # Identify interval breaks where adjacent frame difference exceeds sample rate
    gaps = np.diff(frame_numbers) 
    break_indices = np.where(gaps > AnalysisConfig.SAMPLE_RATE)[0]
    
    current_start = frame_numbers[0]
    for i in break_indices:
        segments.append({
            'start': int(current_start),
            'end': int(frame_numbers[i]),
            'type': column_name.split('_')[-1]
        })
        current_start = frame_numbers[i + 1]

    # Append last segment
    segments.append({
        'start': int(current_start),
        'end': int(frame_numbers[-1]),
        'type': column_name.split('_')[-1]
    })

    return segments

def check_audio_interaction_turn_taking(df: pd.DataFrame, 
                                        fps: int) -> pd.Series:
    """Identifies conversational turn-taking bouts (Rule 1) across speaker turns.

    Chains consecutive vocalizations into communicative bouts based on speaker-specific
    gap ceilings:
    - Alternating speaker gap (KCS <-> KCDS) $\\le$ `MAX_TURN_TAKING_GAP_SEC` (strictly 5.0 s).
    - Same speaker gap $\\le$ `MAX_SAME_SPEAKER_GAP_SEC`.

    A bout qualifies as conversational turn-taking only if it contains vocalizations
    from both interlocutors (at least one KCS turn and at least one KCDS turn).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ['video_id', 'frame_number', 'has_kchi', 'has_cds'].
    fps : int
        Video recording frame rate (e.g., 30 fps), used to convert second thresholds to frames.

    Returns
    -------
    pd.Series
        Boolean Series indexed identically to `df`, indicating whether each sampled
        frame falls within a valid turn-taking bout ($R_{1,v,t}$).
    """
    if df is None or df.empty:
        return pd.Series(False, index=df.index if df is not None else [], name='is_audio_interaction')
    
    # Ensure the DataFrame is sorted by video_id and frame_number
    orig_index = df.index
    max_gap_frames = AnalysisConfig.MAX_TURN_TAKING_GAP_SEC * fps
    max_same_speaker_gap_frames = AnalysisConfig.MAX_SAME_SPEAKER_GAP_SEC * fps
    all_results = []

    # Process each video independently
    for video_id, video_df in df.groupby('video_id'):
        video_df = video_df.copy()
        video_df['orig_idx'] = video_df.index
        video_df.set_index('frame_number', inplace=True)
        video_df['is_audio_interaction'] = False
        
        # Extract discrete contiguous speech turns
        kchi_segments = find_segments(video_df, 'has_kchi')
        kcds_segments = find_segments(video_df, 'has_cds')
        all_segments = sorted(kchi_segments + kcds_segments, key=lambda x: x['start'])
        
        if not all_segments:
            all_results.append(video_df[['orig_idx', 'is_audio_interaction']])
            continue
            
        interaction_windows = []
        current_window = {
            'start': all_segments[0]['start'],
            'end': all_segments[0]['end'],
            'types': {all_segments[0]['type']}
        }

        # Progressively link speech turns into maximal bouts
        for seg in all_segments[1:]:
            is_same_type = seg['type'] in current_window['types']
            gap = seg['start'] - current_window['end']
            
            if is_same_type:
                # Same-speaker pause check
                if gap > max_same_speaker_gap_frames:
                    # Terminate bout if threshold exceeded
                    if 'kchi' in current_window['types'] and 'cds' in current_window['types']:
                        interaction_windows.append(current_window)
                    current_window = {'start': seg['start'], 'end': seg['end'], 'types': {seg['type']}}
                else:
                    current_window['end'] = seg['end']
                    current_window['types'].add(seg['type'])
            else: 
                # Alternating-speaker turn-taking gap check
                if gap <= max_gap_frames:
                    current_window['end'] = seg['end']
                    current_window['types'].add(seg['type'])
                else:
                    # Terminate bout if gap exceeds ceiling
                    if 'kchi' in current_window['types'] and 'cds' in current_window['types']:
                        interaction_windows.append(current_window)
                    current_window = {'start': seg['start'], 'end': seg['end'], 'types': {seg['type']}}

        # Final bout check
        if 'kchi' in current_window['types'] and 'cds' in current_window['types']:
            interaction_windows.append(current_window)

        # Flag all frames within confirmed turn-taking bouts as True
        for window in interaction_windows:
            video_df.loc[window['start'] : window['end'], 'is_audio_interaction'] = True
        
        all_results.append(video_df[['orig_idx', 'is_audio_interaction']])

    result_df = pd.concat(all_results, ignore_index=True).set_index('orig_idx')
    return result_df.reindex(orig_index)['is_audio_interaction'].fillna(False)

def main(db_path: Path, 
         output_dir: Path, 
         social_state_mode: str,
         video_list: list = None,
         hyperparameter_tuning: bool = False):
    """Executes the complete frame-level multimodal classification pipeline.

    Parameters
    ----------
    db_path : Path
        Filesystem path to the input SQLite database containing detections and audio tables.
    output_dir : Path
        Directory where output CSV artifacts will be written.
    social_state_mode : str, default="tertiary"
        Classification granularity: "tertiary" (3 classes) or "binary" (2 classes).
    video_list : list of str, optional
        Subset of video filenames to analyze. If None, analyzes all videos in `db_path`.
    hyperparameter_tuning : bool, default=False
        If True, preserves active tuning configuration set by the parent cross-validation
        controller and avoids resetting parameters via `apply_mode`.
    """
    if not hyperparameter_tuning:
        AnalysisConfig.apply_mode(social_state_mode)
        
    with sqlite3.connect(db_path) as conn:
        prepare_persistent_tables(conn)
        
        all_data = get_all_analysis_data(conn, video_list)
        
        # 1. Rule 1: Conversational Turn-Taking (Audio)
        all_data['is_audio_interaction'] = check_audio_interaction_turn_taking(all_data, DataConfig.FPS)
        
        # 2. Window Features (Rolling Presence, Visual Persistence, Sustained Audio)
        all_data = calculate_window_features(all_data)
        
        # 3. Rule Classification & Top-down Priority State Assignment
        all_data = classify_frames(all_data, social_state_mode=social_state_mode)
        
        output_path = output_dir / Analysis.FRAME_LEVEL_INTERACTIONS_CSV.name
        all_data.to_csv(output_path, index=False)
        logging.info(f"✅ Frame-level analysis saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorized Frame-Level Social Analysis")
    parser.add_argument('--social_state_mode', type=str, choices=['binary', 'tertiary'], default='tertiary', help="Classification mode: 'binary' (Interacting vs Not Interacting) or 'tertiary' (Interacting, Available, Alone).")
    parser.add_argument('--video_list', type=str, nargs='+', default=None, help="Optional list of specific video names to process. If omitted, evaluates all database videos.")
    parser.add_argument('--output_dir', type=str, default=None, help="Custom destination directory for output CSVs. Used by CV tuning scripts.")
    parser.add_argument('--hyperparameter_tuning', action='store_true', help="Flag indicating execution within automated hyperparameter search or cross-validation.")
    args = parser.parse_args()
    
    if args.output_dir:
        run_output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = Analysis.BASE_OUTPUT_DIR / f"analysis_{timestamp}"
    
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    main(
        db_path=Path(DataPaths.INFERENCE_DB_PATH), 
        output_dir=run_output_dir,
        social_state_mode=args.social_state_mode, 
        video_list=args.video_list,
        hyperparameter_tuning=args.hyperparameter_tuning
    )