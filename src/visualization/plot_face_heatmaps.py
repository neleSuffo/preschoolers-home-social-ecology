"""
src/visualization/plot_face_heatmaps.py

Reconstructs 2D egocentric social space from logarithmic face proximity and
azimuthal projection across social states (Interacting, Available, Alone).
Generates KDE contour heatmaps and computes face presence rates (FDFR).
"""

import argparse
import logging
from pathlib import Path
import sqlite3
from typing import Optional, Tuple

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import AnalysisConfig, DetectionConfig, VideoSpecs
from src.constants import AnalysisPaths, BasePaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLED_FPS: float = VideoSpecs.FPS / AnalysisConfig.SAMPLE_RATE  # 30 / 10 = 3.0 fps
EXCLUSION_FRAMES: int = AnalysisConfig.EXCLUSION_SECONDS * VideoSpecs.FPS  # 30s * 30 fps = 900 frames


def filter_initial_setup_frames(
    df: pd.DataFrame,
    exclusion_frames: int = EXCLUSION_FRAMES,
    video_col: str = "video_name",
    frame_col: str = "frame_number",
) -> pd.DataFrame:
    """
    Exclude the first 30 seconds of recording setup only for starter videos ending in '_01'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing frame numbers and video identifiers.
    exclusion_frames : int, default=900
        Number of frames to truncate from the session start.
    video_col : str, default='video_name'
        Column name for video names.
    frame_col : str, default='frame_number'
        Column name for chronological frame index.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with camera-setup frames excluded.
    """
    df_clean = df.copy()

    is_starter_video = (
        df_clean[video_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\.[^.]+$", "", regex=True)
        .str.contains(r"(?:^|[_/-])01$", regex=True)
    )

    is_excluded_startup = is_starter_video & (df_clean[frame_col] < exclusion_frames)
    dropped_count = int(is_excluded_startup.sum())
    starter_vids_count = df_clean.loc[is_starter_video, video_col].nunique()

    logger.info(
        "Startup Filter: Evaluated %d starter ('_01') recordings. "
        "Excluded %d setup detections (frames 0-%d).",
        starter_vids_count,
        dropped_count,
        exclusion_frames - 1,
    )

    return df_clean[~is_excluded_startup].copy()


def load_or_create_matched_full_data(
    segments_csv_path: Path,
    cache_path: Path,
    db_path: Path = AnalysisPaths.INFERENCE_DB,
    sample_rate: int = AnalysisConfig.SAMPLE_RATE,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Query SQLite for face detections, join interaction segment states, and cache results.

    Parameters
    ----------
    segments_csv_path : Path
        Path to interaction_segments.csv.
    cache_path : Path
        Path to save/load tabular matched detections.
    db_path : Path, default=AnalysisPaths.INFERENCE_DB
        Path to inference.db.
    sample_rate : int, default=10
        Sub-sampling frame interval.
    force_recompute : bool, default=False
        If True, ignores existing cache and re-runs the SQL query and spatial join.

    Returns
    -------
    pd.DataFrame
        Dataframe containing matched face detections with social state labels.
    """
    if cache_path.exists() and not force_recompute:
        logger.info("Loading cached matched detections from: %s", cache_path)
        return pd.read_csv(cache_path)

    logger.info("Reading interaction segments from: %s", segments_csv_path)
    df_segments = pd.read_csv(segments_csv_path)

    df_segments["video_name_clean"] = (
        df_segments["video_name"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.[^.]+$", "", regex=True)
    )
    df_segments["segment_start"] = df_segments["segment_start"].astype(int)
    df_segments["segment_end"] = df_segments["segment_end"].astype(int)
    df_segments["social_state"] = df_segments.get(
        "social_state", df_segments.get("interaction_type", "Unknown")
    ).astype(str).str.capitalize()

    logger.info("Connecting to database at: %s", db_path)
    with sqlite3.connect(db_path) as conn:
        # Check if PersistentExclusions table exists
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='PersistentExclusions'"
        )
        has_exclusions = cursor.fetchone() is not None

        exclusion_clause = (
            "AND f.detection_id NOT IN ("
            "    SELECT detection_id FROM PersistentExclusions "
            "    WHERE type = 'face' AND detection_id IS NOT NULL"
            ")"
            if has_exclusions
            else ""
        )

        face_query = f"""
            SELECT 
                v.video_name,
                f.detection_id,
                f.frame_number,
                f.x_min,
                f.y_min,
                f.x_max,
                f.y_max,
                f.proximity,
                f.confidence_score
            FROM FaceDetections f
            JOIN Videos v ON f.video_id = v.video_id
            WHERE f.frame_number % {sample_rate} = 0
            {exclusion_clause}
        """
        try:
            df_faces = pd.read_sql_query(face_query, conn)
        except sqlite3.OperationalError:
            # Fallback if primary key in Videos is named 'id' instead of 'video_id'
            df_faces = pd.read_sql_query(
                face_query.replace("f.video_id = v.video_id", "f.video_id = v.id"),
                conn,
            )

    if df_faces.empty:
        logger.warning("No face detections found in %s", db_path)
        return pd.DataFrame()

    logger.info("Retrieved %d face detections from database.", len(df_faces))

    df_faces["video_name_clean"] = (
        df_faces["video_name"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.[^.]+$", "", regex=True)
    )

    # Filter setup frames
    df_faces = filter_initial_setup_frames(
        df_faces,
        exclusion_frames=EXCLUSION_FRAMES,
        video_col="video_name_clean",
        frame_col="frame_number",
    )

    df_faces["x_center"] = (df_faces["x_min"] + df_faces["x_max"]) / 2.0
    df_faces["y_center"] = (df_faces["y_min"] + df_faces["y_max"]) / 2.0

    logger.info("Executing in-memory SQL interval join with interaction segments...")
    with sqlite3.connect(":memory:") as mem_conn:
        df_faces.to_sql("faces", mem_conn, index=False)
        df_segments.to_sql("segments", mem_conn, index=False)

        mem_conn.execute("CREATE INDEX idx_faces ON faces(video_name_clean, frame_number)")
        mem_conn.execute("CREATE INDEX idx_segments ON segments(video_name_clean, segment_start, segment_end)")

        join_query = """
            SELECT 
                f.video_name_clean AS video_name,
                f.detection_id,
                f.frame_number,
                f.x_min,
                f.y_min,
                f.x_max,
                f.y_max,
                f.x_center,
                f.y_center,
                f.proximity,
                f.confidence_score,
                s.social_state
            FROM faces f
            INNER JOIN segments s 
                ON f.video_name_clean = s.video_name_clean 
                AND f.frame_number >= s.segment_start 
                AND f.frame_number <= s.segment_end
        """
        df_matched = pd.read_sql_query(join_query, mem_conn)

    logger.info("Total matched detections: %d", len(df_matched))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_matched.to_csv(cache_path, index=False)
    logger.info("Cached matched detections to: %s", cache_path)

    return df_matched


def compute_calibrated_3d_coordinates(
    df: pd.DataFrame,
    d_min: float = DetectionConfig.PROXIMITY_MIN_DISTANCE_M,
    d_max: float = DetectionConfig.PROXIMITY_MAX_DISTANCE_M,
    fov_deg: float = VideoSpecs.HORIZONTAL_FOV_DEG,
    frame_width: int = VideoSpecs.FRAME_WIDTH,
) -> pd.DataFrame:
    """
    Project bounding box centers into metric lateral (ego_x) and depth (ego_z) coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        Face detections containing 'proximity' and bounding box coordinates.
    d_min : float, default=0.30
        Distance in meters corresponding to maximum proximity (1.0).
    d_max : float, default=5.00
        Distance in meters corresponding to minimum proximity (0.0).
    fov_deg : float, default=140.0
        Camera horizontal field-of-view in degrees.
    frame_width : int, default=2304
        Camera sensor frame width in pixels.

    Returns
    -------
    pd.DataFrame
        Dataframe with appended 'distance_m', 'ego_x', and 'ego_z' columns.
    """
    df_ego = df.copy()

    if "x_center" not in df_ego.columns:
        df_ego["x_center"] = (df_ego["x_min"] + df_ego["x_max"]) / 2.0

    # Azimuth angle theta (-70° to +70° across 140° FOV)
    norm_x = (df_ego["x_center"] - (frame_width / 2.0)) / (frame_width / 2.0)
    theta_rad = norm_x * np.radians(fov_deg / 2.0)

    # Invert logarithmic area scaling: d = d_min * (d_max / d_min)**(1 - prox)
    prox_clipped = df_ego["proximity"].clip(0.0, 1.0)
    r_meters = d_min * np.power(d_max / d_min, 1.0 - prox_clipped)

    df_ego["distance_m"] = r_meters
    df_ego["ego_x"] = r_meters * np.sin(theta_rad)
    df_ego["ego_z"] = r_meters * np.cos(theta_rad)

    return df_ego


def plot_egocentric_social_space_3m_horizon(
    df_matched: pd.DataFrame,
    output_path: Path,
    df_stats: Optional[pd.DataFrame] = None,
    plot_horizon_m: float = DetectionConfig.PLOT_HORIZON_METERS,
    d_min: float = DetectionConfig.PROXIMITY_MIN_DISTANCE_M,
    d_max: float = DetectionConfig.PROXIMITY_MAX_DISTANCE_M,
    fov_deg: float = VideoSpecs.HORIZONTAL_FOV_DEG,
) -> None:
    """
    Render 3-panel 2D KDE contour map representing the child's egocentric visual field.

    Parameters
    ----------
    df_matched : pd.DataFrame
        Dataframe containing matched face detections with social states.
    output_path : Path
        Destination PNG path.
    df_stats : Optional[pd.DataFrame], default=None
        Summary statistics table containing Face Presence Rates (FDFR).
    plot_horizon_m : float, default=3.0
        Maximum distance threshold plotted in the contour cone.
    d_min : float, default=0.30
        Minimum distance parameter in meters.
    d_max : float, default=5.00
        Maximum distance parameter in meters.
    fov_deg : float, default=140.0
        Horizontal field of view in degrees.
    """
    states = ["Interacting", "Available", "Alone"]
    state_anchor_colors = {
        "Interacting": "#691633",  # Deep Burgundy
        "Available": "#8E7C3B",    # Muted Olive / Gold
        "Alone": "#69777B",        # Slate Gray
    }

    state_palettes = {
        "Interacting": LinearSegmentedColormap.from_list(
            "InteractingCmap", ["#ffffff", "#c75d7e", "#691633"]
        ),
        "Available": LinearSegmentedColormap.from_list(
            "AvailableCmap", ["#ffffff", "#cfbc76", "#8E7C3B"]
        ),
        "Alone": LinearSegmentedColormap.from_list(
            "AloneCmap", ["#ffffff", "#9faeb3", "#69777B"]
        ),
    }

    df_ego = compute_calibrated_3d_coordinates(
        df_matched, d_min=d_min, d_max=d_max, fov_deg=fov_deg
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 6.8), sharex=True, sharey=True, constrained_layout=True
    )

    half_fov_rad = np.radians(fov_deg / 2.0)
    angles_arc = np.linspace(-half_fov_rad, half_fov_rad, 150)

    cone_edge_x = plot_horizon_m * np.sin(half_fov_rad)
    cone_edge_z = plot_horizon_m * np.cos(half_fov_rad)

    for ax, state in zip(axes, states):
        sub = df_ego[df_ego["social_state"] == state]
        n_total = len(sub)

        in_horizon = sub[sub["distance_m"] <= plot_horizon_m]
        beyond_horizon = sub[sub["distance_m"] > plot_horizon_m]
        n_beyond = len(beyond_horizon)
        pct_beyond = (n_beyond / n_total * 100.0) if n_total > 0 else 0.0

        ax.set_facecolor("#fcfcfc")

        # 1. 2D KDE Contours
        if len(in_horizon) > 20:
            sample_size = min(35000, len(in_horizon))
            sample = in_horizon.sample(n=sample_size, random_state=42)

            sns.kdeplot(
                data=sample,
                x="ego_x",
                y="ego_z",
                ax=ax,
                cmap=state_palettes[state],
                fill=True,
                thresh=0.025,
                levels=14,
                alpha=0.95,
            )

        # 2. Camera Field-of-View Cone Boundaries (140°)
        ax.plot([0, -cone_edge_x], [0, cone_edge_z], color="#444444", linestyle="--", linewidth=1.1, alpha=0.6)
        ax.plot([0, cone_edge_x], [0, cone_edge_z], color="#444444", linestyle="--", linewidth=1.1, alpha=0.6)

        # 3. Softened Radial Distance Arcs up to 3m
        dist_rings = [
            (0.5, "0.5m"),
            (1.5, "1.5m"),
            (3.0, "3.0m"),
        ]
        for r, label in dist_rings:
            arc_x = r * np.sin(angles_arc)
            arc_z = r * np.cos(angles_arc)

            ax.plot(arc_x, arc_z, color="white", linestyle="-", linewidth=2.0, alpha=0.65, zorder=4)
            ax.plot(arc_x, arc_z, color="#555555", linestyle="-" if r == plot_horizon_m else "--", linewidth=1.0, alpha=0.55, zorder=5)

            ax.text(
                0.06,
                r - 0.09,
                label,
                color="#333333",
                fontsize=13,
                fontweight="bold",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="none",
                    alpha=0.8,
                ),
            )

        # 4. Child Origin Marker
        ax.plot(
            0,
            0,
            marker="o",
            markersize=11,
            color="#111111",
            markeredgecolor="white",
            markeredgewidth=1.8,
            zorder=10,
        )
        ax.annotate(
            "CHILD",
            xy=(0, -0.12),
            xycoords="data",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="#111111",
        )

        # 5. Distant Faces Reporting Box (> 3.0m)
        annotation_text = f">3.0m:\nn = {n_beyond:,}\n({pct_beyond:.1f}%)"
        ax.text(
            0.96,
            0.95,
            annotation_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=13,
            fontweight="semibold",
            color="#333333",
            bbox=dict(
                boxstyle="round,pad=0.35",
                fc="white",
                ec="#cccccc",
                alpha=0.9,
                linewidth=0.8,
            ),
        )

        # 6. Title and Quantitative Metrics
        mean_dist = sub["distance_m"].mean() if n_total > 0 else 0.0
        mean_prox = sub["proximity"].mean() if n_total > 0 else 0.0

        fdfr_str = " | FDFR: 0.0%"
        if df_stats is not None and not df_stats.empty:
            stat_row = df_stats[df_stats["Social State"] == state]
            if not stat_row.empty:
                fdfr_val = stat_row["Face Presence Rate (%)"].values[0]
                fdfr_str = f" | FDFR: {fdfr_val:.1f}%"

        ax.set_title(
            f"{state}\n"
            f"n = {n_total:,}{fdfr_str}\n"
            rf"$\mathit{{M}}_{{\mathrm{{distance}}}}$ = {mean_dist:.2f}m | $\mathit{{M}}_{{\mathrm{{proximity}}}}$ = {mean_prox:.2f}",
            fontsize=16,
            fontweight="bold",
            color=state_anchor_colors[state],
            pad=10,
        )

        ax.tick_params(axis="both", which="major", labelsize=10.5)
        ax.set_xlim(-cone_edge_x - 0.25, cone_edge_x + 0.25)
        ax.set_ylim(-0.38, plot_horizon_m + 0.2)
        ax.set_aspect("equal")
        ax.set_xlabel("Horizontal Position (Left ← 0 → Right) [m]", fontsize=14, labelpad=6)

    axes[0].set_ylabel("Distance in Front of Child [m]", fontsize=14, labelpad=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved egocentric space plot to: %s", output_path)


def compute_segment_and_detection_statistics(
    segments_csv_path: Path,
    df_matched: pd.DataFrame,
    output_summary_csv: Path,
    sampled_fps: float = SAMPLED_FPS,
    exclusion_seconds: int = AnalysisConfig.EXCLUSION_SECONDS,
) -> pd.DataFrame:
    """
    Compute duration, frame capacity, and face detection frequencies per social state.

    Parameters
    ----------
    segments_csv_path : Path
        Path to interaction_segments.csv.
    df_matched : pd.DataFrame
        Dataframe containing matched face detections.
    output_summary_csv : Path
        Destination CSV for summary statistics.
    sampled_fps : float, default=3.0
        Sampled analysis rate in frames per second.
    exclusion_seconds : int, default=30
        Startup boundary exclusion margin in seconds.

    Returns
    -------
    pd.DataFrame
        Summary statistics table per social state.
    """
    logger.info("Computing summary statistics from segments: %s", segments_csv_path)
    df_seg = pd.read_csv(segments_csv_path)

    df_seg["video_name_clean"] = (
        df_seg["video_name"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.[^.]+$", "", regex=True)
    )
    df_seg["social_state"] = df_seg.get(
        "social_state", df_seg.get("interaction_type", "Unknown")
    ).astype(str).str.capitalize()

    # Truncate first 30 seconds for '_01' starter recordings
    is_starter = df_seg["video_name_clean"].str.contains(r"(?:^|[_/-])01$", regex=True)
    adj_start = df_seg["start_time_sec"].copy()
    adj_start[is_starter] = adj_start[is_starter].clip(lower=exclusion_seconds)
    df_seg["effective_duration_s"] = (df_seg["end_time_sec"] - adj_start).clip(lower=0.0)

    state_summary = []
    states = ["Interacting", "Available", "Alone"]

    for state in states:
        sub_seg = df_seg[df_seg["social_state"] == state]
        total_seconds = float(sub_seg["effective_duration_s"].sum())
        total_hours = total_seconds / 3600.0
        total_minutes = total_seconds / 60.0

        total_sampled_frames = int(round(total_seconds * sampled_fps))

        sub_faces = df_matched[df_matched["social_state"] == state]
        total_face_dets = len(sub_faces)

        if not sub_faces.empty and "video_name" in sub_faces.columns:
            frames_with_faces = (
                sub_faces.groupby(["video_name", "frame_number"])
                .size()
                .shape[0]
            )
        else:
            frames_with_faces = 0

        face_presence_rate_pct = (
            (frames_with_faces / total_sampled_frames * 100.0)
            if total_sampled_frames > 0
            else 0.0
        )
        dets_per_sampled_frame = (
            (total_face_dets / total_sampled_frames)
            if total_sampled_frames > 0
            else 0.0
        )
        dets_per_hour = (
            (total_face_dets / total_hours)
            if total_hours > 0
            else 0.0
        )

        state_summary.append({
            "Social State": state,
            "Total Segments": len(sub_seg),
            "Total Duration (Hours)": round(total_hours, 2),
            "Total Duration (Minutes)": round(total_minutes, 1),
            "Total Sampled Frames": total_sampled_frames,
            "Total Face Detections": total_face_dets,
            "Frames with ≥1 Face": frames_with_faces,
            "Face Presence Rate (%)": round(face_presence_rate_pct, 2),
            "Faces / Sampled Frame": round(dets_per_sampled_frame, 3),
            "Faces / Hour": round(dets_per_hour, 1),
        })

    df_stats = pd.DataFrame(state_summary)
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(output_summary_csv, index=False, sep=";")
    logger.info("Saved state statistics to: %s", output_summary_csv)
    return df_stats


def process_dataset_mode(
    dataset_mode: str,
    output_dir: Path,
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    """
    Load data, execute startup filtering, and compute presence statistics.

    Parameters
    ----------
    dataset_mode : str
        'gt' for ground truth benchmark, or 'full' for complete corpus.
    output_dir : Path
        Directory to store outputs and intermediate caches.
    force_recompute : bool, default=False
        Whether to re-extract detections from SQLite.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Path]
        Filtered detections dataframe, summary statistics dataframe, and output figure path.
    """
    if dataset_mode == "gt":
        segments_csv = AnalysisPaths.GT_SEGMENTS_CSV
        cache_path = AnalysisPaths.GT_MATCHED_FACES_CSV
        prefix = "gt_"
    else:
        segments_csv = AnalysisPaths.INTERACTION_SEGMENTS_CSV
        cache_path = AnalysisPaths.FULL_MATCHED_FACES_CSV
        prefix = "full_"

    output_stats_csv = output_dir / f"{prefix}face_heatmap_stats.csv"
    output_figure = output_dir / f"{prefix}face_heatmaps_per_state.png"

    if cache_path.exists() and not force_recompute:
        logger.info("Loading cached face detections from: %s", cache_path)
        df_data = pd.read_csv(cache_path)
    else:
        logger.info("Recomputing matched face data for mode '%s'...", dataset_mode)
        df_data = load_or_create_matched_full_data(
            segments_csv_path=segments_csv,
            cache_path=cache_path,
            force_recompute=True,
        )

    if "video_name" not in df_data.columns and "video_name_clean" in df_data.columns:
        df_data["video_name"] = df_data["video_name_clean"]

    df_filtered = filter_initial_setup_frames(
        df_data,
        exclusion_frames=EXCLUSION_FRAMES,
        video_col="video_name",
        frame_col="frame_number",
    )

    df_stats = compute_segment_and_detection_statistics(
        segments_csv_path=segments_csv,
        df_matched=df_filtered,
        output_summary_csv=output_stats_csv,
        sampled_fps=SAMPLED_FPS,
        exclusion_seconds=AnalysisConfig.EXCLUSION_SECONDS,
    )

    return df_filtered, df_stats, output_figure


def main() -> None:
    """CLI entrypoint for generating 2D egocentric face heatmaps."""
    parser = argparse.ArgumentParser(description="Calibrated Egocentric Social Space Analysis")
    parser.add_argument(
        "--dataset",
        choices=["gt", "full"],
        default="gt",
        help="Target dataset: 'gt' (ground truth benchmark) or 'full' (full corpus)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=AnalysisPaths.HEATMAP_DIR,
        help="Directory to store generated plots and statistics CSVs",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Recompute detection matching from SQLite instead of loading cache",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_filtered, df_stats, output_figure = process_dataset_mode(
        dataset_mode=args.dataset,
        output_dir=args.output_dir,
        force_recompute=args.force_recompute,
    )

    plot_egocentric_social_space_3m_horizon(
        df_matched=df_filtered,
        output_path=output_figure,
        df_stats=df_stats,
        plot_horizon_m=DetectionConfig.PLOT_HORIZON_METERS,
        d_min=DetectionConfig.PROXIMITY_MIN_DISTANCE_M,
        d_max=DetectionConfig.PROXIMITY_MAX_DISTANCE_M,
        fov_deg=VideoSpecs.HORIZONTAL_FOV_DEG,
    )


if __name__ == "__main__":
    main()