# The social ecology of development: The changing structure of preschoolers' everyday experiences at home

[![DOI (Code)](https://zenodo.org/badge/1030951680.svg)](https://doi.org/10.5281/zenodo.23651513)
[![DOI (Detection Database)](https://zenodo.org/badge/DOI/10.5281/zenodo.23604952.svg)](https://doi.org/10.5281/zenodo.23604952)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Weights-ffd21e.svg)](https://huggingface.co/NeSuf/preschoolers-home-social-ecology-weights)

## Overview

This repository contains the complete code for the manuscript:  
**"The social ecology of development: The changing structure of preschoolers' everyday experiences at home"**.

It provides an end-to-end, open-source computational pipeline that allows researchers to both **fully reproduce the published paper results** and **deploy the pipeline on new child-centered egocentric datasets**:

1. **Reproduce Manuscript Results:** Re-run the empirical evaluations, cross-validation benchmarks, and Bayesian models from the paper, tracking developmental shifts in social state time allocations, conversational turn dynamics, child communicative agency, and the decreasing specificity of isolated social cues.
2. **Extract Multimodal Cues from New Video Data:** Ingest new, raw egocentric audio-visual recordings to detect faces and persons, compute proximity, filter book illustrations, and align these visual cues with frame-level speech diarization tokens (child speech, child-directed speech, and overheard speech).
3. **Reconstruct Social Ecology States:** Apply classification rules and PELT change-point segmentation with tunable parameters to classify continuous recordings into three distinct social states (*Interacting*, *Available*, and *Alone*). If ground-truth segment annotations are available for a new corpus, the built-in cross-validation engine can re-tune all thresholds and penalties to optimize performance for that specific recording context.

## Repository Structure

```text
.
├── .github/                          # CI/CD and repository workflows
│
├── data/                             # Tabular datasets, databases & sample frames
│   ├── inference.db                  # [Zenodo Download] Frame-level multimodal detections
│   ├── ground_truth_segments.csv     # Validation benchmark
│   ├── metadata.csv                  # Participant IDs, child age, and recording dates
│   ├── video_list_*.txt              # Video subsets for validation and tuning
│   └── frames_<state>/               # Sample frame sequences (alone, available, interacting)
│
├── models/                           # Fine-tuned model checkpoints (.pt)
│   ├── yolov12_face_best.pt          # Face & age classifier (Hugging Face)
│   ├── yolov12_person_best.pt        # Social partner detector (Hugging Face)
│   └── babyhubert_stc_best.pt        # Child vocalization classifier (Hugging Face)
│
├── paper/                            # Manuscript, Bayesian analyses & figures
│   ├── analysis/                     # Standalone Bayesian analysis workflows
│   │   ├── full_corpus_analysis.Rmd  # Full-corpus Bayesian models (N = 107)
│   │   └── gt_replication_analysis.Rmd # Ground-truth benchmark replication models
│   │
│   ├── data/                         # Tabular data & model bundles for Rmd compilation
│   │   ├── pipeline_output/          # Pipeline segment outputs
│   │   ├── cv_summary_final.csv      # 5-fold cross-validation performance metrics
│   │   ├── participants_demographics.csv # Sibling & kindergarten metadata
│   │   └── social_ecology_estimates.rds  # Precomputed posterior estimates bundle
│   │
│   ├── images/                       # Figures, heatmaps, and qualitative sequence frames
│   ├── apa.csl                       # APA 7th edition citation style
│   ├── bibliography.bib              # BibTeX reference database
│   ├── manuscript.Rmd                # Main manuscript source (papaja)
│   ├── manuscript.pdf                # Compiled manuscript PDF
│   ├── supplement.Rmd                # Supplementary Information source document
│   └── supplement.pdf                # Compiled Supplementary Information PDF
│
├── src/                              # Python processing pipeline
│   ├── config.py                     # Central configuration (thresholds, grid ranges, models)
│   ├── constants.py                  # Directory paths, model asset metadata, and anchors
│   │ 
│   ├── feature_extraction/           # Raw cue extraction & database ingestion
│   │   ├── download_model_weights.py # Hugging Face weight downloader
│   │   ├── main.py                   # Master CLI orchestrator for feature extraction
│   │   ├── run_video_detection.py    # Unified YOLO batch engine (face, person, book)
│   │   ├── run_audio_ingestion.py    # VTC audio diarization loader
│   │   ├── setup_inference_db.py     # SQLite schema and index builder
│   │   └── utils.py                  # Frame indexing and progress-log utilities
│   │
│   ├── models/                       # Inference architectures & qualitative visualizers
│   │   ├── proximity.py              # Logarithmic face proximity calculation
│   │   └── run_yolo.py               # Qualitative visualizer with TP/FP/FN overlays
│   │
│   ├── pipeline/                     # Core classification & aggregation machinery
│   │   ├── frame_level_rules.py      # Vectorized multi-modal rule evaluation (Rules 1–5)
│   │   ├── segment_aggregation.py    # PELT change-point segmentation & temporal smoothing
│   │   ├── pipeline_runner.py        # Execution runner for single/batch videos
│   │   └── pipeline_orchestrator.py  # Multi-process parallel batch controller
│   │
│   ├── evaluation/                   # Model validation & cross-validation tuning
│   │   ├── eval_segment_performance.py # Frame- and segment-level metric evaluations
│   │   └── tune_hyperparameters.py   # Group 5-fold cross-validation grid search
│   │
│   ├── research_questions/           # Downstream behavioral analyses (RQs)
│   │   ├── rq_02_kcs.py              # Child vocalization (KCS) extractor
│   │   ├── rq_03_kcds.py             # Child-directed & overheard speech extractor
│   │   ├── rq_04_turn_taking.py      # Conversational turns, latencies & agency
│   │   ├── rq_05_cue_specificity.py  # Diagnostic cue specificity analyzer
│   │   └── utils.py                  # Shared analytical helper functions
│   │
│   └── visualization/                # Qualitative plotting & video HUD rendering
│       ├── plot_face_heatmaps.py     # Spatial face distribution plots
│       ├── plot_multimodal_social_state_frames.py # Exemplar consecutive sequences
│       └── plot_segments_timeline.py # GT vs. predicted timeline bar plots
│
├── .gitignore
├── poetry.lock                       # Deterministic Python dependency lockfile
├── pyproject.toml                    # Poetry package configuration & dependencies
└── README.md                         # Repository documentation
```

> **Note on Large Assets:**  
> Large files (`data/inference.db` and checkpoints in `models/`) are managed externally. They can be downloaded directly from Zenodo and Hugging Face as described below.

## Installation & Setup

### 1. Python Environment (Poetry)
This project uses **Poetry** (with Python 3.12) for dependency locking:

```bash
# Clone the repository
git clone https://github.com/neleSuffo/preschoolers-home-social-ecology.git
cd preschoolers-home-social-ecology

# Install dependencies into virtual environment
poetry install
```

### 2. Download the Inference Database
Download `inference.db` from the Zenodo record and place it into the `data/` directory:

* **Zenodo Dataset DOI:** [10.5281/zenodo.23604952](https://doi.org/10.5281/zenodo.23604952)
* Target path: `data/inference.db`

### 3. Model Weights (Hugging Face)
Pre-trained model checkpoints are hosted on Hugging Face:  
**[huggingface.co/NeSuf/preschoolers-home-social-ecology-weights](https://huggingface.co/NeSuf/preschoolers-home-social-ecology-weights)**

Download the checkpoints into `models/`:
```bash
poetry run python -m src.feature_extraction.download_model_weights
```
*(Alternatively, pipeline and visualization scripts will automatically download required weights on demand if they are not detected locally).*

### 4. R Environment (Bayesian Statistics)
Compiling the manuscript and executing Bayesian models in `paper/analysis/` requires **R (version 4.2+)**:
```r
install.packages(c("tidyverse", "brms", "projpred", "cowplot", "patchwork", "magick", "papaja"))
```

## Pipeline Execution & Analysis

All pipeline scripts run from the repository root via Poetry.

### 1. Full Corpus Reconstruction
To run the multimodal rule engine and PELT change-point segmentation over the complete database:
```bash
poetry run python -m src.pipeline.pipeline_runner
```

### 2. Group Cross-Validation Benchmark
To run the group 5-fold cross-validation against the human ground-truth benchmark (reporting F1, Precision, and Recall):
```bash
poetry run python -m src.pipeline.pipeline_orchestrator --mode validation
```

### 3. Hyperparameter Tuning
To perform random-sampling or grid search over candidate a specified parameter search grid:
```bash
poetry run python -m src.pipeline.pipeline_orchestrator --mode tuning --max_combos 20
```

### 4. Segment Evaluation & Confusion Matrix
To evaluate final segment predictions against human ground truth and calculate confusion matrices:
```bash
poetry run python -m src.evaluation.eval_segment_performance
```

---

## Visualizations

### 1. Representative Frame Sequences with HUD Checklist
Render publication-ready qualitative exemplar frames showing YOLO detections, continuous proximity badges, acoustic tokens, and active rule checklists:
```bash
# State Exemplars
poetry run python -m src.visualization.plot_multimodal_social_state_frames --image_path data/frames_interacting --align left
poetry run python -m src.visualization.plot_multimodal_social_state_frames --image_path data/frames_alone --align right
poetry run python -m src.visualization.plot_multimodal_social_state_frames --image_path data/frames_available --align left
```

### 2. Video Timelines: Pipeline vs. Ground Truth
Generate comparative continuous timeline plots for individual video recordings:
```bash
# Single session timeline with legend
poetry run python -m src.visualization.plot_segments_timeline --video quantex_at_home_id262333_2024_12_01_03
poetry run python -m src.visualization.plot_segments_timeline --video quantex_at_home_id257609_2022_11_09_02 --no_legend
poetry run python -m src.visualization.plot_segments_timeline --video quantex_at_home_id260106_2022_11_13_01 --no_legend
```

### 3. Proximity and Face Location Heatmaps
Generate 2D spatial face distribution density heatmaps across developmental social states:
```bash
# Ground-truth validation sub-corpus
poetry run python -m src.visualization.plot_face_heatmaps --dataset gt

# Full unannotated home corpus (N = 107)
poetry run python -m src.visualization.plot_face_heatmaps --dataset full
```

---

## Model Benchmark Performance

Evaluating the automated reconstruction pipeline against the human ground-truth benchmark (30.19 hours across 14 children) yields the following performance across folds:

| Metric | Interacting | Available | Alone | Macro Average |
| :--- | :--- | :--- | :--- | :--- |
| **F1-Score** | 0.74 ± 0.06 | 0.57 ± 0.04 | 0.58 ± 0.12 | 0.63 ± 0.03 |
| **Precision** | 0.84 ± 0.02 | 0.50 ± 0.05 | 0.60 ± 0.07 | 0.65 ± 0.03 |
| **Recall** | 0.67 ± 0.10 | 0.66 ± 0.07 | 0.58 ± 0.19 | 0.64 ± 0.04 |

---

## Citation & Open Assets

If you use this pipeline, models, or data in your research, please cite:

> Suffo, N.-P., Suffo, A., Martin, P.-E., Kim, Y., Zahra, A., Haun, D., & Bohn, M. (2026). *The Social Ecology of Development: The Changing Structure of Preschoolers’ Everyday Experiences*. https://doi.org/10.31234/osf.io/bmz3e_v2

* **Code Pipeline (Zenodo):** [![DOI](https://zenodo.org/badge/1030951680.svg)](https://doi.org/10.5281/zenodo.23651513)
* **Inference Dataset (Zenodo):** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23604952.svg)](https://doi.org/10.5281/zenodo.23604952)
* **Model Checkpoints (Hugging Face):** [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffd21e.svg)](https://huggingface.co/NeSuf/preschoolers-home-social-ecology-weights)

```bibtex
@online{suffoSocialEcologyDevelopment2026,
  title = {The Social Ecology of Development: {{The}} Changing Structure of Preschoolers’ Everyday Experiences},
  shorttitle = {The Social Ecology of Development},
  author = {Suffo, Nele-Pauline and Suffo, Anas and Martin, Pierre-Etienne and Kim, Yujin and Zahra, Anam and Haun, Daniel and Bohn, Manuel},
  date = {2026-06-30},
  doi = {10.31234/osf.io/bmz3e_v2},
  url = {https://doi.org/10.31234/osf.io/bmz3e_v2},
  urldate = {2026-07-04},
  pubstate = {prepublished}
}
```