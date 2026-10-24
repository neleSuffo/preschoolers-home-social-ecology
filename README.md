# The social ecology of development: The changing structure of preschoolers' everyday experiences at home

This repository contains the automated multimodal processing pipeline and the Bayesian statistical analysis for the paper:  
**"The social ecology of development: The changing structure of preschoolers' everyday experiences at home"**.

## Overview

We present a fully automated pipeline designed to bridge the "measurement gap" in developmental science. The framework transforms raw, long-form egocentric audio and video from home environments into a high-dimensional map of a child’s social ecology. This project enables the study of social interaction *in vivo* at a scale previously impossible with manual coding.

## Repository Structure

```text
.
├── .github/                          # CI/CD and repository workflows
│
├── data/                             # Precompiled databases & tabular metadata
│   ├── inference.db                  # Frame-level multimodal detections
│   ├── metadata.csv                  # Child age, recording dates, and participant IDs
│   └── ground_truth_segments.csv     # 30.19-hr validation benchmark
│
├── models/                           # Fine-tuned model checkpoints (.pt)
│   ├── yolov12_face_best.pt          # Hugging Face: Ego-child & adult face detector
│   ├── yolov12_person_best.pt        # Hugging Face: Social partner body detector
│   └── babyhubert_stc_best.pt        # Hugging Face: Key child vocalization classifier
│
├── paper/                            # Manuscript, Bayesian analyses & figures
│   ├── analysis/                     # Standalone Bayesian analysis workflows
│   │   ├── full_corpus_analysis.Rmd    # Full-corpus Bayesian models (N = 107)
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
│   │   ├── download_assets.py        # Hugging Face weight & Zenodo database downloader
│   │   ├── run_pipeline.py           # Master CLI orchestrator for feature extraction
│   │   ├── run_video_detection.py    # Unified YOLO batch engine (face, person, book)
│   │   ├── run_audio_rttm_ingestion.py # VTC audio diarization loader
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
│   │   ├── eval_irr.py               # Human inter-rater reliability (Cohen's Kappa)
│   │   ├── eval_segment_performance.py # Frame- and segment-level metric evaluations
│   │   └── tune_hyperparameters.py   # Group 5-fold cross-validation grid search
│   │
│   ├── research_questions/           # Downstream behavioral analyses (RQs)
│   │   ├── rq_02_kcs.py              # Child vocalization (KCS) extractor
│   │   ├── rq_03_kcds.py             # Child-directed & overheard speech extractor
│   │   ├── rq_04_turn_taking.py      # Conversational turns, latencies & agency
│   │   └── rq_05_interaction_composition.py # Diagnostic cue specificity analyzer
│   │
│   └── visualization/                # Qualitative plotting & video HUD rendering
│       ├── create_annotated_video.py # Diagnostic video generator with detection HUD
│       ├── plot_face_heatmaps.py     # Spatial face distribution plots
│       ├── plot_multimodal_social_state_frames.py # Exemplar consecutive sequences
│       └── plot_segments_timeline.py # GT vs. predicted timeline bar plots
│
├── .gitignore
├── poetry.lock                       # Deterministic Python dependency lockfile
├── pyproject.toml                    # Poetry package configuration & dependencies
└── README.md                         # Repository documentation
```

## Installation

### Python Environment (Pipeline)
This project uses **Poetry** for dependency management. This ensures that all researchers use the exact same versions of libraries (PyTorch, OpenCV, etc.) required for the pipeline to function correctly.

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone and Install dependencies:**
   ```bash
   git clone https://github.com/neleSuffo/preschool-social-ecology.git
   cd preschool-social-ecology
   poetry install
   ```

3. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```

### R Environment (Statistics)
The statistical analysis in `results_analysis/` requires **R (version 4.2+)**. We recommend installing the following libraries:
* `brms`: For Bayesian Multilevel Modeling.
* `projpred`: For the Bayesian variable selection and horseshoe prior diagnostics.
* `tidyverse`: For general data wrangling and visualization.
* `cowplot`, `patchwork`, `magick`: For multi-panel scientific figure layout.

---

## Pipeline Workflow & Usage

To ensure all scripts find the internal modules, always set your `PYTHONPATH` from the root directory:
```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

### 1. Model Training (Optional)
Pre-trained model weights are located in `weights/`. If you wish to retrain or fine-tune specialized detection models from scratch:

* **Annotation Preprocessing:** Run `python feature_extraction/setup_annotations_db.py` to ingest XML annotations into a central SQL database.
* **Face/Person Models:** Navigate to the respective subfolders in `models/`. Use `create_input_txt_files.py --fetch-annotations` to prepare YOLO-formatted datasets, and train using `train_yolo.py`.
* **Speech Type:** We utilized the [VTC 2.0 framework](https://github.com/arxaqapi/vtc-finetune) to fine-tune Voice Type Classification for our specific home environment classes. The final weights are included in this repository.

### 2. Video Inference
Run the automated inference on your raw dataset to populate the results database:

```bash
python python feature_extraction/main.py --video_path /path/to/videos --models all
```
* **Arguments:** `--models` options include `person`, `face_proximity`, `speech_type`, `book`, or `all`.
* **Output:** `inference.db` containing frame-by-frame predictions.

### 3. Social Ecology Reconstruction (Analysis)
This stage converts raw ML predictions into meaningful social states (Interaction, Available, Alone).

#### Hyperparameter Tuning
Run nested cross-fold validation to find the optimal smoothing and thresholding parameters:
```bash
python pipeline/pipeline_orchestrator.py --mode tuning --max_combos 50
```

#### Final Validation
To generate the performance metrics reported in the paper:
```bash
python pipeline/pipeline_orchestrator.py --mode validation
```

| Metric | Interacting | Available | Alone | Average |
| :--- | :--- | :--- | :--- | :--- |
| **F1** | 0.74 ± 0.06 | 0.57 ± 0.04 | 0.58 ± 0.12 | 0.63 ± 0.03 |
| **Precision** | 0.84 ± 0.02 | 0.50 ± 0.05 | 0.60 ± 0.07 | 0.65 ± 0.03 |
| **Recall** | 0.67 ± 0.10 | 0.66 ± 0.07 | 0.58 ± 0.19 | 0.64 ± 0.04 |

#### Full Dataset Processing
To generate the final CSV files for the statistical analysis:
```bash
python python pipeline/pipeline_runner.py --video_list video_list.txt
```

---
## Citation
If you use this pipeline or data in your research, please cite the associated preprint:
> Suffo, N.-P., Suffo, A., Martin, P.-E., Kim, Y., Zahra, A., Haun, D., & Bohn, M. (2026). *The Social Ecology of Development: The Changing Structure of Preschoolers’ Everyday Experiences*. https://doi.org/10.31234/osf.io/bmz3e_v1

```bibtex
@online{suffoSocialEcologyDevelopment2026,
  title = {The Social Ecology of Development: {{The}} Changing Structure of Preschoolers’ Everyday Experiences},
  shorttitle = {The Social Ecology of Development},
  author = {Suffo, Nele-Pauline and Suffo, Anas and Martin, Pierre-Etienne and Kim, Yujin and Zahra, Anam and Haun, Daniel and Bohn, Manuel},
  date = {2026-06-30},
  doi = {10.31234/osf.io/bmz3e_v1},
  url = {https://osf.io/bmz3e_v1},
  urldate = {2026-07-04},
  pubstate = {prepublished}
}
```
