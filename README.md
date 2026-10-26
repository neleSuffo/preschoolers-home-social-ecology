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