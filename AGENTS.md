# AGENTS.md - Pure ML / Kaggle Titanic

This document serves as the primary operational blueprint for automated agents and developers working on this repository.

## Directory Structure
- `data/`: Contains raw and processed dataset layers.
- `src/`: Contains the sequential execution scripts.
- `models/`: Stores trained models and pipelines.
- `submissions/`: Output directory for Kaggle predictions.
- `tests/`: Automated test suite for data pipeline validation.
- `experiments/`: Scratchpad for ad-hoc exploration.

## Sequential Execution Protocol
The pipeline relies on a specific sequence to prevent out-of-order execution anomalies. Always execute scripts sequentially in the following order:

1. `python -m src.features`
2. `python -m src.imputation`
3. `python -m src.modeling`
4. `python -m src.stacking`
5. `python -m src.final_submission`

## Target Leakage & Out-Of-Fold (OOF) Guardrails
To prevent target leakage and maintain a strict CV validation setup:
- All target encodings, group calculations, and complex imputations must be strictly computed within local CV fold boundaries.
- Stacking uses OOF predictions.
- For group-level target encoding (like WCG Two-Pass grouping) within a CV pipeline, store a mapping of `Group_ID -> list of (index, target)` during `fit`. In `transform`, use the passenger's DataFrame index to exclude their own target value from the group calculation to perfectly prevent target leakage.

## Pre-submission Checklist
- Ensure data shape hygiene (891 train rows, 418 test rows).
- Ensure final submission has exact Passenger IDs from 892 to 1309.
- Submission CSV must strictly output binary predictions (0 or 1) in the 'Survived' column.
