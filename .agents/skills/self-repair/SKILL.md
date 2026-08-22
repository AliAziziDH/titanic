---
name: self-repair
description: Autonomous diagnostic and surgical self-repair protocol for data pipelines and ML workflows.
version: 1.0.0
triggers:
  - test_failure
  - diagnostic_anomaly
  - pipeline_crash
---

# Self-Repair Skill

## Overview
The `self-repair` skill defines an autonomous protocol for diagnosing, isolating, and fixing defects in the Titanic ML pipeline without introducing regressions or target leakage.

## Mandatory Policies

### 1. Failing Test First Policy
- **Requirement**: Before modifying any production/pipeline code, create or run a localized test in the sandbox/test suite that reproduces the exact failure.
- **Verification**: Ensure the test fails under the exact conditions observed.
- **Resolution Gate**: The repair is considered viable only when the localized test passes alongside the full test suite (`pytest tests/ -v`).

### 2. Surgical Scope Policy
- **Requirement**: Confine edits strictly to the minimal lines and root cause identified.
- **Prohibitions**:
  - Do NOT perform speculative refactoring.
  - Do NOT rename unrelated variables or functions.
  - Do NOT alter unrelated files or dependencies.

## Execution Workflow

1. **Anomaly Isolation & Triage**
   - Capture error stack trace from diagnostic suite (`python -m src.diagnose_submission`) or pytest.
   - Trace root cause to exact pipeline step (features, imputation, modeling, stacking, final_submission).

2. **Reproduction (Failing Test Creation)**
   - Add a unit test or integration test capturing the edge case / failure in `tests/`.
   - Run `pytest tests/` to confirm failure.

3. **Surgical Patch Application**
   - Apply the targeted fix directly to the affected module.
   - Maintain all target leakage guardrails (e.g. fold-isolation, WCG index-exclusion).

4. **Verification & Regression Checks**
   - Run `pytest tests/ -v` to ensure all tests pass.
   - Run `python -m src.diagnose_submission` to confirm dimensional integrity and schema compliance.
