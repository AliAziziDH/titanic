# SPEC-01: Genetic Feature Synthesis (Phase 3)

```yaml
schema_version: "2.0.0"
spec_type: "Spec-Driven Development (SDD)"
experiment_id: "EXP-01"
title: "Genetic Feature Synthesis (Phase 3)"
status: "DRAFT"
created_at: "2026-08-22"
```

## 1. Objective
- Synthesize new causal and non-leaking mathematical features in Phase 3 using Symbolic Genetic Programming (GP) operators.
- Improve the baseline local Cross-Validation (CV) score of the stacking meta-classifier without introducing target leakage or overfitting.

---

## 2. Acceptance Criteria

- **Strict Leakage Guardrails**:
  - All genetic feature calculations and operator fittings must occur strictly within fold-local training splits.
  - Estimators or genetic synthesis operators must never fit on validation or test splits.
  - Feature transformations on validation/test sets must strictly use parameters learned from the fold-local training split.

- **Dimension & Schema Integrity**:
  - Training dataset shape must strictly maintain **891 rows**.
  - Test dataset shape must strictly maintain **418 rows** (Passenger IDs: 892 to 1309).
  - Target variable format: Binary predictions (0 or 1) in the `Survived` column.

- **Out-of-Fold (OOF) CV Improvement**:
  - The local Stratified 5-Fold CV score must demonstrate statistically sound improvement over the baseline stacking meta-classifier.

---

## 3. Verification Protocol

Follow the sequential execution and evaluation gates strictly:

1. **Feature Synthesis & Pipeline Run**:
   ```bash
   python -m src.features
   python -m src.imputation
   python -m src.modeling
   python -m src.stacking
   python -m src.final_submission
   ```

2. **Automated Regression Testing**:
   ```bash
   pytest tests/ -v
   ```
   *Requirement:* All unit and integration tests must pass with zero failures.

3. **Submission Diagnostics & Leakage Audit**:
   ```bash
   python -m src.diagnose_submission
   ```
   *Requirement:* Confirm full schema compliance, correct row counts, and zero target leakage anomalies.

---

## 4. Agent Brakes (Circuit Breaker)

- **Iteration Limit**:
  - Maximum of **3 search iterations** for symbolic genetic synthesis runs.

- **Halt Conditions**:
  - If local CV score degrades compared to the baseline.
  - If `pytest tests/ -v` fails.
  - If `python -m src.diagnose_submission` detects dimension or leakage violations.

- **Action on Trigger**:
  - Agent must immediately halt further execution.
  - Output trajectory log detailing the failure or degradation.
  - Prompt for human feedback before any retry or self-repair.
