---
name: feature-engineering
description: Non-leaking mathematical causal feature generation using Genetic Programming (GP) operators within fold-local splits.
version: 1.0.0
triggers:
  - feature_synthesis
  - gp_operator_generation
  - tabular_feature_expansion
---

# Feature Engineering Skill: Genetic Programming & Causal Operators

## Overview
This skill specifies the protocol for synthesizing high-order, non-linear mathematical and causal interaction features using Genetic Programming (GP) symbolic operators, strictly confined within fold-local cross-validation boundaries to prevent data and target leakage.

## Core Principles & Leakage Guardrails

### 1. Fold-Local Fit Isolation
- **Strict CV Partitioning**: All GP symbolic regressors/classifiers, statistics-based transformations, and causal interaction trees must be fitted *only* on the training indices of the active CV fold.
- **Out-of-Fold Transform**: Test or validation fold sets must only receive transformations via `.transform()` or deterministic algebraic expression evaluation.

### 2. Genetic Programming (GP) Operator Set
Safe symbolic primitives for algebraic feature synthesis:
- **Arithmetic Operators**: `add`, `sub`, `mul`, `div` (protected division: `x / (y + eps)`).
- **Non-Linear Transformations**: `log1p(abs(x))`, `sqrt(abs(x))`, `tanh`, `sigmoid`.
- **Causal Interaction Formulations**:
  - Demographic & Socio-economic Ratios: `AdjFare = Fare / GroupSize`, `AgeRatio = Age / AvgFamilyAge`.
  - Resource & Survival Density: `FamilyRiskIndex = SibSp + Parch + 1`.

### 3. Symbolic Complexity & Bloat Control
- **Parsimony Pressure**: Enforce strict penalty coefficients against tree depth (max tree depth <= 4) to avoid memorizing sample noise.
- **Fitness Criteria**: Evaluate candidate GP features using OOF correlation / Mutual Information with target residual or fold-local log-loss improvement.

## Execution Workflow

1. **Local Fold Initialization**:
   - Extract fold-local feature matrix `X_train_fold` and target `y_train_fold`.

2. **Symbolic Feature Evolution**:
   - Fit GP symbolic feature generator using primitive operator set.
   - Extract top K symbolic mathematical expressions exhibiting highest out-of-bag validation gain.

3. **Deterministic Integration**:
   - Export synthesized features as vector expressions to `src/features.py` / `src/modeling.py`.
   - Validate that engineered feature vectors contain no NaN/Inf artifacts across train and test partitions.
