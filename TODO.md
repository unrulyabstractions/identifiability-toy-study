# Consolidated TODO List - Iteration 10 (Final)

## Iteration History
- **Iteration 1:** Initial consolidation from 5 agent analyses
- **Iteration 2:** Removed auto-import (6.3) - too risky, manual imports are clearer
- **Iteration 3:** Reordered: do trial refactoring BEFORE causal (trial depends on causal)
- **Iteration 4:** Combined 2.4 with 3.1 (rename happens during split)
- **Iteration 5:** Removed 7.1 (split schemas.py) - too invasive, keep monolithic
- **Iteration 6:** Removed 3.3 (compose FaithfulnessMetrics) - breaking change, skip
- **Iteration 7:** Simplified Phase 2 - only do SampleType enum (safe)
- **Iteration 8:** Removed 4.4 (restructure trial/) - too invasive, just do 4.1-4.3
- **Iteration 9:** Removed 5.2, 5.3 (spd_internal split) - 1800 line file, too risky
- **Iteration 10:** Final review - focus on SAFE, HIGH-VALUE changes only

## Final Scope (After 10 Iterations)
Only implement changes that are:
1. Low risk of breaking existing functionality
2. Easy to verify with existing tests
3. Improve code quality without major restructuring

## Overview
This document consolidates all TODO(claude) comments from the codebase into an actionable plan.
Total TODOs identified: 50+ across multiple files.

**Scope:** Only our code (src/), NOT the external SPD submodule (src/spd/).

---

## PRIORITY 1: Quick Wins (Low complexity, high impact)

### 1.1 Delete benchmark_parallel.py
- **File:** `scripts/benchmark_parallel.py`
- **Action:** Delete file, remove scripts/ folder if empty
- **Complexity:** Trivial

### 1.2 Move scipy imports to top of file
- **File:** `src/spd_internal/analysis.py:367`
- **Action:** Move imports to top with SCIPY_AVAILABLE flag
- **Complexity:** Low

### 1.3 Improve function documentation
- **File:** `src/spd_internal/analysis.py:419`
- **Action:** Expand docstring for `map_clusters_to_functions`
- **Complexity:** Low

### 1.4 Create SampleType Enum
- **File:** `src/common/schemas.py:461`
- **Action:** Replace string literal with Enum
- **Complexity:** Low

---

## PRIORITY 2: Schema Refactoring (Foundation for other changes)

### 2.1 Create SimilarityMetrics dataclass
- **Files:** `src/common/schemas.py` (lines 224, 243, 449, 454)
- **Action:** Create reusable SimilarityMetrics class, refactor all usages
- **Dependencies:** Used in InterventionSample, PatchStatistics, RobustnessSample

### 2.2 Create InterventionResult dataclass
- **File:** `src/common/schemas.py:536`
- **Action:** Refactor InterventionalMetrics to use 4 InterventionResult instances
- **Dependencies:** None

### 2.3 Clarify epsilon documentation
- **File:** `src/common/schemas.py:589`
- **Action:** Document relationship to IdentifiabilityConstraints.epsilon
- **Complexity:** Low

### 2.4 Rename RobustnessMetrics to ObservationalMetrics
- **Files:** `src/common/schemas.py`, `src/causal/observational.py:116`
- **Action:** Rename class, update all references
- **Dependencies:** Many files

---

## PRIORITY 3: Causal Module Refactoring

### 3.1 Split calculate_robustness_metrics
- **File:** `src/causal/observational.py:109-110`
- **Action:** Create calculate_noise_robustness_metrics + calculate_ood_extension_metrics
- **Rename:** Main function to calculate_observational_metrics

### 3.2 Split calculate_faithfulness_metrics
- **File:** `src/causal/metrics.py:79-80`
- **Action:** Create calculate_interventional_metrics + calculate_counterfactual_metrics
- **Dependencies:** Depends on 2.1, 2.2

### 3.3 Compose FaithfulnessMetrics
- **File:** `src/common/schemas.py:359`
- **Action:** FaithfulnessMetrics = ObservationalMetrics + InterventionalMetrics + CounterfactualMetrics
- **Dependencies:** Depends on 3.1, 3.2

---

## PRIORITY 4: Trial Module Refactoring

### 4.1 Move _get_eval_device utility
- **File:** `src/trial.py:35`
- **Action:** Move to src/common/parallelization.py
- **Complexity:** Low

### 4.2 Remove wrapper functions
- **File:** `src/trial.py:182`
- **Action:** Add @profile_fn to underlying functions, remove wrappers
- **Dependencies:** src/common/batched_eval.py, src/common/helpers.py

### 4.3 Create analyze_gate function
- **File:** `src/trial.py:417`
- **Action:** Extract 120-line loop body into analyze_gate()
- **Complexity:** Medium

### 4.4 Restructure trial.py into trial/ module
- **File:** `src/trial.py:317`
- **Action:** Only run_trial in main file, move helpers to submodules
- **Dependencies:** Depends on 4.1, 4.2, 4.3

---

## PRIORITY 5: SPD Internal Refactoring

### 5.1 Move parameter_decomposition.py
- **File:** `src/parameter_decomposition.py:21`
- **Action:** Move to src/spd_internal/decomposition.py
- **Complexity:** Low

### 5.2 Separate spd_internal/analysis.py
- **File:** `src/spd_internal/analysis.py:47`
- **Action:** Split into schemas.py, validation.py, importance.py, clustering.py, robustness.py
- **Complexity:** Medium

### 5.3 Move visualization code to viz/
- **File:** `src/spd_internal/analysis.py:616`
- **Action:** Move 8 visualization functions to src/viz/spd_viz.py
- **Dependencies:** Depends on 5.2

### 5.4 Separate persistence from subcircuits.py
- **File:** `src/spd_internal/subcircuits.py:175`
- **Action:** Move save_spd_estimate/load_spd_estimate to persistence.py
- **Complexity:** Low

---

## PRIORITY 6: Documentation & Cleanup

### 6.1 Update main README.md
- **File:** `README.md`
- **Action:** Focus on how to run, parameters, metrics
- **Complexity:** Medium

### 6.2 Update persistence README.md
- **File:** `src/persistence/README.md`
- **Action:** Review and update
- **Complexity:** Low

### 6.3 Auto-import in __init__.py files
- **File:** `src/persistence/__init__.py:9`
- **Action:** Implement auto-import utility for all modules
- **Complexity:** Medium-High (risk of breaking imports)
- **SKIP:** Too risky, manual imports are fine

---

## PRIORITY 7: Schema File Organization (Last)

### 7.1 Split schemas.py into separate files
- **File:** `src/common/schemas.py:50`
- **Action:** Create src/common/schemas/ with separate files per category
- **Dependencies:** Do AFTER all other schema changes
- **Complexity:** High

### 7.2 Move generate_spd_sweep_configs
- **File:** `src/common/schemas.py:126`
- **Action:** Move to appropriate utility file
- **Complexity:** Low

### 7.3 Remove per_gate_bests_robust
- **File:** `src/common/schemas.py:626`
- **Action:** Only keep per_gate_bests_faith (includes observational)
- **Dependencies:** Depends on 3.3

---

## NOT IN SCOPE (External SPD submodule)
The following TODOs are in the external SPD submodule and should not be modified:
- src/spd/spd/*.py (all files)
- These are maintained by goodfire-ai

---

## FINAL Implementation Order (After 10 Iterations)

**IMPLEMENT NOW (Safe, high-value):**
- 1.1: Delete benchmark_parallel.py
- 1.2: Move scipy imports to top
- 1.3: Improve function documentation
- 1.4: Create SampleType Enum
- 4.1: Move _get_eval_device to parallelization.py
- 5.1: Move parameter_decomposition.py to spd_internal/
- 5.4: Separate persistence from subcircuits.py
- 6.1: Update README.md

**SKIP (Too risky/invasive):**
- 2.1-2.4: Schema refactoring (breaking changes)
- 3.1-3.3: Causal refactoring (breaking changes)
- 4.2-4.4: Trial restructuring (too invasive)
- 5.2-5.3: spd_internal split (1800 lines, too risky)
- 6.3: Auto-import (risky)
- 7.1-7.3: Schema file organization (too invasive)

**REMOVE TODO COMMENTS:** After implementation, remove all TODO(claude) comments from code
