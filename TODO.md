# Faithfulness Visualization Refactoring - COMPLETED

## Summary of Changes

### Final Folder Structure
```
faithfulness/
├── summary.json                      # Overall summary metrics
├── observational/
│   ├── result.json                   # Detailed observational metrics
│   └── circuit_viz/
│       ├── noise_perturbations/
│       │   ├── 0_0.png, 0_1.png, 1_0.png, 1_1.png
│       │   └── summary.png
│       └── out_distribution_transformations/
│           ├── multiply/
│           │   ├── 0_1_positive.png, 0_1_negative.png, ...
│           │   ├── summary_positive.png
│           │   └── summary_negative.png
│           ├── add/
│           │   ├── 0_0.png, 0_1.png, 1_0.png, 1_1.png
│           │   └── summary.png
│           ├── subtract/
│           │   ├── 0_0.png, 0_1.png, 1_0.png, 1_1.png
│           │   └── summary.png
│           └── bimodal/
│               ├── 0_0.png, 0_1.png, ... (order-preserving)
│               ├── 0_0_inv.png, 0_1_inv.png, ... (inverted)
│               ├── summary.png
│               └── summary_inv.png
├── counterfactual/
│   ├── result.json                   # Detailed counterfactual metrics
│   ├── sufficiency/
│   ├── completeness/
│   ├── necessity/
│   ├── independence/
│   ├── counterfact_summary.png
│   ├── denoising_per_input.png
│   └── noising_per_input.png
└── interventional/
    ├── result.json                   # Detailed interventional metrics
    ├── in_circuit/
    │   ├── in_distribution/
    │   ├── out_of_distribution/
    │   └── *_stats.png
    ├── out_circuit/
    │   ├── in_distribution/
    │   ├── out_of_distribution/
    │   └── *_stats.png
    ├── in_distribution_summary.png
    └── out_distribution_summary.png
```

### JSON Schema Types Added
- `ObservationalMetrics`: Per-type OOD agreement metrics
- `InterventionalMetrics`: Per-circuit-type and per-distribution metrics
- `CounterfactualMetrics`: Denoising/noising counts and means
- `FaithfulnessSummary`: Overall scores for summary.json

### Visual Fixes Applied
- **All titles**: Fixed with `tight_layout(rect=[0, 0, 1, 0.94])` before `suptitle(y=0.99)`
- **Circuit viz titles**: "(0, 1) → 1" as main title, transformation type as subtitle
- **counterfact_summary.png**: Simple title, column labels high, no colorbar
- **denoising_per_input.png**: Title "Denoising", no legend
- **noising_per_input.png**: Title "Noising", no legend
- **interventional titles**: Show agreement % over ALL samples

### Files Modified
- `identifiability_toy_study/visualization.py` - Major refactoring
- `identifiability_toy_study/causal_analysis.py` - Added new OOD sample types
- `identifiability_toy_study/common/schemas.py` - Added sample_type field + metric schemas
- `identifiability_toy_study/profiler.py` - Added `@logged()` decorator
- `tests/test_batch_sequential_equivalence.py` - Fixed for 4-value return
- `tests/test_batched_eval.py` - Fixed for 4-value return
- `tests/test_causal_analysis.py` - Updated for 4-tuple sample returns

### Key Changes
1. Reorganized circuit_viz/ with new folder structure
2. Separate summary files: multiply/{positive,negative}, bimodal/{summary,inv}
3. Added new OOD transformation types: add, subtract, bimodal
4. Fixed title positioning across all figures
5. Added result.json in observational/, interventional/, counterfactual/
6. Added summary.json in faithfulness/ with overall metrics
7. Bimodal transformation interprets outputs with threshold at 0 instead of 0.5
