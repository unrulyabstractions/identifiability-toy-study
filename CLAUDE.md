# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Identifiability Toy Study: A research framework for analyzing subcircuit faithfulness in neural networks trained on logic gates. Tests whether small subcircuits use the same computation as the full network or rely on shortcuts.

## Development Commands

```bash
uv run main.py                    # Run full experiment
uv run main.py --test 0           # Fast test mode (XOR only)
uv run main.py --test 0 --viz 2   # With visualization
uv run main.py --viz-only --run <name>  # Re-run viz on existing run
```

**Visualization levels:**
- `--viz 0` (default): No PNGs, only JSON/TXT summaries
- `--viz 1`: Summary plots only
- `--viz 2`: + Top subcircuit per gate
- `--viz 3`: + Top 5 subcircuits

## Architecture

```
src/
├── analysis/           # Faithfulness analysis (observational, interventional, counterfactual)
├── circuit/            # Circuit enumeration and visualization
├── model/              # MLP model with intervention support
├── persistence/        # Save/load results
├── trial/              # Per-trial execution
├── viz/                # Visualization generation
├── experiment.py       # Trial settings generation
├── experiment_config.py # Config dataclasses
└── pipeline.py         # Main execution pipeline
```

**Key data flow:**
1. `main.py` → `pipeline.py` (run_iteratively or run_monolith)
2. `trial/trial_run.py` → trains model, enumerates subcircuits
3. `analysis/faithfulness.py` → computes observational/interventional/counterfactual metrics
4. `viz/experiment_viz.py` → generates visualizations
5. `persistence/save.py` → saves results to disk

## Key Concepts

- **Node pattern**: Which hidden neurons are active (bitmask)
- **Edge variant**: Which edges are active within a node pattern
- **Subcircuit**: Specific (node_pattern, edge_variant) combination
- **Faithfulness**: How well subcircuit matches full model behavior
  - Observational: Same input → same output?
  - Interventional: Patching activations → same effect?
  - Counterfactual: Denoising/noising → same response?

## Workflow Orchestration

1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
