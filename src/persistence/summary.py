"""Enhanced summaries for experiment results.

Provides comprehensive analysis data structures for:
- Gate training comparisons (alone vs with others)
- Faithfulness-accuracy correlation analysis
- Subcircuit mapping across trials
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.circuit.enumeration import parse_subcircuit_idx


@dataclass
class GateContextMetrics:
    """Metrics for a gate in a specific training context."""

    context: str  # e.g., "XOR" or "XOR,OR"
    n_gates: int  # Number of gates in this context
    n_trials: int  # Number of trials with this context
    avg_gate_acc: float
    avg_subcircuit_acc: float
    avg_faithfulness: float
    # Faithfulness breakdown
    avg_observational: float | None = None
    avg_interventional: float | None = None
    avg_counterfactual: float | None = None


@dataclass
class GateComparison:
    """Compare gate performance across training contexts."""

    gate: str
    contexts: list[GateContextMetrics]
    best_context: str
    delta_vs_alone: float  # Improvement when training with others

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "gate": self.gate,
            "contexts": [
                {
                    "context": c.context,
                    "n_gates": c.n_gates,
                    "n_trials": c.n_trials,
                    "avg_gate_acc": round(c.avg_gate_acc, 4) if c.avg_gate_acc else None,
                    "avg_subcircuit_acc": round(c.avg_subcircuit_acc, 4)
                    if c.avg_subcircuit_acc
                    else None,
                    "avg_faithfulness": round(c.avg_faithfulness, 4)
                    if c.avg_faithfulness
                    else None,
                    "avg_observational": round(c.avg_observational, 4)
                    if c.avg_observational
                    else None,
                    "avg_interventional": round(c.avg_interventional, 4)
                    if c.avg_interventional
                    else None,
                    "avg_counterfactual": round(c.avg_counterfactual, 4)
                    if c.avg_counterfactual
                    else None,
                }
                for c in self.contexts
            ],
            "best_context": self.best_context,
            "delta_vs_alone": round(self.delta_vs_alone, 4),
        }


@dataclass
class FaithfulnessEntry:
    """A single faithfulness-accuracy entry for analysis."""

    trial_id: str
    gate: str
    subcircuit_idx: int
    accuracy: float
    faithfulness: float


@dataclass
class FaithfulnessAnalysis:
    """Analyze faithfulness-accuracy correlation."""

    correlation: float  # Pearson r
    high_faith_low_acc: list[FaithfulnessEntry]  # Potential structural issues
    low_faith_high_acc: list[FaithfulnessEntry]  # Potential shortcuts

    # Thresholds for categorization
    HIGH_FAITH_THRESHOLD = 0.9
    LOW_ACC_THRESHOLD = 0.85
    LOW_FAITH_THRESHOLD = 0.7
    HIGH_ACC_THRESHOLD = 0.95

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "_description": (
                "Faithfulness-accuracy correlation analysis. "
                "Faithfulness measures how well a subcircuit's behavior matches the full model. "
                "high_faith_low_acc: subcircuits that are faithful but inaccurate (structural issues). "
                "low_faith_high_acc: subcircuits that are accurate but unfaithful (shortcuts)."
            ),
            "_field_definitions": {
                "correlation": "Pearson correlation coefficient between accuracy and faithfulness (-1 to 1)",
                "accuracy": "Proportion of test samples where subcircuit output matches ground truth (0-1)",
                "faithfulness": "Overall faithfulness score combining observational, interventional, and counterfactual tests (0-1)",
            },
            "_thresholds": {
                "high_faithfulness": f">= {self.HIGH_FAITH_THRESHOLD}",
                "low_accuracy": f"< {self.LOW_ACC_THRESHOLD}",
                "low_faithfulness": f"< {self.LOW_FAITH_THRESHOLD}",
                "high_accuracy": f">= {self.HIGH_ACC_THRESHOLD}",
            },
            "correlation": round(self.correlation, 4),
            "high_faith_low_acc": [
                {
                    "trial_id": e.trial_id,
                    "gate": e.gate,
                    "subcircuit_idx": e.subcircuit_idx,
                    "accuracy": round(e.accuracy, 4),
                    "faithfulness": round(e.faithfulness, 4),
                }
                for e in self.high_faith_low_acc
            ],
            "low_faith_high_acc": [
                {
                    "trial_id": e.trial_id,
                    "gate": e.gate,
                    "subcircuit_idx": e.subcircuit_idx,
                    "accuracy": round(e.accuracy, 4),
                    "faithfulness": round(e.faithfulness, 4),
                }
                for e in self.low_faith_high_acc
            ],
        }


@dataclass
class SubcircuitMapping:
    """Track which trials found which subcircuits."""

    gate: str
    subcircuit_idx: int
    found_by: list[str]  # trial_ids
    best_accuracy: float
    best_trial: str

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "gate": self.gate,
            "subcircuit_idx": self.subcircuit_idx,
            "found_by": self.found_by,
            "n_trials": len(self.found_by),
            "best_accuracy": round(self.best_accuracy, 4),
            "best_trial": self.best_trial,
        }


@dataclass
class RunSummary:
    """Complete run summary - single source of truth."""

    experiment_id: str
    n_trials: int
    gates: list[str]

    # Per-gate comparisons
    gate_comparisons: dict[str, GateComparison] = field(default_factory=dict)

    # Faithfulness analysis
    faithfulness_analysis: FaithfulnessAnalysis | None = None

    # Subcircuit mapping
    subcircuit_mapping: dict[str, list[SubcircuitMapping]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "_description": "Enhanced run summary with gate comparisons and faithfulness analysis",
            "experiment_id": self.experiment_id,
            "n_trials": self.n_trials,
            "gates": self.gates,
            "gate_comparisons": {
                g: c.to_dict() for g, c in self.gate_comparisons.items()
            },
            "faithfulness_analysis": self.faithfulness_analysis.to_dict()
            if self.faithfulness_analysis
            else None,
            "subcircuit_mapping": {
                g: [m.to_dict() for m in mappings]
                for g, mappings in self.subcircuit_mapping.items()
            },
        }

    def to_text(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "EXPERIMENT SUMMARY",
            "=" * 70,
            "",
            "WHAT WE DID:",
            "  We trained a neural network (a program that learns patterns from examples)",
            "  to follow simple logical rules, like:",
            "    \"Let someone in if they have a ticket OR a VIP pass\"",
            "",
            "  A neural network has many small 'neurons' connected together.",
            "  We turned off neurons one-by-one to find the SMALLEST group",
            "  that still gives correct answers. We call this the 'subcircuit'.",
            "  (Smaller is easier to understand and verify.)",
            "",
            "THE KEY QUESTION:",
            "  Did this small part learn the REAL rule, or did it find a shortcut?",
            "",
            "  Imagine two students who both scored 100% on a math test:",
            "    Student A: Learned to actually solve the problems",
            "    Student B: Memorized the specific answers on this test",
            "",
            "  Both got 100% - but Student B will FAIL on a different test!",
            "  We want to catch 'memorizers' before they fail on new questions.",
            "",
            "HOW WE CHECK:",
            "  We spy on WHAT the computer looks at to make each decision,",
            "  not just whether it got the right answer.",
            "",
            "  Example of a shortcut:",
            "    Full computer: 'Do they have a ticket? OR a VIP pass? Let them in.'",
            "    Small part:    'Are they wearing a red shirt? Let them in.'",
            "    Both said YES for the same people - but for different reasons!",
            "    The shortcut works TODAY but might fail TOMORROW.",
            "",
            f"Rules tested: {', '.join(self.gates)}",
            f"Tests run:    {self.n_trials}",
            "",
            "READING THE RESULTS:",
            "  full   = % correct answers by the FULL network (all neurons)",
            "  small  = % correct answers by the SMALLEST working subcircuit",
            "  match  = How similar are their internal decision processes?",
            "           We run multiple tests that probe internal behavior,",
            "           not just final answers. (70%+ is considered good.)",
            "",
            "  Example: full:100% small:100% match:60%",
            "    → Both got all answers right",
            "    → But 40% of the time, their internal processing differed",
            "    → The subcircuit might be using a shortcut that fails later",
            "",
        ]

        # Gate training comparisons
        if self.gate_comparisons:
            lines.extend(
                [
                    "-" * 70,
                    "RESULTS",
                    "-" * 70,
                    "",
                    "We test each rule in different training setups:",
                    "  - Trained ALONE: Network only learned this one rule",
                    "  - Trained WITH others: Network learned multiple rules together",
                    "",
                ]
            )

            for gate, comp in self.gate_comparisons.items():
                for ctx in comp.contexts:
                    gate_acc = ctx.avg_gate_acc if ctx.avg_gate_acc else 0
                    sc_acc = ctx.avg_subcircuit_acc if ctx.avg_subcircuit_acc else 0
                    faith = ctx.avg_faithfulness if ctx.avg_faithfulness else 0

                    # Format training context description
                    other_gates = [g for g in ctx.context.split(",") if g.strip() != gate]
                    if other_gates:
                        context_desc = f"trained with {', '.join(other_gates)}"
                    else:
                        context_desc = "trained alone"

                    # Format with percentages for clarity
                    gate_pct = f"{gate_acc*100:.0f}%"
                    sc_pct = f"{sc_acc*100:.0f}%"
                    faith_pct = f"{faith*100:.0f}%"

                    lines.append(f"{gate}:")
                    lines.append(
                        f"  {context_desc:20} → full:{gate_pct:>4} | small:{sc_pct:>4} | match:{faith_pct:>4}"
                    )

        # Best circuits found
        if self.subcircuit_mapping:
            lines.extend(
                [
                    "",
                    "-" * 70,
                    "BEST CIRCUITS FOUND",
                    "-" * 70,
                    "",
                    "Each 'circuit' is a unique combination of which neurons are kept ON.",
                    "The # is just an identifier - lower isn't better or worse.",
                    "",
                ]
            )
            for gate, mappings in self.subcircuit_mapping.items():
                if mappings:
                    best = mappings[0]
                    lines.append(f"  {gate}: Circuit #{best.subcircuit_idx} (accuracy: {best.best_accuracy*100:.0f}%)")

        # Bottom line analysis
        lines.extend(
            [
                "",
                "-" * 70,
                "BOTTOM LINE",
                "-" * 70,
            ]
        )

        # Collect anomalies
        shortcuts = []  # High accuracy, low faithfulness

        for gate, comp in self.gate_comparisons.items():
            for ctx in comp.contexts:
                acc = ctx.avg_subcircuit_acc or 0
                faith = ctx.avg_faithfulness or 0

                # Shortcut: accurate but different method
                if acc >= 0.95 and faith < 0.70:
                    shortcuts.append((gate, acc, faith))

        lines.append("")
        lines.append("THINGS TO WATCH:")

        if shortcuts:
            lines.append(f"\n  POSSIBLE SHORTCUTS ({len(shortcuts)} found):")
            lines.append("    The small piece gets right answers but uses a DIFFERENT method.")
            lines.append("    Like a student who scored 100% by memorizing answers instead")
            lines.append("    of learning the material. May fail on new questions later.")
            for gate, acc, faith in shortcuts[:3]:
                lines.append(f"    - {gate}: {acc*100:.0f}% correct, but only {faith*100:.0f}% same method")
            lines.append("")

        if not shortcuts:
            lines.append("")
            lines.append("  No issues found. The small pieces appear to use similar methods")
            lines.append("  as the full network.")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


def _safe_mean(values: list[float | None]) -> float | None:
    """Compute mean of values, ignoring None and NaN."""
    valid = [v for v in values if v is not None and v == v]  # v == v filters NaN
    return sum(valid) / len(valid) if valid else None


def _pearson_correlation(pairs: list[tuple[float, float]]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(pairs) < 2:
        return 0.0

    n = len(pairs)
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]

    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / n
    std_x = (sum((x - mean_x) ** 2 for x in x_vals) / n) ** 0.5
    std_y = (sum((y - mean_y) ** 2 for y in y_vals) / n) ** 0.5

    if std_x == 0 or std_y == 0:
        return 0.0

    return cov / (std_x * std_y)


def build_gate_comparison(result, trial_org: dict) -> dict[str, GateComparison]:
    """Compare each gate's performance when trained alone vs with others.

    Uses trial_org.by_axis.gates to find:
    - "XOR" -> trials training XOR alone
    - "XOR,OR" -> trials training XOR+OR together
    """
    comparisons = {}

    # Get gates-to-trials mapping
    gates_to_trials = trial_org.get("by_axis", {}).get("gates", {})
    if not gates_to_trials:
        return comparisons

    # Find all base gates
    base_gates = set()
    for combo in gates_to_trials.keys():
        for g in combo.split(","):
            base_gates.add(g.strip())

    for gate in base_gates:
        contexts = []

        for combo, trial_ids in gates_to_trials.items():
            # Check if this gate is in this combo
            combo_gates = [g.strip() for g in combo.split(",")]
            if gate not in combo_gates:
                continue

            # Compute avg metrics for trials with this context
            gate_accs = []
            sc_accs = []
            faiths = []
            observationals = []
            interventionals = []
            counterfactuals = []

            for trial_id in trial_ids:
                trial = result.trials.get(trial_id)
                if not trial:
                    continue

                # Gate test accuracy
                gate_metrics = trial.metrics.per_gate_metrics.get(gate)
                if gate_metrics:
                    gate_accs.append(getattr(gate_metrics, "test_acc", None))

                    # Get model dimensions for parsing flat indices
                    width = trial.setup.model_params.width
                    depth = trial.setup.model_params.depth

                    # Build mapping from node_mask_idx to accuracy
                    sc_metrics = getattr(gate_metrics, "subcircuit_metrics", [])
                    idx_to_acc = {sm.idx: sm.accuracy for sm in sc_metrics}

                    # Get best subcircuit indices and faithfulness (they're aligned)
                    bests_indices = trial.metrics.per_gate_bests.get(gate, [])
                    faith_list = trial.metrics.per_gate_bests_faith.get(gate, [])

                    # Use first best subcircuit (if exists) - correctly paired data
                    if bests_indices and faith_list:
                        flat_idx = bests_indices[0]
                        node_mask_idx, _ = parse_subcircuit_idx(width, depth, flat_idx)

                        # Look up accuracy for this specific subcircuit
                        if node_mask_idx in idx_to_acc:
                            sc_accs.append(idx_to_acc[node_mask_idx])

                        # Get faithfulness for same subcircuit (overall + breakdown)
                        best_faith = faith_list[0]
                        if best_faith and best_faith.overall_faithfulness is not None:
                            faiths.append(best_faith.overall_faithfulness)
                            # Extract breakdown
                            if best_faith.observational:
                                observationals.append(best_faith.observational.overall_observational)
                            if best_faith.interventional:
                                interventionals.append(best_faith.interventional.overall_interventional)
                            if best_faith.counterfactual:
                                counterfactuals.append(best_faith.counterfactual.overall_counterfactual)

            if gate_accs:
                ctx = GateContextMetrics(
                    context=combo,
                    n_gates=len(combo_gates),
                    n_trials=len(trial_ids),
                    avg_gate_acc=_safe_mean(gate_accs),
                    avg_subcircuit_acc=_safe_mean(sc_accs),
                    avg_faithfulness=_safe_mean(faiths),
                    avg_observational=_safe_mean(observationals),
                    avg_interventional=_safe_mean(interventionals),
                    avg_counterfactual=_safe_mean(counterfactuals),
                )
                contexts.append(ctx)

        if not contexts:
            continue

        # Sort by accuracy, find best
        contexts.sort(
            key=lambda c: c.avg_gate_acc if c.avg_gate_acc else 0, reverse=True
        )
        alone = next((c for c in contexts if c.n_gates == 1), None)
        best = contexts[0] if contexts else None

        # Compute delta vs alone
        delta = 0.0
        if alone and best and alone.avg_gate_acc and best.avg_gate_acc:
            delta = best.avg_gate_acc - alone.avg_gate_acc

        comparisons[gate] = GateComparison(
            gate=gate,
            contexts=contexts,
            best_context=best.context if best else "",
            delta_vs_alone=delta,
        )

    return comparisons


def build_faithfulness_analysis(result) -> FaithfulnessAnalysis:
    """Identify interesting faithfulness-accuracy patterns.

    IMPORTANT: This correctly pairs accuracy and faithfulness for the SAME subcircuit
    by using per_gate_bests (flat indices) and per_gate_bests_faith (faithfulness)
    which are aligned, then looking up accuracy via node_mask_idx.
    """
    pairs = []  # (accuracy, faithfulness)
    high_faith_low_acc = []
    low_faith_high_acc = []

    for trial_id, trial in result.trials.items():
        # Get model dimensions for parsing flat indices
        width = trial.setup.model_params.width
        depth = trial.setup.model_params.depth

        for gate, gate_metrics in trial.metrics.per_gate_metrics.items():
            # Build mapping from node_mask_idx to accuracy
            sc_metrics_list = getattr(gate_metrics, "subcircuit_metrics", [])
            idx_to_acc = {sm.idx: sm.accuracy for sm in sc_metrics_list}

            # Get aligned lists of best subcircuit indices and their faithfulness
            bests_indices = trial.metrics.per_gate_bests.get(gate, [])
            faith_list = trial.metrics.per_gate_bests_faith.get(gate, [])

            # Iterate over paired data (both lists are aligned)
            for flat_idx, faith_data in zip(bests_indices[:5], faith_list[:5]):
                if faith_data is None:
                    continue

                faith = faith_data.overall_faithfulness
                if faith is None:
                    continue

                # Parse flat index to get node_mask_idx, then look up accuracy
                node_mask_idx, _ = parse_subcircuit_idx(width, depth, flat_idx)
                acc = idx_to_acc.get(node_mask_idx, 0) or 0

                pairs.append((acc, faith))

                entry = FaithfulnessEntry(
                    trial_id=trial_id,
                    gate=gate,
                    subcircuit_idx=node_mask_idx,
                    accuracy=acc,
                    faithfulness=faith,
                )

                # Classify using FaithfulnessAnalysis thresholds
                if faith >= FaithfulnessAnalysis.HIGH_FAITH_THRESHOLD and acc < FaithfulnessAnalysis.LOW_ACC_THRESHOLD:
                    high_faith_low_acc.append(entry)
                elif faith < FaithfulnessAnalysis.LOW_FAITH_THRESHOLD and acc >= FaithfulnessAnalysis.HIGH_ACC_THRESHOLD:
                    low_faith_high_acc.append(entry)

    # Compute Pearson correlation
    corr = _pearson_correlation(pairs) if len(pairs) >= 2 else 0

    return FaithfulnessAnalysis(
        correlation=corr,
        high_faith_low_acc=high_faith_low_acc,
        low_faith_high_acc=low_faith_high_acc,
    )


def build_subcircuit_mapping(result) -> dict[str, list[SubcircuitMapping]]:
    """Track which trials found which subcircuits.

    Uses the TOP 5 subcircuits by accuracy (not first 5 by enumeration order).
    """
    # gate -> subcircuit_idx -> {trial_ids, best_acc, best_trial}
    mapping_data: dict[str, dict[int, dict]] = {}

    for trial_id, trial in result.trials.items():
        for gate, gate_metrics in trial.metrics.per_gate_metrics.items():
            if gate not in mapping_data:
                mapping_data[gate] = {}

            sc_metrics_list = getattr(gate_metrics, "subcircuit_metrics", [])

            # Sort by accuracy descending, then take top 5
            sorted_sc = sorted(
                sc_metrics_list, key=lambda sm: sm.accuracy or 0, reverse=True
            )

            for sm in sorted_sc[:5]:  # Top 5 by accuracy
                acc = sm.accuracy or 0
                idx = sm.idx

                if idx not in mapping_data[gate]:
                    mapping_data[gate][idx] = {
                        "trial_ids": [],
                        "best_acc": acc,
                        "best_trial": trial_id,
                    }

                mapping_data[gate][idx]["trial_ids"].append(trial_id)
                if acc > mapping_data[gate][idx]["best_acc"]:
                    mapping_data[gate][idx]["best_acc"] = acc
                    mapping_data[gate][idx]["best_trial"] = trial_id

    # Convert to SubcircuitMapping objects
    result_mapping = {}
    for gate, idx_data in mapping_data.items():
        mappings = []
        for idx, data in idx_data.items():
            mappings.append(
                SubcircuitMapping(
                    gate=gate,
                    subcircuit_idx=idx,
                    found_by=data["trial_ids"],
                    best_accuracy=data["best_acc"],
                    best_trial=data["best_trial"],
                )
            )
        # Sort by best accuracy descending
        mappings.sort(key=lambda m: m.best_accuracy, reverse=True)
        result_mapping[gate] = mappings

    return result_mapping


def build_run_summary(result, trial_org: dict) -> RunSummary:
    """Build complete run summary from experiment result."""
    # Get gate names from first trial
    first_trial = next(iter(result.trials.values()), None)
    gates = first_trial.setup.model_params.logic_gates if first_trial else []

    summary = RunSummary(
        experiment_id=result.experiment_id,
        n_trials=len(result.trials),
        gates=gates,
    )

    # Build gate comparisons
    summary.gate_comparisons = build_gate_comparison(result, trial_org)

    # Build faithfulness analysis
    summary.faithfulness_analysis = build_faithfulness_analysis(result)

    # Build subcircuit mapping
    summary.subcircuit_mapping = build_subcircuit_mapping(result)

    return summary
