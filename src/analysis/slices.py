"""Slice analysis for ranking subcircuits by different metrics.

Each Slice class computes rankings for a specific faithfulness category:
- ObservationalSlice: noise perturbations, OOD transformations
- InterventionalSlice: in-circuit/out-circuit, ID/OOD
- CounterfactualSlice: denoising, noising (sufficiency, necessity, etc.)

Usage:
    obs_slice = ObservationalSlice.from_trial(trial, gate_name, width, depth)
    obs_slice.save(output_dir)
"""

from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING
import json
import os

from src.circuit import parse_subcircuit_idx

if TYPE_CHECKING:
    from src.schemas import TrialResult, FaithfulnessMetrics


@dataclass
class SubcircuitStats:
    """Stats for a single subcircuit (node pattern + edge variant)."""
    subcircuit_idx: int
    node_pattern: int
    edge_variant_rank: int
    sparsity: float  # fraction of nodes pruned
    accuracy: float = 0.0
    score: float = 0.0  # the metric being ranked by


@dataclass
class NodePatternStats:
    """Aggregated stats for a node pattern across its edge variants."""
    node_pattern: int
    n_edge_variants: int
    sparsity: float
    mean_score: float
    max_score: float
    min_score: float
    best_edge_variant: int
    best_subcircuit_idx: int


@dataclass
class RankingResult:
    """A ranking of node patterns and subcircuits by a specific metric."""
    metric_name: str
    description: str
    node_patterns: list[NodePatternStats] = field(default_factory=list)
    subcircuits: list[SubcircuitStats] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "description": self.description,
            "node_patterns": [asdict(np) for np in self.node_patterns],
            "subcircuits": [asdict(sc) for sc in self.subcircuits],
        }


def _compute_total_edges(input_size: int, width: int, depth: int) -> int:
    """Compute total number of edges in full circuit."""
    # Layer 0->1: input_size * width
    # Layer i->i+1 (hidden): width * width for each of depth-1 transitions
    # Last layer: width * 1
    total = input_size * width  # input to first hidden
    total += (depth - 1) * width * width  # between hidden layers
    total += width * 1  # last hidden to output
    return total


def _get_subcircuit_edge_sparsity(
    subcircuit_idx: int,
    circuits: list = None,
    width: int = 4,
    depth: int = 2,
    input_size: int = 2,
    edge_variant_rank: int = 0,
    max_edge_variants: int = 10,
) -> float:
    """Get edge sparsity for a subcircuit.

    Edge sparsity = 1 - (active_edges / total_edges)

    Since edge variants prune edges from the base circuit,
    higher edge_variant_rank typically means more edges pruned.
    """
    total_edges = _compute_total_edges(input_size, width, depth)

    if circuits:
        node_idx, _ = parse_subcircuit_idx(width, depth, subcircuit_idx)
        if node_idx < len(circuits):
            circuit = circuits[node_idx]
            # Count active edges from edge_masks
            base_active_edges = 0
            for edge_mask in circuit.edge_masks:
                base_active_edges += int(edge_mask.sum())

            # Edge variants prune edges from base configuration
            # Estimate: rank 0 has all base edges, higher ranks have fewer
            # Scale the reduction based on rank
            if max_edge_variants > 1 and edge_variant_rank > 0:
                # Estimate edge reduction: each rank removes ~5-15% of edges
                reduction_factor = 1.0 - (edge_variant_rank / max_edge_variants) * 0.3
                active_edges = base_active_edges * reduction_factor
            else:
                active_edges = base_active_edges

            return 1 - active_edges / total_edges if total_edges > 0 else 0

    # Fallback: rough estimate based on node pattern
    return 0.5  # Default middle value


@dataclass
class ObservationalSlice:
    """Observational analysis slice - ranks by noise and OOD metrics."""

    gate_name: str
    rank_accuracy: RankingResult = None
    rank_noise: RankingResult = None
    rank_ood: RankingResult = None

    @classmethod
    def from_trial(
        cls,
        trial: "TrialResult",
        gate_name: str,
        width: int,
        depth: int,
        circuits: list = None,
        input_size: int = 2,
    ) -> "ObservationalSlice":
        """Build observational slice from trial data."""
        best_keys = trial.metrics.per_gate_bests.get(gate_name, [])
        bests_faith = trial.metrics.per_gate_bests_faith.get(gate_name, [])
        gate_metrics = trial.metrics.per_gate_metrics.get(gate_name)

        # Build accuracy lookup by node_mask_idx
        # Note: subcircuit_metrics.idx is the node_mask_idx (0-223), not flat subcircuit index
        acc_lookup = {}
        if gate_metrics:
            for sm in gate_metrics.subcircuit_metrics:
                acc_lookup[sm.idx] = sm.accuracy or 0

        # Count max edge variants per node pattern for sparsity estimation
        edge_variants_per_node = {}
        for key in best_keys:
            node_idx, _ = parse_subcircuit_idx(width, depth, key)
            edge_variants_per_node[node_idx] = edge_variants_per_node.get(node_idx, 0) + 1
        max_edge_variants = max(edge_variants_per_node.values()) if edge_variants_per_node else 10

        # Collect data for each subcircuit
        subcircuit_data = []
        for i, key in enumerate(best_keys):
            node_idx, edge_rank = parse_subcircuit_idx(width, depth, key)
            faith = bests_faith[i] if i < len(bests_faith) else None

            sparsity = _get_subcircuit_edge_sparsity(
                key, circuits, width, depth, input_size, edge_rank, max_edge_variants
            )
            accuracy = acc_lookup.get(node_idx, 0)  # Use node_idx, not flat key

            # Extract observational scores
            noise_score = 0.0
            ood_score = 0.0
            if faith and faith.observational:
                obs = faith.observational
                # Noise score from similarity
                if obs.noise:
                    # Use bit similarity as the noise score
                    noise_score = obs.noise.similarity.bit if obs.noise.similarity else 0.0

                # OOD score from multiplicative transformations only
                if obs.ood:
                    # Only use multiplicative OOD (positive and negative scaling)
                    ood_scores = []
                    if obs.ood.multiply_positive_n_samples > 0:
                        ood_scores.append(obs.ood.multiply_positive_agreement)
                    if obs.ood.multiply_negative_n_samples > 0:
                        ood_scores.append(obs.ood.multiply_negative_agreement)
                    ood_score = sum(ood_scores) / len(ood_scores) if ood_scores else 0.0

            subcircuit_data.append({
                "subcircuit_idx": key,
                "node_pattern": node_idx,
                "edge_variant_rank": edge_rank,
                "sparsity": sparsity,
                "accuracy": accuracy,
                "noise_score": noise_score,
                "ood_score": ood_score,
            })

        # Build rankings
        rank_accuracy = cls._build_ranking(
            subcircuit_data, "accuracy", "accuracy",
            "Ranking by subcircuit accuracy on test set"
        )
        rank_noise = cls._build_ranking(
            subcircuit_data, "noise_score", "noise_perturbations",
            "Ranking by agreement under noise perturbations"
        )
        rank_ood = cls._build_ranking(
            subcircuit_data, "ood_score", "out_distribution_transformations",
            "Ranking by agreement under multiplicative OOD (positive and negative scaling)"
        )

        return cls(
            gate_name=gate_name,
            rank_accuracy=rank_accuracy,
            rank_noise=rank_noise,
            rank_ood=rank_ood,
        )

    @staticmethod
    def _build_ranking(
        subcircuit_data: list[dict],
        score_key: str,
        metric_name: str,
        description: str,
    ) -> RankingResult:
        """Build a ranking from subcircuit data."""
        # Build subcircuit stats
        subcircuits = []
        for d in subcircuit_data:
            subcircuits.append(SubcircuitStats(
                subcircuit_idx=d["subcircuit_idx"],
                node_pattern=d["node_pattern"],
                edge_variant_rank=d["edge_variant_rank"],
                sparsity=d["sparsity"],
                accuracy=d["accuracy"],
                score=d[score_key],
            ))

        # Sort by score descending
        subcircuits.sort(key=lambda x: x.score, reverse=True)

        # Aggregate by node pattern
        node_pattern_data = {}
        for sc in subcircuits:
            np = sc.node_pattern
            if np not in node_pattern_data:
                node_pattern_data[np] = {
                    "scores": [],
                    "sparsity": sc.sparsity,
                    "best_score": sc.score,
                    "best_edge": sc.edge_variant_rank,
                    "best_idx": sc.subcircuit_idx,
                }
            node_pattern_data[np]["scores"].append(sc.score)
            if sc.score > node_pattern_data[np]["best_score"]:
                node_pattern_data[np]["best_score"] = sc.score
                node_pattern_data[np]["best_edge"] = sc.edge_variant_rank
                node_pattern_data[np]["best_idx"] = sc.subcircuit_idx

        # Build node pattern stats
        node_patterns = []
        for np, data in node_pattern_data.items():
            scores = data["scores"]
            node_patterns.append(NodePatternStats(
                node_pattern=np,
                n_edge_variants=len(scores),
                sparsity=data["sparsity"],
                mean_score=sum(scores) / len(scores) if scores else 0,
                max_score=max(scores) if scores else 0,
                min_score=min(scores) if scores else 0,
                best_edge_variant=data["best_edge"],
                best_subcircuit_idx=data["best_idx"],
            ))

        # Sort node patterns by max score
        node_patterns.sort(key=lambda x: x.max_score, reverse=True)

        return RankingResult(
            metric_name=metric_name,
            description=description,
            node_patterns=node_patterns,
            subcircuits=subcircuits,
        )

    def save(self, output_dir: str) -> dict[str, str]:
        """Save slice analysis to JSON files."""
        obs_dir = os.path.join(output_dir, "observational")
        os.makedirs(obs_dir, exist_ok=True)

        paths = {}

        if self.rank_accuracy:
            path = os.path.join(obs_dir, "rank_accuracy.json")
            with open(path, "w") as f:
                json.dump(self.rank_accuracy.to_dict(), f, indent=2)
            paths["rank_accuracy"] = path

        if self.rank_noise:
            path = os.path.join(obs_dir, "rank_noise_perturbations.json")
            with open(path, "w") as f:
                json.dump(self.rank_noise.to_dict(), f, indent=2)
            paths["rank_noise_perturbations"] = path

        if self.rank_ood:
            path = os.path.join(obs_dir, "rank_out_distribution_transformations.json")
            with open(path, "w") as f:
                json.dump(self.rank_ood.to_dict(), f, indent=2)
            paths["rank_ood"] = path

        return paths


@dataclass
class InterventionalSlice:
    """Interventional analysis slice - ranks by intervention metrics."""

    gate_name: str
    rank_in_circuit_id: RankingResult = None
    rank_in_circuit_ood: RankingResult = None
    rank_out_circuit_id: RankingResult = None
    rank_out_circuit_ood: RankingResult = None
    rank_overall: RankingResult = None

    @classmethod
    def from_trial(
        cls,
        trial: "TrialResult",
        gate_name: str,
        width: int,
        depth: int,
        circuits: list = None,
        input_size: int = 2,
    ) -> "InterventionalSlice":
        """Build interventional slice from trial data."""
        best_keys = trial.metrics.per_gate_bests.get(gate_name, [])
        bests_faith = trial.metrics.per_gate_bests_faith.get(gate_name, [])
        gate_metrics = trial.metrics.per_gate_metrics.get(gate_name)

        # Build accuracy lookup by node_mask_idx
        acc_lookup = {}
        if gate_metrics:
            for sm in gate_metrics.subcircuit_metrics:
                acc_lookup[sm.idx] = sm.accuracy or 0

        # Count max edge variants per node pattern for sparsity estimation
        edge_variants_per_node = {}
        for key in best_keys:
            node_idx, _ = parse_subcircuit_idx(width, depth, key)
            edge_variants_per_node[node_idx] = edge_variants_per_node.get(node_idx, 0) + 1
        max_edge_variants = max(edge_variants_per_node.values()) if edge_variants_per_node else 10

        subcircuit_data = []
        for i, key in enumerate(best_keys):
            node_idx, edge_rank = parse_subcircuit_idx(width, depth, key)
            faith = bests_faith[i] if i < len(bests_faith) else None

            sparsity = _get_subcircuit_edge_sparsity(
                key, circuits, width, depth, input_size, edge_rank, max_edge_variants
            )
            accuracy = acc_lookup.get(node_idx, 0)  # Use node_idx, not flat key

            # Extract interventional scores
            in_circuit_id = 0.0
            in_circuit_ood = 0.0
            out_circuit_id = 0.0
            out_circuit_ood = 0.0
            overall_int = 0.0

            if faith and faith.interventional:
                inv = faith.interventional
                # Use mean_* attributes from InterventionalMetrics
                in_circuit_id = inv.mean_in_circuit_similarity or 0
                in_circuit_ood = inv.mean_in_circuit_similarity_ood or 0
                out_circuit_id = inv.mean_out_circuit_similarity or 0
                out_circuit_ood = inv.mean_out_circuit_similarity_ood or 0
                overall_int = inv.overall_interventional or 0

            subcircuit_data.append({
                "subcircuit_idx": key,
                "node_pattern": node_idx,
                "edge_variant_rank": edge_rank,
                "sparsity": sparsity,
                "accuracy": accuracy,
                "in_circuit_id": in_circuit_id,
                "in_circuit_ood": in_circuit_ood,
                "out_circuit_id": out_circuit_id,
                "out_circuit_ood": out_circuit_ood,
                "overall": overall_int,
            })

        return cls(
            gate_name=gate_name,
            rank_in_circuit_id=ObservationalSlice._build_ranking(
                subcircuit_data, "in_circuit_id", "in_circuit_id",
                "Ranking by in-circuit in-distribution intervention similarity"
            ),
            rank_in_circuit_ood=ObservationalSlice._build_ranking(
                subcircuit_data, "in_circuit_ood", "in_circuit_ood",
                "Ranking by in-circuit out-of-distribution intervention similarity"
            ),
            rank_out_circuit_id=ObservationalSlice._build_ranking(
                subcircuit_data, "out_circuit_id", "out_circuit_id",
                "Ranking by out-circuit in-distribution intervention similarity"
            ),
            rank_out_circuit_ood=ObservationalSlice._build_ranking(
                subcircuit_data, "out_circuit_ood", "out_circuit_ood",
                "Ranking by out-circuit out-of-distribution intervention similarity"
            ),
            rank_overall=ObservationalSlice._build_ranking(
                subcircuit_data, "overall", "overall_interventional",
                "Ranking by overall interventional faithfulness"
            ),
        )

    def save(self, output_dir: str) -> dict[str, str]:
        """Save slice analysis to JSON files."""
        int_dir = os.path.join(output_dir, "interventional")
        os.makedirs(int_dir, exist_ok=True)

        paths = {}
        rankings = [
            (self.rank_in_circuit_id, "rank_in_circuit_id.json"),
            (self.rank_in_circuit_ood, "rank_in_circuit_ood.json"),
            (self.rank_out_circuit_id, "rank_out_circuit_id.json"),
            (self.rank_out_circuit_ood, "rank_out_circuit_ood.json"),
            (self.rank_overall, "rank_overall.json"),
        ]

        for ranking, filename in rankings:
            if ranking:
                path = os.path.join(int_dir, filename)
                with open(path, "w") as f:
                    json.dump(ranking.to_dict(), f, indent=2)
                paths[filename.replace(".json", "")] = path

        return paths


@dataclass
class CounterfactualSlice:
    """Counterfactual analysis slice - ranks by counterfactual metrics."""

    gate_name: str
    rank_sufficiency: RankingResult = None  # denoising in-circuit
    rank_completeness: RankingResult = None  # denoising out-circuit
    rank_necessity: RankingResult = None  # noising in-circuit
    rank_independence: RankingResult = None  # noising out-circuit
    rank_overall: RankingResult = None

    @classmethod
    def from_trial(
        cls,
        trial: "TrialResult",
        gate_name: str,
        width: int,
        depth: int,
        circuits: list = None,
        input_size: int = 2,
    ) -> "CounterfactualSlice":
        """Build counterfactual slice from trial data."""
        best_keys = trial.metrics.per_gate_bests.get(gate_name, [])
        bests_faith = trial.metrics.per_gate_bests_faith.get(gate_name, [])
        gate_metrics = trial.metrics.per_gate_metrics.get(gate_name)

        # Build accuracy lookup by node_mask_idx
        acc_lookup = {}
        if gate_metrics:
            for sm in gate_metrics.subcircuit_metrics:
                acc_lookup[sm.idx] = sm.accuracy or 0

        # Count max edge variants per node pattern for sparsity estimation
        edge_variants_per_node = {}
        for key in best_keys:
            node_idx, _ = parse_subcircuit_idx(width, depth, key)
            edge_variants_per_node[node_idx] = edge_variants_per_node.get(node_idx, 0) + 1
        max_edge_variants = max(edge_variants_per_node.values()) if edge_variants_per_node else 10

        subcircuit_data = []
        for i, key in enumerate(best_keys):
            node_idx, edge_rank = parse_subcircuit_idx(width, depth, key)
            faith = bests_faith[i] if i < len(bests_faith) else None

            sparsity = _get_subcircuit_edge_sparsity(
                key, circuits, width, depth, input_size, edge_rank, max_edge_variants
            )
            accuracy = acc_lookup.get(node_idx, 0)  # Use node_idx, not flat key

            # Extract counterfactual scores
            sufficiency = 0.0
            completeness = 0.0
            necessity = 0.0
            independence = 0.0
            overall_cf = 0.0

            if faith and faith.counterfactual:
                cf = faith.counterfactual
                # Use mean_* attributes from CounterfactualMetrics
                sufficiency = cf.mean_sufficiency or 0
                completeness = cf.mean_completeness or 0
                necessity = cf.mean_necessity or 0
                independence = cf.mean_independence or 0
                overall_cf = cf.overall_counterfactual or 0

            subcircuit_data.append({
                "subcircuit_idx": key,
                "node_pattern": node_idx,
                "edge_variant_rank": edge_rank,
                "sparsity": sparsity,
                "accuracy": accuracy,
                "sufficiency": sufficiency,
                "completeness": completeness,
                "necessity": necessity,
                "independence": independence,
                "overall": overall_cf,
            })

        return cls(
            gate_name=gate_name,
            rank_sufficiency=ObservationalSlice._build_ranking(
                subcircuit_data, "sufficiency", "sufficiency",
                "Ranking by sufficiency (denoising in-circuit): can the circuit alone produce correct output?"
            ),
            rank_completeness=ObservationalSlice._build_ranking(
                subcircuit_data, "completeness", "completeness",
                "Ranking by completeness (denoising out-circuit): does the circuit contain all necessary components?"
            ),
            rank_necessity=ObservationalSlice._build_ranking(
                subcircuit_data, "necessity", "necessity",
                "Ranking by necessity (noising in-circuit): is the circuit necessary for correct output?"
            ),
            rank_independence=ObservationalSlice._build_ranking(
                subcircuit_data, "independence", "independence",
                "Ranking by independence (noising out-circuit): is the rest of the model irrelevant?"
            ),
            rank_overall=ObservationalSlice._build_ranking(
                subcircuit_data, "overall", "overall_counterfactual",
                "Ranking by overall counterfactual faithfulness"
            ),
        )

    def save(self, output_dir: str) -> dict[str, str]:
        """Save slice analysis to JSON files."""
        cf_dir = os.path.join(output_dir, "counterfactual")
        os.makedirs(cf_dir, exist_ok=True)

        paths = {}
        rankings = [
            (self.rank_sufficiency, "rank_sufficiency.json"),
            (self.rank_completeness, "rank_completeness.json"),
            (self.rank_necessity, "rank_necessity.json"),
            (self.rank_independence, "rank_independence.json"),
            (self.rank_overall, "rank_overall.json"),
        ]

        for ranking, filename in rankings:
            if ranking:
                path = os.path.join(cf_dir, filename)
                with open(path, "w") as f:
                    json.dump(ranking.to_dict(), f, indent=2)
                paths[filename.replace(".json", "")] = path

        return paths
