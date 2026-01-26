#!/usr/bin/env python3
"""
Benchmark different parallelization configurations.

Tests combinations of:
- CPU vs MPS for batched evaluation
- ParallelTasks enabled/disabled for different stages
- Different worker counts
- Precomputed masks vs on-the-fly

System: M4 Max (16 cores, 48GB RAM, 40 GPU cores)
"""

import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identifiability_toy_study.common.neural_model import MLP
from identifiability_toy_study.common.circuit import enumerate_all_valid_circuit
from identifiability_toy_study.common.batched_eval import (
    batch_evaluate_subcircuits,
    batch_compute_metrics,
    precompute_circuit_masks,
)
from identifiability_toy_study.parallelization import ParallelTasks


@dataclass
class BenchmarkResult:
    name: str
    time_ms: float
    throughput: float  # circuits/second
    memory_mb: float


def get_memory_mb() -> float:
    """Get current memory usage in MB."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def benchmark(name: str, fn: Callable, n_iters: int = 5, warmup: int = 1) -> BenchmarkResult:
    """Run a benchmark with warmup."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Sync if using MPS
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    mem_before = get_memory_mb()

    # Time
    start = time.time()
    for _ in range(n_iters):
        fn()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    elapsed = (time.time() - start) / n_iters

    mem_after = get_memory_mb()

    return BenchmarkResult(
        name=name,
        time_ms=elapsed * 1000,
        throughput=0,  # Will be set by caller
        memory_mb=mem_after - mem_before,
    )


def run_benchmarks():
    print("=" * 70)
    print("PARALLELIZATION BENCHMARK")
    print("=" * 70)
    print(f"System: {torch.backends.mps.is_available() and 'MPS available' or 'CPU only'}")
    print(f"CPU cores: {os.cpu_count()}")
    print()

    # Configuration
    width, depth = 4, 3
    batch_size = 128
    n_circuits_expected = (2**width - 1) ** depth

    print(f"Model: width={width}, depth={depth}")
    print(f"Expected circuits: {n_circuits_expected}")
    print(f"Batch size: {batch_size}")
    print()

    # Create model and circuits
    print("Setting up...")
    model_cpu = MLP(hidden_sizes=[width] * depth, input_size=2, output_size=1, device="cpu")
    circuits = enumerate_all_valid_circuit(model_cpu, use_tqdm=False)
    print(f"Circuits enumerated: {len(circuits)}")

    x_cpu = torch.randn(batch_size, 2)
    y_target = torch.randint(0, 2, (batch_size, 1)).float()

    with torch.no_grad():
        y_pred_cpu = model_cpu(x_cpu)

    results = []

    # =========================================================================
    # 1. BATCHED EVALUATION: CPU vs MPS, with/without precompute
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. BATCHED EVALUATION (gate_metrics)")
    print("=" * 70)

    for device in ["cpu", "mps"]:
        if device == "mps" and not torch.backends.mps.is_available():
            continue

        x = x_cpu.to(device)
        y_target_d = y_target.to(device)
        y_pred = y_pred_cpu.to(device)

        # Without precompute
        def eval_no_precompute():
            return batch_compute_metrics(model_cpu, circuits, x, y_target_d, y_pred, eval_device=device)

        result = benchmark(f"{device.upper()} (no precompute)", eval_no_precompute)
        result.throughput = len(circuits) / (result.time_ms / 1000)
        results.append(result)
        print(f"  {result.name}: {result.time_ms:.1f}ms, {result.throughput:.0f} circuits/s")

        # With precompute
        precomputed = precompute_circuit_masks(circuits, len(model_cpu.layers), gate_idx=0, device=device)

        def eval_precompute():
            return batch_compute_metrics(
                model_cpu, circuits, x, y_target_d, y_pred,
                precomputed_masks=precomputed, eval_device=device
            )

        result = benchmark(f"{device.upper()} (precomputed)", eval_precompute)
        result.throughput = len(circuits) / (result.time_ms / 1000)
        results.append(result)
        print(f"  {result.name}: {result.time_ms:.1f}ms, {result.throughput:.0f} circuits/s")

    # =========================================================================
    # 2. STRUCTURE ANALYSIS: ParallelTasks vs Sequential
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. STRUCTURE ANALYSIS (CPU-bound)")
    print("=" * 70)

    # Sequential
    def structure_sequential():
        return [c.analyze_structure() for c in circuits]

    result = benchmark("Sequential", structure_sequential)
    result.throughput = len(circuits) / (result.time_ms / 1000)
    results.append(result)
    print(f"  {result.name}: {result.time_ms:.1f}ms, {result.throughput:.0f} circuits/s")

    # Parallel with different worker counts
    for workers in [2, 4, 8, 12, 16]:
        def structure_parallel(w=workers):
            with ParallelTasks(max_workers=w) as tasks:
                futures = [tasks.submit(c.analyze_structure) for c in circuits]
            return [f.result() for f in futures]

        result = benchmark(f"Parallel ({workers} workers)", structure_parallel)
        result.throughput = len(circuits) / (result.time_ms / 1000)
        results.append(result)
        print(f"  {result.name}: {result.time_ms:.1f}ms, {result.throughput:.0f} circuits/s")

    # =========================================================================
    # 3. COMBINED PIPELINE: Different configurations
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. COMBINED PIPELINE (enum + structure + metrics)")
    print("=" * 70)

    configs = [
        ("CPU sequential", "cpu", False, 1),
        ("CPU parallel(8)", "cpu", True, 8),
        ("MPS sequential", "mps", False, 1),
        ("MPS parallel(8)", "mps", True, 8),
        ("MPS parallel(16)", "mps", True, 16),
    ]

    for name, device, parallel_structure, workers in configs:
        if device == "mps" and not torch.backends.mps.is_available():
            continue

        x = x_cpu.to(device)
        y_target_d = y_target.to(device)
        y_pred = y_pred_cpu.to(device)

        def full_pipeline(d=device, parallel=parallel_structure, w=workers):
            # 1. Enumerate circuits (always CPU)
            circs = enumerate_all_valid_circuit(model_cpu, use_tqdm=False)

            # 2. Structure analysis
            if parallel:
                with ParallelTasks(max_workers=w) as tasks:
                    futures = [tasks.submit(c.analyze_structure) for c in circs]
                structures = [f.result() for f in futures]
            else:
                structures = [c.analyze_structure() for c in circs]

            # 3. Precompute masks
            precomputed = precompute_circuit_masks(circs, len(model_cpu.layers), gate_idx=0, device=d)

            # 4. Batched metrics
            x_d = x_cpu.to(d)
            y_t = y_target.to(d)
            y_p = y_pred_cpu.to(d)
            accs, logits, bits = batch_compute_metrics(
                model_cpu, circs, x_d, y_t, y_p,
                precomputed_masks=precomputed, eval_device=d
            )

            return accs, logits, bits, structures

        result = benchmark(name, full_pipeline, n_iters=3, warmup=1)
        result.throughput = len(circuits) / (result.time_ms / 1000)
        results.append(result)
        print(f"  {result.name}: {result.time_ms:.1f}ms, {result.throughput:.0f} circuits/s")

    # =========================================================================
    # 4. MEMORY USAGE: Different batch sizes
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. MEMORY VS SPEED TRADEOFF (batch size)")
    print("=" * 70)

    batch_sizes = [32, 64, 128, 256, 512, 1024]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    precomputed = precompute_circuit_masks(circuits, len(model_cpu.layers), gate_idx=0, device=device)

    for bs in batch_sizes:
        x = torch.randn(bs, 2, device=device)
        y_target_d = torch.randint(0, 2, (bs, 1), device=device).float()

        with torch.no_grad():
            x_for_pred = torch.randn(bs, 2)
            y_pred_d = model_cpu(x_for_pred).to(device)

        def eval_batch(x=x, yt=y_target_d, yp=y_pred_d):
            return batch_compute_metrics(
                model_cpu, circuits, x, yt, yp,
                precomputed_masks=precomputed, eval_device=device
            )

        result = benchmark(f"Batch {bs}", eval_batch)
        result.throughput = len(circuits) * bs / (result.time_ms / 1000)
        results.append(result)
        print(f"  {result.name}: {result.time_ms:.1f}ms, {result.throughput:.0f} samples*circuits/s")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: RECOMMENDED CONFIGURATION")
    print("=" * 70)

    # Find best for each category
    eval_results = [r for r in results if "precomputed" in r.name.lower() or "no precompute" in r.name.lower()]
    if eval_results:
        best_eval = max(eval_results, key=lambda r: r.throughput)
        print(f"  Best batched eval: {best_eval.name} ({best_eval.time_ms:.1f}ms)")

    structure_results = [r for r in results if "worker" in r.name.lower() or r.name == "Sequential"]
    if structure_results:
        best_structure = max(structure_results, key=lambda r: r.throughput)
        print(f"  Best structure analysis: {best_structure.name} ({best_structure.time_ms:.1f}ms)")

    pipeline_results = [r for r in results if "parallel" in r.name.lower() and "CPU" in r.name or "MPS" in r.name]
    if pipeline_results:
        best_pipeline = max(pipeline_results, key=lambda r: r.throughput)
        print(f"  Best pipeline: {best_pipeline.name} ({best_pipeline.time_ms:.1f}ms)")


if __name__ == "__main__":
    run_benchmarks()
