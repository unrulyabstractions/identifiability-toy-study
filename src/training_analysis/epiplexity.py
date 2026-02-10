"""Epiplexity estimation for training analysis.

Implements prequential coding to measure structure (S_T) vs noise (H_T)
learned during training.

Theory:
    At each training step k, BEFORE updating weights, record the loss l_k.
    After all training, re-evaluate the FINAL model on the SAME data points -> l_k^final.
    The GAP (l_k - l_k^final) = info the model hadn't yet learned at step k.
    Sum of all gaps = total structural info absorbed = EPIPLEXITY (S_T).
    Sum of all final losses = residual noise = TIME-BOUNDED ENTROPY (H_T).

Formulas (all in BITS, not nats):
    S_T = sum_k max(l_k - l_k^final, 0)      # epiplexity
    H_T = sum_k l_k^final                     # time-bounded entropy
    MDL = S_T + H_T                           # total description length
"""

import math

import torch
import torch.nn as nn

from .types import (
    EpiplexityResult,
    InterpretationLabel,
    StepDiagnostics,
    TrainingAnalysis,
    TrainingRecord,
)

# Convert nats (natural log) to bits (log base 2)
NATS_TO_BITS = 1.0 / math.log(2)


def phase2_evaluate_final(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    training_record: TrainingRecord,
    device: str = "cpu",
) -> TrainingRecord:
    """Phase 2: Re-evaluate final model on all historical batches.

    For each step k, computes l_k^final = final model's loss on that batch.
    Fills in training_record.step_records[k].loss_final_bits.

    Args:
        model: The FINAL trained model (frozen, eval mode)
        x: Training input data
        y: Training target data
        training_record: TrainingRecord from Phase 1 with step_records
        device: Device to run on

    Returns:
        Same TrainingRecord with loss_final_bits filled in
    """
    model.eval()
    model.to(device)
    x = x.to(device)
    y = y.to(device)

    # Use sum reduction for total bits per batch
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    with torch.no_grad():
        for record in training_record.step_records:
            # Get the same batch that was used during training
            batch_indices = record.batch_indices
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]

            # Compute loss with final model
            logits = model(x_batch)
            loss_nats = loss_fn(logits, y_batch)
            record.loss_final_bits = loss_nats.item() * NATS_TO_BITS

    return training_record


def phase3_compute_metrics(training_record: TrainingRecord) -> EpiplexityResult:
    """Phase 3: Compute S_T, H_T, and derived metrics from the training log.

    S_T = sum_k max(l_k - l_k^final, 0)  # Structure learned
    H_T = sum_k l_k^final                 # Noise remaining

    Args:
        training_record: TrainingRecord with both loss_pre_bits and loss_final_bits

    Returns:
        EpiplexityResult with all computed metrics
    """
    S_T = 0.0
    H_T = 0.0

    for r in training_record.step_records:
        gap = r.loss_pre_bits - r.loss_final_bits
        S_T += max(gap, 0.0)
        H_T += r.loss_final_bits

    MDL = S_T + H_T
    total_steps = training_record.total_steps
    total_samples = training_record.total_samples

    return EpiplexityResult(
        S_T=S_T,
        H_T=H_T,
        MDL=MDL,
        structure_ratio=S_T / MDL if MDL > 0 else 0.0,
        total_steps=total_steps,
        total_samples=total_samples,
        S_T_per_sample=S_T / total_samples if total_samples > 0 else 0.0,
        H_T_per_sample=H_T / total_samples if total_samples > 0 else 0.0,
    )


def compute_diagnostics(training_record: TrainingRecord) -> list[StepDiagnostics]:
    """Compute per-step diagnostics for plotting loss curves and cumulative epiplexity.

    Args:
        training_record: TrainingRecord with filled step_records

    Returns:
        List of StepDiagnostics for each training step
    """
    diagnostics = []
    cumulative = 0.0

    for r in training_record.step_records:
        absorbed = max(r.loss_pre_bits - r.loss_final_bits, 0.0)
        cumulative += absorbed
        diagnostics.append(
            StepDiagnostics(
                step=r.step,
                loss_pre_bits=r.loss_pre_bits,
                loss_final_bits=r.loss_final_bits,
                info_absorbed=absorbed,
                cumulative_S_T=cumulative,
            )
        )

    return diagnostics


def interpret(result: EpiplexityResult) -> InterpretationLabel:
    """Map numeric epiplexity scores to human-readable interpretation.

    Two axes:
        - Structure ratio r = S_T / MDL (how much is learnable structure?)
        - Absolute H_T per sample (how noisy is each observation?)

    Quadrant map:
        HIGH_S_LOW_H  = "Rich, clean"       -> best for transfer/OOD
        HIGH_S_HIGH_H = "Rich, noisy"       -> structure exists but buried in noise
        LOW_S_LOW_H   = "Simple, clean"     -> trivial problem
        LOW_S_HIGH_H  = "Unlearnable noise" -> avoid for training

    Args:
        result: EpiplexityResult with computed metrics

    Returns:
        InterpretationLabel with human-readable diagnosis
    """
    r = result.structure_ratio
    h = result.H_T_per_sample

    # Structure axis
    if r >= 0.5:
        structure_label = "High structure"
    elif r >= 0.2:
        structure_label = "Moderate structure"
    elif r >= 0.05:
        structure_label = "Low structure"
    else:
        structure_label = "Noise-dominated"

    # Noise axis
    # h is bits per sample. For binary gates with N outputs, max = N bits.
    # Thresholds: < 0.1 bits/sample = low, > 0.5 = high
    if h < 0.1:
        noise_label = "Low noise"
    elif h < 0.5:
        noise_label = "Moderate noise"
    else:
        noise_label = "High noise"

    # Quadrant determination
    high_s = r >= 0.3
    high_h = h >= 0.3

    if high_s and not high_h:
        quadrant = "HIGH_S_LOW_H"
        data_quality = "Rich, clean - prefer for transfer/OOD"
    elif high_s and high_h:
        quadrant = "HIGH_S_HIGH_H"
        data_quality = "Rich but noisy - structure exists, buried in noise"
    elif not high_s and not high_h:
        quadrant = "LOW_S_LOW_H"
        data_quality = "Simple, clean - trivial problem, little to transfer"
    else:
        quadrant = "LOW_S_HIGH_H"
        data_quality = "Unlearnable noise - avoid for training"

    return InterpretationLabel(
        structure_label=structure_label,
        noise_label=noise_label,
        data_quality=data_quality,
        quadrant=quadrant,
    )


def do_training_analysis(
    training_record: TrainingRecord,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: str = "cpu",
) -> TrainingAnalysis:
    """Full epiplexity analysis pipeline.

    Phases:
        Phase 1 (already done in train_mlp): Record loss before each update
        Phase 2: Re-evaluate final model on same batches
        Phase 3: Compute S_T, H_T, MDL metrics
        Phase 4: Interpret results

    Args:
        training_record: TrainingRecord from train_mlp (Phase 1 complete)
        model: The FINAL trained model
        x: Training input data (for Phase 2 replay)
        y: Training target data (for Phase 2 replay)
        device: Device to run on

    Returns:
        TrainingAnalysis with epiplexity results and interpretation
    """
    # Skip if no step records (training was skipped or failed)
    if not training_record.step_records:
        return TrainingAnalysis()

    # Phase 2: Re-evaluate with final model
    training_record = phase2_evaluate_final(model, x, y, training_record, device)

    # Phase 3: Compute metrics
    epi_result = phase3_compute_metrics(training_record)

    # Phase 4: Interpret
    interpretation = interpret(epi_result)

    # Compute diagnostics for plotting
    diagnostics = compute_diagnostics(training_record)

    return TrainingAnalysis(
        epi_result=epi_result,
        interpretation=interpretation,
        diagnostics=diagnostics,
    )
