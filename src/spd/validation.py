"""SPD validation metrics: MMCS, ML2R, and faithfulness.

Metrics from the SPD paper for validating decomposition quality:
- MMCS (Mean Max Cosine Similarity): Directional alignment
- ML2R (Mean L2 Ratio): Magnitude correspondence
- Faithfulness loss: MSE between target and reconstructed weights
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.model import DecomposedMLP


def compute_validation_metrics(
    decomposed_model: "DecomposedMLP",
) -> dict[str, float]:
    """
    Compute SPD validation metrics: MMCS and ML2R.

    MMCS (Mean Max Cosine Similarity): Measures directional alignment between
    learned subcomponents and target model weights. Value of 1.0 means perfect
    directional match.

    ML2R (Mean L2 Ratio): Measures magnitude correspondence between reconstructed
    and original weights. Value close to 1.0 means minimal shrinkage.

    Args:
        decomposed_model: Trained SPD decomposition

    Returns:
        Dict with 'mmcs' and 'ml2r' metrics (higher is better, 1.0 is perfect)
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return {"mmcs": 0.0, "ml2r": 0.0, "faithfulness_loss": float("inf")}

    component_model = decomposed_model.component_model

    mmcs_values = []
    ml2r_values = []
    faithfulness_losses = []

    for module_name, components in component_model.components.items():
        # Get target weight
        target_weight = component_model.target_weight(module_name)

        # Get reconstructed weight (U @ V^T)
        U = components.U  # [C, d_out]
        V = components.V  # [d_in, C]
        reconstructed = (V @ U).T  # [d_out, d_in]

        target_np = target_weight.detach().cpu().numpy()
        recon_np = reconstructed.detach().cpu().numpy()

        # Faithfulness loss (MSE)
        mse = ((target_np - recon_np) ** 2).mean()
        faithfulness_losses.append(mse)

        # MMCS: For each column in target, find max cosine similarity with any component
        for col_idx in range(target_np.shape[1]):
            target_col = target_np[:, col_idx]
            target_norm = np.linalg.norm(target_col)
            if target_norm < 1e-8:
                continue

            max_cos_sim = 0.0
            for c in range(U.shape[0]):
                # Component c contributes: U[c,:] * V[:,c]^T as rank-1 matrix
                # For column col_idx, the contribution is U[c,:] * V[col_idx, c]
                comp_col = U[c, :].detach().cpu().numpy() * V[col_idx, c].item()
                comp_norm = np.linalg.norm(comp_col)
                if comp_norm < 1e-8:
                    continue

                cos_sim = np.dot(target_col, comp_col) / (target_norm * comp_norm)
                max_cos_sim = max(max_cos_sim, abs(cos_sim))

            mmcs_values.append(max_cos_sim)

        # ML2R: Ratio of reconstructed to target magnitude
        target_norm = np.linalg.norm(target_np, "fro")
        recon_norm = np.linalg.norm(recon_np, "fro")
        if target_norm > 1e-8:
            ml2r_values.append(recon_norm / target_norm)

    return {
        "mmcs": float(np.mean(mmcs_values)) if mmcs_values else 0.0,
        "ml2r": float(np.mean(ml2r_values)) if ml2r_values else 0.0,
        "faithfulness_loss": float(np.mean(faithfulness_losses))
        if faithfulness_losses
        else float("inf"),
    }
