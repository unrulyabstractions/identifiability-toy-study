from model_train import *
from easydict import EasyDict
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from pathlib import Path

torch.set_grad_enabled(False)
model_path = Path(".")
with open(model_path / "config.json", "r") as f:
    config = EasyDict(json.load(f))
config.x_dim = sum(config.n_feature)

model = toyModel(config.x_dim, config.h_dim)
model.load_state_dict(torch.load(model_path / "model.pt"))

# true dimension of subspaces
for w in model.W.split(config.n_feature, dim=0):
    S = torch.linalg.svdvals(w)
    print(w.size(), S)

fig, axes = plt.subplots(len(config.n_feature), 1, figsize=(3, 3 * len(config.n_feature)))
if len(config.n_feature) == 1:
    axes = [axes]
for i, (w, ax) in enumerate(zip(model.W.split(config.n_feature, dim=0), axes)):
    S = torch.linalg.svdvals(w).numpy()
    ax.plot(S, marker='o')
    start = sum(config.n_feature[:i])
    end = start + config.n_feature[i]
    ax.set_title(f"Singular values of $W_{{[:, {start}:{end}]}}$")
    ax.set_xlabel("Index")
plt.tight_layout()
plt.show()

# W^TW, b
# product = model.W @ model.W.T
# b = model.b.unsqueeze(1).numpy()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.05})
# max_abs = product.abs().max()
# heatmap1 = ax1.imshow(product.numpy(), cmap="bwr", norm=Normalize(vmin=-max_abs, vmax=max_abs))
# ax1.set_title("$W^T W$", fontsize=14)
# ax1.axis('off')

# heatmap2 = ax2.imshow(b, cmap="bwr", norm=Normalize(vmin=-max_abs, vmax=max_abs))
# ax2.set_title("$b$", fontsize=14)
# ax2.axis('off')

# cbar = fig.colorbar(heatmap1, ax=[ax1, ax2], orientation='vertical', fraction=0.02, pad=0.02)
# cbar.set_label("Weight / Bias\nElement Values", rotation=0, labelpad=20, fontsize=10, ha='left')

# plt.show()

product = model.W @ model.W.T

fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
max_abs = product.abs().max()
heatmap1 = ax.imshow(product.numpy(), cmap="bwr", norm=Normalize(vmin=-max_abs, vmax=max_abs))
ax.set_title("$W^T W$", fontsize=14)
ax.axis('off')

cbar = fig.colorbar(heatmap1, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
plt.tight_layout()
plt.show()

# ||W||
norm = torch.linalg.vector_norm(model.W, dim=1).numpy()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.05})
ax1.set_title("norm")
ax1.barh(range(len(norm)), norm, height=0.8)
ax1.invert_yaxis()

# superposition
mask = torch.eye(product.size(0))
temp = product.masked_fill(mask.bool(), 0)

s_idx = 0
superpos_per_group = []
for group_size in config.n_feature:
    avg = (temp[:, s_idx:s_idx+group_size]**2).sum(dim=1) / (1 - mask[:, s_idx:s_idx+group_size]).sum(dim=1)
    superpos_per_group.append(avg)
    s_idx += group_size
superpos_per_group = torch.stack(superpos_per_group).transpose(0, 1).numpy()

ax2.imshow(superpos_per_group, cmap="viridis", aspect=0.5)
ax2.set_title("superposition")

plt.show()
