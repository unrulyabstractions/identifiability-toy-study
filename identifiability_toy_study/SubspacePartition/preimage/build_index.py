import faiss
from faiss import read_index, write_index
import os
import torch
import json
from tqdm import tqdm
from glob import glob
import numpy as np
import argparse
from pathlib import Path
from utils import *

torch.set_grad_enabled(False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str)
parser.add_argument("--override", action="store_true")
parser.add_argument("--euclidean", action="store_true")
args = parser.parse_args()

assert os.getcwd().endswith("preimage")
exp_dir = Path("..") / "trainedRs" / args.exp_name
assert exp_dir.exists()
output_dir = Path("..") / "visualizations" / f"index-{args.exp_name}"
if not output_dir.exists():
    output_dir.mkdir(parents=True)

with open(exp_dir / "training_args.json") as f:
    exp_cfg = json.load(f)
cosine = not args.euclidean
subtract_mean = False



for file in glob("R*.pt", root_dir=exp_dir):
    _, model_name, site_name = file[:-3].split("-")
    if site_name != "x11.mid":
        continue

    output_dir_site = output_dir / f"{model_name}-{site_name}"
    if output_dir_site.exists():
        if args.override:
            print("delete existing index...")
            for idx_file in glob("*index", root_dir=output_dir_site):
                os.remove(os.path.join(output_dir_site, idx_file))
        else:
            continue
    else:
        output_dir_site.mkdir(parents=True)

    R_path = exp_dir / file
    print("build index using R from", R_path)
    R = torch.load(R_path, map_location="cpu")["R.parametrizations.weight.0.base"]
    R = R.to(device)

    with open(exp_dir / f"R_config{file[1:-3]}.json") as f:
        partition = json.load(f)["partition"]
    
    if cosine:
        indices = [faiss.IndexFlatIP(p) for p in partition]
    else:
        indices = [faiss.IndexFlatL2(p) for p in partition]

    act_data_path = Path("..") / "visualizations" / f"shared_acts-{model_name}" / f"{site_name}.pt"
    acts = torch.load(act_data_path)

    batch_size = 1024
    norms = torch.empty((acts.size(0), len(partition)), device=device, dtype=torch.float)

    if cosine and subtract_mean:
        random_idx = torch.randperm(acts.size(0))
        sum = 0
        total_num = 0
        for i in range(0, min(acts.size(0), 50_000), batch_size):
            rotated = acts[random_idx[i: i+batch_size]].to(device).to(R.dtype) @ R
            sum += rotated.sum(dim=0)
            total_num += rotated.size(0)
        mean = (sum / total_num).unsqueeze(0)

    for i in tqdm(range(0, acts.size(0), batch_size)):
        rotated = acts[i: i+batch_size].to(device).to(R.dtype) @ R
        if cosine and subtract_mean:
            rotated -= mean
        for j, (chunk, index) in enumerate(zip(rotated.split(partition, dim=1), indices)):
            chunk_norm = torch.linalg.vector_norm(chunk, dim=1, keepdim=True).clamp(min=1e-8)
            norms[i: i+batch_size, j] = chunk_norm.squeeze(1)
            if cosine:
                chunk /= chunk_norm
            index.add(chunk.cpu())

    # save indices
    print("saving indices...")
    for i, (index, p) in enumerate(zip(indices, partition)):
        assert index.d == p        
        write_index(index, os.path.join(output_dir_site, f"{i}-{p}.index"))
    np.save(os.path.join(output_dir_site, "norms.npy"), norms.cpu().numpy())