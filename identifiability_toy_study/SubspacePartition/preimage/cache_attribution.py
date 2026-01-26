import faiss
from faiss import read_index
import os
import torch
import json
from tqdm import tqdm
from glob import glob
import pickle
import random
import numpy as np
from transformer_lens import HookedTransformer
import torch.nn.functional as F
import re
from pathlib import Path
import shutil
import argparse
import itertools
from utils import *



def zero_baseline_IG(model: HookedTransformer, batch_ids, batch_pos_ids, batch_q_acts, R_chunk, act_site):
    stop_at_layer = int(re.search(r"blocks\.(\d+)\.", act_site).group(1)) + 1

    bz = batch_ids.size(0)
    fix_attn = True     # in some cases, much better than no fixing

    if fix_attn:
        with torch.no_grad():
            _, cache = model.run_with_cache(batch_ids, stop_at_layer=stop_at_layer, names_filter=lambda n: n.endswith("hook_pattern"))

        def override_attn(a, hook):
            return cache[hook.name].unsqueeze(0).expand(num_interpolation, -1, -1, -1, -1).flatten(end_dim=1)

    embed = model.embed(batch_ids)  
    pos_embed = model.pos_embed(batch_ids, 0, None)  
    x = embed + pos_embed

    # interpolate
    alpha = torch.arange(1, num_interpolation+1).to(device) / num_interpolation
    initial_residual = x.unsqueeze(0).expand(num_interpolation, -1, -1, -1) * alpha.view(-1, 1, 1, 1)
    initial_residual = initial_residual.flatten(end_dim=1)
    initial_residual.requires_grad_(True)

    acts = None
    def collect_act(a, hook):
        nonlocal acts
        acts = a
    
    fwd_hooks = [(act_site, collect_act)]
    if fix_attn:
        fwd_hooks.append((lambda n: n.endswith("hook_pattern"), override_attn))

    model.run_with_hooks(initial_residual, start_at_layer=0, stop_at_layer=stop_at_layer, 
                        fwd_hooks=fwd_hooks)
    
    acts = acts.view(num_interpolation, bz, acts.size(1), acts.size(2))
    batch_pos_ids = batch_pos_ids.view(1, -1, 1, 1).expand(num_interpolation, -1, -1, acts.size(-1))
    acts = torch.gather(acts, dim=2, index=batch_pos_ids).squeeze(2) # num_interpo, bz, h_dim

    acts = torch.bmm(acts, R_chunk.unsqueeze(0).expand(num_interpolation, -1, -1))
    sim = F.cosine_similarity(acts, batch_q_acts.unsqueeze(0), dim=-1)
    sim.sum().backward()

    grad = initial_residual.grad.view(num_interpolation, bz, initial_residual.size(1), initial_residual.size(2))
    integrated = grad.mean(dim=0)
    integrated = (x * integrated).sum(dim=-1)   # bz, seq_len

    return integrated

def pad_inputs(input_ids: list[list[int]], pad_id):
    max_len = max(len(seq) for seq in input_ids)
    new_input_ids = []
    for seq in input_ids:
        new_input_ids.append(seq + [pad_id] * (max_len - len(seq)))
    return new_input_ids

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(0)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str)
parser.add_argument("--num_preimage", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--override", action="store_true")
parser.add_argument("--euclidean", action="store_true")
args = parser.parse_args()

batch_size = args.batch_size
num_interpolation = 10

assert os.getcwd().endswith("preimage")
exp_dir = Path("..") / "trainedRs" / args.exp_name
index_dir = Path("..") / "visualizations" / f"index-{args.exp_name}"
assert index_dir.exists()
output_dir = Path("..") / "visualizations" / f"preimage-{args.exp_name}"
if not output_dir.exists():
    output_dir.mkdir()

with open(exp_dir / "training_args.json") as f:
    exp_cfg = json.load(f)
cosine = not args.euclidean
subtract_mean = False

for file in glob("R*.pt", root_dir=exp_dir):
    _, model_name, site_name = file[:-3].split("-")

    act_site = short_name_to_site_name(site_name)

    output_dir_site = output_dir / f"{model_name}-{site_name}"
    if output_dir_site.exists():
        if args.override:
            print("delete existing folder...")
            shutil.rmtree(output_dir_site)
        else:
            continue

    R_path = exp_dir / file
    print("loading R... from", R_path)
    R = torch.load(R_path, map_location="cpu")["R.parametrizations.weight.0.base"]
    R = R.to(device)
    with open(exp_dir / f"R_config{file[1:-3]}.json") as f:
        partition = json.load(f)["partition"]


    R_chunks = R.split(partition, dim=1)

    print("reading index...")
    indices = {}
    index_dir_site = index_dir / f"{model_name}-{site_name}"
    for file in glob("*index", root_dir=index_dir_site):
        indices[int(file.split("-")[0])] = read_index(str(index_dir_site / file))
    assert [indices[i].d for i in range(len(indices))] == partition
    assert all(isinstance(indices[i], faiss.IndexFlatIP) == cosine for i in indices), [type(indices[i]) for i in indices]
    norms = np.load(index_dir_site / "norms.npy")

    print("reading input...")
    with open(Path("..") / "visualizations" / f"shared_acts-{model_name}" / "str_tokens.pkl", "rb") as f:
        cached_input = pickle.load(f)
    seq_lens = [len(seq) for seq in cached_input]
    seq_edges = list(itertools.accumulate(seq_lens, initial=0))


    print("loading model...")
    model = HookedTransformer.from_pretrained(
        to_valid_model_name(model_name),
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        device=device,
    )
    for p in model.parameters():
        p.requires_grad_(False)


    num_sample = 20 # sample per preimage
    q_idx = list(range(indices[0].ntotal))
    random.shuffle(q_idx)
    preimages = {}  # {q_idx: {space_idx: [(d1, i1), (d2, i2), ...]} }
    query_acts = {} # {q_idx: {space_idx: tensor}}
    for q_i in q_idx[:args.num_preimage]:
        
        preimage = {}
        q_i_acts = {}
        for space_i in range(len(partition)):
            index = indices[space_i]  
            query_act = np.empty((index.d,), dtype=np.float32)
            index.reconstruct(q_i, query_act)
            D, I = index.search(query_act[None, :], k=num_sample)

            D = D[0].tolist()
            I = I[0].tolist()

            preimage[space_i] = list(zip(D, I))
            q_i_acts[space_i] = torch.from_numpy(query_act)

        preimages[q_i] = preimage
        query_acts[q_i] = q_i_acts


    for space_i in range(len(partition)):
        print("compute attribution for", space_i)
        input_ids = []
        pos_ids = []
        q_acts = []
        total_info = []
        for q_i in preimages:
            for sim, input_idx in preimages[q_i][space_i]:
                seq_idx, pos_idx = locate_str_tokens(input_idx, seq_edges)
                str_tokens = cached_input[seq_idx]

                norm = norms[input_idx, space_i].item()

                input_ids.append( model.tokenizer.convert_tokens_to_ids(str_tokens[:pos_idx+1]) )
                pos_ids.append(pos_idx)
                q_acts.append(query_acts[q_i][space_i]) 
                total_info.append((q_i, input_idx, sim, str_tokens, pos_idx, norm))
        
        attribution = []
        for batch_s in tqdm(range(0, len(input_ids), batch_size)):
            batch_ids = pad_inputs(input_ids[batch_s: batch_s+batch_size], model.tokenizer.pad_token_id)
            batch_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)

            batch_pos_ids = torch.tensor(pos_ids[batch_s: batch_s+batch_size]).to(dtype=torch.long, device=device)
            batch_q_acts = torch.stack(q_acts[batch_s: batch_s+batch_size]).to(dtype=torch.float, device=device) # already normalized

            batch_attribution = zero_baseline_IG(model, batch_ids, batch_pos_ids, batch_q_acts, R_chunks[space_i], act_site)
            attribution.extend(batch_attribution.tolist())


        for i in range(len(input_ids)):
            seq_len = len(total_info[i][3])
            if len(attribution[i]) >= seq_len:
                a = attribution[i][:seq_len]
            else:
                a = attribution[i] + [0] * (seq_len - len(attribution[i]))
            total_info[i] = total_info[i] + (a,)

        subspace_folder = output_dir_site / f"{space_i}-{partition[space_i]}-attr"
        subspace_folder.mkdir(parents=True)
        save_obj = {}
        for q_i in preimages:
            seq_idx, pos_idx = locate_str_tokens(q_i, seq_edges)
            str_tokens = cached_input[seq_idx]
            norm = norms[q_i, space_i].item()
            
            index = indices[space_i]
            query_act = np.empty((index.d,), dtype=np.float32)
            index.reconstruct(q_i, query_act)
            _, D, _ = index.range_search(query_act[None, :], -1.0 if cosine else 100000)
            counts, bin_edges = np.histogram(D, bins=100, range=(-1.0, 1.0))

            save_obj[q_i] = {"query_info": (str_tokens, pos_idx, norm, counts.tolist(), bin_edges.tolist()), "preimage": []}
        
        for item in total_info:     # q_i, input_idx, sim, str_tokens, pos_idx, norm, attr
            save_obj[item[0]]["preimage"].append(item)

        for k, v in save_obj.items():
            with open(subspace_folder / f"{k}.json", "w") as f:
                json.dump(v, f)
