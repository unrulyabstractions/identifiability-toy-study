from make_data_ioi import IOIDataset, NAMES
from make_data_greater_than import get_prompts_and_more
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from copy import deepcopy
import argparse
import os
from pathlib import Path
import json
from utils import *
from transformer_lens import HookedTransformer
from itertools import accumulate
from datasets import load_dataset
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from functools import partial

torch.set_grad_enabled(False)

def draw_figures(partition, var, effect, path: Path, normalize=True):
    print_("draw figures..")
    print_("partition\n", partition)
    print_("var\n", var)
    print_("effect\n", effect)

    partition = np.array(partition, dtype=np.float32)
    var = np.array(var)
    effect = np.abs(np.array(effect))
    if normalize:
        partition /= partition.sum()
        var /= var.sum()
        effect /= effect.sum()

    for label, denominator in [("unnormed", np.ones_like(effect)), ("dimension", partition), ("variance", var)]:
        x = effect / denominator
        gini_coef = np.abs(x[None, :] - x[:, None]).sum() / (2 * len(x) * x.sum())
        print_("gini coef", path.name, label, " ", gini_coef)

        sizes = denominator * 800

        radii = np.sqrt(sizes) / 2

        # Group nearly-identical x values
        grouped = defaultdict(list)
        for i, xi in enumerate(x):
            key = round(xi, 0)
            grouped[key].append(i)

        y = np.zeros(len(x))
        for indices in grouped.values():
            random.shuffle(indices)  

            y_pos = []
            current_y = 0
            prev_r = None
            for i in indices:
                if prev_r is not None:
                    gap = (radii[i] + prev_r) * 1.2
                    current_y += gap
                y_pos.append(current_y)
                prev_r = radii[i]

            # Center stack vertically
            y_centered = np.array(y_pos) - np.median(y_pos)
            for i, yi in zip(indices, y_centered):
                y[i] = yi

        # Plot
        plt.figure(figsize=(10, 4))
        plt.scatter(x, y, s=sizes * 5, alpha=0.6, edgecolors='k')
        plt.yticks([])
        plt.xlabel("ratio")
        plt.xticks(fontsize=14)
        ymin, ymax = plt.ylim()
        padding = (ymax - ymin) * 0.1
        plt.ylim(ymin - padding, ymax + padding)
        plt.title(f"Effect / {label}")
        plt.grid(True, axis='x', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        plt.tight_layout()
        plt.savefig(str(path)+f"-{label}.png")
        print_('figure saved')

def get_activations(model: HookedTransformer, layer: int):
    data = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)    # may be changed if not gpt2
    data = data.shuffle(buffer_size=100_000)
    data = map(lambda x:x["text"], data)
    model.reset_hooks()
    cache = model.add_caching_hooks(f"blocks.{layer}.hook_resid_post")

    caching_batch_size = 32
    all_acts = []
    for i in range(1):
        tokens = []
        while True:
            input_ids = model.tokenizer(next(data))["input_ids"]
            tokens.extend(input_ids)
            if len(tokens) >= caching_batch_size * model.cfg.n_ctx:
                break
            tokens.append(model.tokenizer.eos_token_id)
        tokens = tokens[:caching_batch_size * model.cfg.n_ctx]
        input_batch = torch.tensor(tokens, device=device, dtype=torch.long).view(caching_batch_size, -1)

        model(input_batch, stop_at_layer=layer+1)

        acts = cache[f"blocks.{layer}.hook_resid_post"].flatten(end_dim=1)
        all_acts.append(acts)
    all_acts = torch.cat(all_acts, dim=0)

    model.reset_hooks()

    return all_acts

def measure_subspace_var(R: torch.FloatTensor, partition: list[int], model: HookedTransformer, layer: int):
    acts = get_activations(model, layer)
    var = (acts @ R).var(dim=0)
    var_per_subspace = [v.sum().item() for v in var.split(partition, dim=0)]

    return var_per_subspace

def evaluate_ioi(R: torch.FloatTensor, partition: list[int], subspace_idx: int,
                prompts, 
                hooked_model: HookedTransformer, 
                layer: int, pos_lis: list[int], modified_info: str):
    assert modified_info in ["pos", "S_name"]
    tokenizer = hooked_model.tokenizer

    partition_edges = list(accumulate(partition, initial=0))
    if subspace_idx is not None:
        s, e = partition_edges[subspace_idx], partition_edges[subspace_idx+1]
    else:
        s, e = partition_edges[0], partition_edges[-1]

    should_be_names = []
    logit_diff_drop = []
    io_prob_drop = []
    for p in tqdm(prompts):
        orig_input_ids = tokenizer(p["text"])["input_ids"]
        io_name_id, s_name_id = orig_input_ids[p["pos_IO"]], orig_input_ids[p["pos_S"]]
        should_be_names.extend([io_name_id, s_name_id])
        corp_input_ids = orig_input_ids.copy()
        if modified_info == "pos":
            corp_input_ids[p["pos_S"]], corp_input_ids[p["pos_IO"]] = orig_input_ids[p["pos_IO"]], orig_input_ids[p["pos_S"]]
        elif modified_info == "S_name":
            corp_input_ids[p["pos_S"]] = orig_input_ids[p["pos_IO"]]
            corp_input_ids[p["pos_S2"]] = orig_input_ids[p["pos_IO"]]
            corp_input_ids[p["pos_IO"]] = orig_input_ids[p["pos_S"]]
        
        orig_input_ids.insert(0, tokenizer.eos_token_id)
        corp_input_ids.insert(0, tokenizer.eos_token_id)
        assert len(orig_input_ids) == len(corp_input_ids)

        pos_to_patch = [p["pos_"+n]+1 for n in pos_lis] # consider Bos
        
        # clean run
        orig_logits = hooked_model(torch.tensor([orig_input_ids], dtype=torch.long))
        orig_logit_diff = orig_logits[0, -2, io_name_id] - orig_logits[0, -2, s_name_id]
        orig_io_prob = F.softmax(orig_logits[0, -2], dim=0)[io_name_id]

        # save corrupt act
        corp_act = None
        def save_hook(x, hook):
            nonlocal corp_act
            corp_act = x

        hooked_model.run_with_hooks(
            torch.tensor([corp_input_ids], dtype=torch.long), 
            stop_at_layer=layer+1,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", save_hook)])
        
        assert corp_act is not None

        def patch_hook(x, hook):
            rotated_x = x[0, pos_to_patch] @ R
            rotated_x[:, s: e] = (corp_act[0, pos_to_patch] @ R)[:, s: e]
            new_x = rotated_x @ R.T
            x[0, pos_to_patch] = new_x
            return x
        # run subspace patching
        corp_logits = hooked_model.run_with_hooks(
            torch.tensor([orig_input_ids], dtype=torch.long),
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_hook)]
        )
        corp_logit_diff = corp_logits[0, -2, io_name_id] - corp_logits[0, -2, s_name_id]
        corp_io_prob = F.softmax(corp_logits[0, -2], dim=0)[io_name_id]

        logit_diff_drop.append(orig_logit_diff - corp_logit_diff)
        io_prob_drop.append(orig_io_prob - corp_io_prob)

    should_be_names = set([t.replace("Ġ", "") for t in tokenizer.convert_ids_to_tokens(should_be_names)])
    assert should_be_names.issubset(set(NAMES)), should_be_names

    logit_diff_drop = torch.stack(logit_diff_drop).mean().item()
    io_prob_drop = torch.stack(io_prob_drop).mean().item()
    return logit_diff_drop, io_prob_drop


def evaluate_greater_than(R: torch.FloatTensor, partition: list[int], subspace_idx: int,
                prompts, 
                hooked_model: HookedTransformer, 
                layer: int):
    # modified_info == "YY"
    # pos_lis = [END]
    tokenizer = hooked_model.tokenizer

    partition_edges = list(accumulate(partition, initial=0))
    if subspace_idx is not None:
        s, e = partition_edges[subspace_idx], partition_edges[subspace_idx+1]
    else:
        s, e = partition_edges[0], partition_edges[-1]

    prob_drop = []
    for p in tqdm(prompts):
        orig_input_ids = tokenizer(p["text"])["input_ids"]
        corp_input_ids = tokenizer(p["01text"])["input_ids"]
        targets = tokenizer.convert_tokens_to_ids(p["target"])
        
        orig_input_ids.insert(0, tokenizer.eos_token_id)
        corp_input_ids.insert(0, tokenizer.eos_token_id)
        assert len(orig_input_ids) == len(corp_input_ids), len(orig_input_ids) - len(corp_input_ids)
        
        # clean run
        orig_logits = hooked_model(torch.tensor([orig_input_ids], dtype=torch.long))
        orig_prob = F.softmax(orig_logits[0, -1], dim=0)[targets].sum()

        # save corrupt act
        corp_act = None
        def save_hook(x, hook):
            nonlocal corp_act
            corp_act = x

        hooked_model.run_with_hooks(
            torch.tensor([corp_input_ids], dtype=torch.long), 
            stop_at_layer=layer+1,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", save_hook)])
        
        assert corp_act is not None

        def patch_hook(x, hook):
            rotated_x = x[0, -1:] @ R
            rotated_x[:, s: e] = (corp_act[0, -1:] @ R)[:, s: e]
            new_x = rotated_x @ R.T
            x[0, -1:] = new_x
            return x
        # run subspace patching
        corp_logits = hooked_model.run_with_hooks(
            torch.tensor([orig_input_ids], dtype=torch.long),
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_hook)]
        )
        corp_prob = F.softmax(corp_logits[0, -1], dim=0)[targets].sum()

        prob_drop.append(orig_prob - corp_prob)

    prob_drop = torch.stack(prob_drop).mean().item()
    return prob_drop


def load_R(exp_dir, model_name, layer):
    site_name = site_name_to_short_name(f"blocks.{layer}.hook_resid_post")
    R_path = exp_dir / f"R-{model_name}-{site_name}.pt"
    if R_path.exists():
        print_("build index using R from", R_path)
        R = torch.load(R_path, map_location=device)["R.parametrizations.weight.0.base"]

        with open(exp_dir / f"R_config-{model_name}-{site_name}.json") as f:
            partition = json.load(f)["partition"]

        return R, partition
    else:
        return None, None


def convert_to_baseline_if_necessary(R: torch.FloatTensor, args, hooked_model, layer):
    if args.method == "identity":
        R = torch.eye(R.size(0))
    elif args.method == "random":
        rand, _, _ = torch.linalg.svd(torch.randn_like(R))
        R = rand
    elif args.method.startswith("PCA"):
        acts = get_activations(hooked_model, layer)
        cov = torch.cov(acts.T)
        eig_values, eig_vectors = torch.linalg.eigh(cov)
        print("eigen values", eig_values)
        if args.method == "PCA":
            R = eig_vectors # eig v in ascending order, and partition is in descending order, so smallest subspace get biggest variance
        elif args.method == "PCA2":
            R = eig_vectors.flip([1])
    return R


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(device)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str)
parser.add_argument("--method", type=str, default="trainedR", choices=["trainedR", "identity", "random", "PCA", "PCA2"])
args = parser.parse_args()

assert os.getcwd().endswith("preimage")
exp_dir = Path("..") / "trainedRs" / args.exp_name
assert exp_dir.exists()

log_path = exp_dir / (f"eval_log.txt" if args.method == "trainedR" else f"eval_log_{args.method}.txt")
f = open(log_path, "w")
print_ = partial(print_to_both, f=f)

ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=1000,
    seed = 0,
)
print_("load IOI data (1000 samples)")

def merge_prompts_and_pos(prompts: list[dict[str, str]], word_idx: dict[str, torch.LongTensor]):
    prompts = deepcopy(prompts)
    assert len(prompts) ==  len(word_idx["IO"])
    for i in tqdm(range(len(prompts))):
        for k in word_idx.keys():
            prompts[i][f"pos_{k}"] = word_idx[k][i].item()
    return prompts

    
prompts = merge_prompts_and_pos(ioi_dataset.ioi_prompts, ioi_dataset.word_idx)
print(prompts[0])

hooked_model = HookedTransformer.from_pretrained(
    to_valid_model_name("gpt2"),
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device=device,
)

do_test = [1, 2, 3, 4]
# test 1: ioi, layer 4, subspace patching on IO+1 and S1+1, measuring previous token subspace
print_("******* test 1 *******")
model_name, layer, pos_lis = "gpt2", 4, ["IO+1", "S+1"]
R, partition = load_R(exp_dir, model_name, layer)
if 1 in do_test and R is not None:
    R = convert_to_baseline_if_necessary(R, args, hooked_model, layer)
    subspace_var = measure_subspace_var(R, partition, hooked_model, layer)
    all_LD_drop = []
    for subspace_idx in [None,] + list(range(len(partition))):
        logit_diff_drop, io_prob_drop = evaluate_ioi(R, partition, subspace_idx, prompts, hooked_model, layer, pos_lis, "pos")
        if subspace_idx is not None:
            all_LD_drop.append(logit_diff_drop)
            print_(f"subspace {subspace_idx} (d={partition[subspace_idx]}, var={subspace_var[subspace_idx]:.2f}): logit diff drop {logit_diff_drop:.2f}, io prob drop {io_prob_drop:.3f}")
        else:
            print_(f"whole space (d={R.size(0)}, var={sum(subspace_var):.2f}): logit diff drop {logit_diff_drop:.2f}, io prob drop {io_prob_drop:.3f}")
    draw_figures(partition, subspace_var, all_LD_drop, exp_dir / "test1")

# test 2: ioi, layer 6, subspace patching on S2, measuring induction head output subspace
print_("******* test 2 *******")
model_name, layer, pos_lis = "gpt2", 6, ["S2"]
R, partition = load_R(exp_dir, model_name, layer)
if 2 in do_test and R is not None:
    R = convert_to_baseline_if_necessary(R, args, hooked_model, layer)
    subspace_var = measure_subspace_var(R, partition, hooked_model, layer)
    all_LD_drop = []
    for subspace_idx in [None,] + list(range(len(partition))):
        logit_diff_drop, io_prob_drop = evaluate_ioi(R, partition, subspace_idx, prompts, hooked_model, layer, pos_lis, "pos")
        if subspace_idx is not None:
            all_LD_drop.append(logit_diff_drop)
            print_(f"subspace {subspace_idx} (d={partition[subspace_idx]}, var={subspace_var[subspace_idx]:.2f}): logit diff drop {logit_diff_drop:.2f}, io prob drop {io_prob_drop:.3f}")
        else:
            print_(f"whole space (d={R.size(0)}, var={sum(subspace_var):.2f}): logit diff drop {logit_diff_drop:.2f}, io prob drop {io_prob_drop:.3f}")
    draw_figures(partition, subspace_var, all_LD_drop, exp_dir / "test2")

# test 3: ioi, layer 8, subspace patching on END, measuring S-Inhibition head output subspace
print_("******* test 3 *******")
model_name, layer, pos_lis = "gpt2", 8, ["end"]
R, partition = load_R(exp_dir, model_name, layer)
if 3 in do_test and R is not None:
    R = convert_to_baseline_if_necessary(R, args, hooked_model, layer)
    subspace_var = measure_subspace_var(R, partition, hooked_model, layer)
    print_("test S pos info")
    all_LD_drop = []
    for subspace_idx in [None,] + list(range(len(partition))):
        logit_diff_drop, io_prob_drop = evaluate_ioi(R, partition, subspace_idx, prompts, hooked_model, layer, pos_lis, "pos")
        if subspace_idx is not None:
            all_LD_drop.append(logit_diff_drop)
            print_(f"subspace {subspace_idx} (d={partition[subspace_idx]}, var={subspace_var[subspace_idx]:.2f}): logit diff drop {logit_diff_drop:.2f}, io prob drop {io_prob_drop:.3f}")
        else:
            print_(f"whole space (d={R.size(0)}, var={sum(subspace_var):.2f}): logit diff drop {logit_diff_drop:.2f}, io prob drop {io_prob_drop:.3f}")
    draw_figures(partition, subspace_var, all_LD_drop, exp_dir / "test3-pos")

    print_("test S name info")
    all_LD_drop = []
    for subspace_idx in [None,] + list(range(len(partition))):
        logit_diff_drop, io_prob_drop = evaluate_ioi(R, partition, subspace_idx, prompts, hooked_model, layer, pos_lis, "S_name")
        if subspace_idx is not None:
            all_LD_drop.append(logit_diff_drop)
            print_(f"subspace {subspace_idx} (d={partition[subspace_idx]}, var={subspace_var[subspace_idx]:.2f}): logit diff drop {logit_diff_drop:.2f}, io prob drop {io_prob_drop:.3f}")
        else:
            print_(f"whole space (d={R.size(0)}, var={sum(subspace_var):.2f}): logit diff drop {logit_diff_drop:.2f}, io prob drop {io_prob_drop:.3f}")
    draw_figures(partition, subspace_var, all_LD_drop, exp_dir / "test3-S_name")




prompts = get_prompts_and_more(1000, 0)
print_("load greater than data (1000 samples)")
print(prompts[0])

# test 4: greater than, layer 9, subspace patching on END, measuring YY subspace
print_("******* test 4 *******")
model_name, layer = "gpt2", 9
R, partition = load_R(exp_dir, model_name, layer)
if 4 in do_test and R is not None:
    R = convert_to_baseline_if_necessary(R, args, hooked_model, layer)
    subspace_var = measure_subspace_var(R, partition, hooked_model, layer)
    all_prob_drop = []
    for subspace_idx in [None,] + list(range(len(partition))):
        prob_drop = evaluate_greater_than(R, partition, subspace_idx, prompts, hooked_model, layer)
        if subspace_idx is not None:
            all_prob_drop.append(prob_drop)
            print_(f"subspace {subspace_idx} (d={partition[subspace_idx]}, var={subspace_var[subspace_idx]:.2f}): prob drop {prob_drop:.3f}")
        else:
            print_(f"whole space (d={R.size(0)}, var={sum(subspace_var):.2f}): prob drop {prob_drop:.3f}")
    draw_figures(partition, subspace_var, all_prob_drop, exp_dir / "test4")
    

f.close()