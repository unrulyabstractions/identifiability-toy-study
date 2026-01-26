from transformer_lens import HookedTransformer
from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset
import torch
from tqdm import tqdm
import gc
import os
import pickle
import argparse
import re
from pathlib import Path
from collections import defaultdict
import gc
from utils import *
import shutil
import random
import numpy as np
import json
from transformers import GPT2LMHeadModel

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--max_in_memory", type=int, default=10_000_000)
parser.add_argument("--model_name", type=str, default="gpt2")
parser.add_argument("--override", action="store_true")
args = parser.parse_args()

caching_batch_size = 32
save_dtype = torch.float16
model_name = args.model_name
set_seed(0)

datasets = []
if args.model_name == "gpt2":
    with open("../dataset/prompts_ioi.json") as f:
        prompts_ioi = json.load(f)
    subset = Dataset.from_dict({"sentence": prompts_ioi[:1000]})
    datasets.append(subset)

    with open("../dataset/prompts_greater_than.json") as f:
        prompts_gt = json.load(f)
    subset = Dataset.from_dict({"sentence": prompts_gt[:1000]})
    datasets.append(subset)

elif args.model_name in ["qwen2.5", "gemma2"]:
    with open("../dataset/ravel_250.json") as f:
        subset = Dataset.from_dict({"sentence": json.load(f)})
    datasets.append(subset)

subset = load_dataset("JeanKaddour/minipile", split="train").shuffle(seed=0).select(range(2000)).rename_column("text", "sentence")
datasets.append(subset)

data = concatenate_datasets(datasets)

model = HookedTransformer.from_pretrained(
    to_valid_model_name(model_name),
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device=device,
)


assert os.getcwd().endswith("preimage")
save_dir = Path("..") / "visualizations" / f"shared_acts-{model_name}"
if save_dir.exists() and args.override:
    shutil.rmtree(save_dir)
save_dir.mkdir(parents=True, exist_ok=False)


temp_dir = save_dir / "temp"
temp_dir.mkdir()


model.reset_hooks()
def name_filter(name):
    if name == "blocks.0.hook_resid_pre":
        return True
    if name.endswith("hook_resid_post"):
        return True
    return False
if args.model_name == "gpt2":
    cache = model.add_caching_hooks(name_filter)
    stop_at_layer = None
elif args.model_name == "qwen2.5":
    cache = model.add_caching_hooks("blocks.11.hook_resid_mid")
    stop_at_layer = 12  # to save some time
elif args.model_name == "gemma2":
    cache = model.add_caching_hooks("blocks.9.hook_resid_mid")
    stop_at_layer = 10  # to save some time


cached_act = defaultdict(list)

cached_input = []
total_count = 0
split_count = 0
batch_sents = []
for sent in tqdm(data, desc="caching activations"):
    batch_sents.append(sent["sentence"])
    if len(batch_sents) == caching_batch_size:
        token_ids = model.tokenizer(batch_sents, return_tensors="pt", padding=True, max_length=model.cfg.n_ctx-1, truncation=True)["input_ids"]
        bos = torch.full((token_ids.size(0), 1), fill_value=model.tokenizer.eos_token_id, dtype=token_ids.dtype)
        token_ids = torch.cat([bos, token_ids], dim=1).to(device)

        with torch.autocast("cuda"):
            model(token_ids, return_type=None, stop_at_layer=stop_at_layer)

        m = token_ids != model.tokenizer.eos_token_id
        m[:, 0] = True  # keep bos
        for act_site in cache:
            cached_act[act_site].append(cache[act_site][m].to(save_dtype))

        def select(seq, mask):
            return [t for t, m_element in zip(seq, mask) if m_element]
        cached_input.extend([model.tokenizer.convert_ids_to_tokens(select(seq, mask)) for seq, mask in zip(token_ids.tolist(), m.tolist())])

        num_in_memory = len(cached_act) * sum(a.size(0) for a in next(iter(cached_act.values())))
        if num_in_memory >= args.max_in_memory:
            print("saving batch to disk...")
            for act_site in cached_act:
                file_path = temp_dir / f"{act_site}-split{split_count}.pt"
                torch.save(torch.cat(cached_act[act_site], dim=0), file_path)
            split_count += 1
            cached_act = defaultdict(list)
            gc.collect()
        
        total_count += m.sum().item()
        batch_sents = []

# save last batch
if next(iter(cached_act.values())):
    for act_site in cached_act:
        file_path = temp_dir / f"{act_site}-split{split_count}.pt"
        torch.save(torch.cat(cached_act[act_site], dim=0), file_path)
    split_count += 1
    del cached_act
    gc.collect()

print("loading and merging cached act...")
print("total count", total_count)
for act_site in tqdm(cache):
    merged_act = torch.zeros((total_count, model.cfg.d_model), dtype=save_dtype, device=device)
    cursor = 0
    for i in range(split_count):
        act = torch.load(temp_dir / f"{act_site}-split{i}.pt")
        assert act.dtype == save_dtype
        merged_act[cursor: cursor+act.size(0)] = act
        cursor += act.size(0)
        os.remove(temp_dir / f"{act_site}-split{i}.pt")
    torch.save(merged_act, save_dir / f"{site_name_to_short_name(act_site)}.pt")    # always use short name in file system
temp_dir.rmdir()

print("saving input...")
with open(os.path.join(save_dir, "str_tokens.pkl"), "wb") as f:
    pickle.dump(cached_input, f, protocol=pickle.HIGHEST_PROTOCOL)

