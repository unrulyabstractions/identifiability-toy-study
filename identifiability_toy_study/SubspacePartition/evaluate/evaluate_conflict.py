import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
from itertools import product
import math

MADEUP_NAMES = [
    "Jake Milo",
    "Ella Varn",
    "Liam Dace",
    "Nina Korr",
    "Owen Jall",
    "Zoe Trin",
    "Eli Roven",
    "Amy Pell",
    "Noah Vesk",
    "Lila Dorn",
    "Max Taren",
    "Mira Solt",
    "Ben Carteron",
    "Lena Whitlow",
    "Travis Kenley",
    "Nora Ellman",
    "Caleb Drayton",
    "Maya Vickers",
    "Owen Halberg",
    "Tessa Donley",
    "Jared Minton",
    "Erin Walcott",
    "Lars V. Nygren",
    "Emil Kovarik",
    "Giulia R. Bastianelli",
    "Oskar Feldmann",
    "Antoine M. Beaulac",
    "Katya Vedenina",
    "Ryohei K. Matsuda",
    "Nikhil Venkataram",
    "Sung-Ho J. Min",
    "Weilun Zhang",
    "Nurul Syafiqah",
    "Davaadorj B. Enkhjin",
    "Kofi Mensah",
    "Thandiwe Mahlangu",
    "Adewale T. Okonjo",
    "Samira El-Khatib",
    "Lemlem Tsegaye",
    "Joseph M. Kabwe",
    "Farid Nazari",
    "Leila H. Darwish",
    "Emre Tanrikulu",
    "Ayan S. Mukhamedov",
    "Rafael G. Montes",
    "Camila Rojas",
    "Thiago Nascimento",
    "Lucía C. Villanueva",
    "Tama Raukura",
    "Jacob N. McAllister",
    "Hemi Te Paea",
    "Nathaniel Halemai",
    "Alejandro Sorianez",
    "Mireille Dufournet",
    "Takeshi Morikawara",
    "Anastasiya Vedenitskaya",
    "Abdulrahman El-Mazrouei",
    "Chandrakant Devendranath",
    "Ekaterina Miroshenkova",
    "Hirotaka Nishimuraya",
    "Fatimah Zahrah Baqri",
    "Juan Sebastián Carreñosa",
    "Sibusiso Makhumeni",
    "Vellupillai Ganapathi Subramanian",
    "Nandakumar Haricharan Iyer",
    "Siddananda Raghunatha Pillai",
    "Sir Anantaswamy Venugopal Sharma",
    "Thiruvalluvan Krishnamurthy",
    "Madhavendra Narayana Das",
    "Vishwanatha Keshava Rao",
    "Ramasubramaniam Chidambaram Moorthy",
]

MADEUP_OBJECTS = [
    "glinterhorn", "frindlebox", "vaskit", "plumbark",
    "zorfin", "krelbin", "snoffet", "brindlewatt",
    "narpin", "trombeck", "quassel", "gropple",
    "zindlecone", "wharkle", "trasket", "flarnish",
    "drumbletart", "clatterhook", "yindlebeast", "snorbelvine",
    "plixnet", "muffrake", "twaggle", "flarnic root",
    "zorbettle", "quintbucket", "grebblin", "hexaddle",
    "skarnuff", "blinterseed", "cragglehorn", "jellithorpe",
    "murquix", "brollisk", "thrindle",
    "snarp", "grib", "plent", "zawk", "drux",
    "nibber", "franx", "twud", "glintz", "vrop",
    "gravel harp", "plume socket", "crimson flitch",
    "nubble vase", "wicker fang", "snozzle trap",
    "kettle braid", "mothen crank", "bramble wick", "luster pinch",
    "zelt", "frin", "snol", "plax", "grop", "dorv", "misk", "vlat", "crun",
]

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class InterventionDataset(Dataset):
    def __init__(self, prompts):
        super().__init__()
        self.prompts = prompts

    def __getitem__(self, index):
        return self.prompts[index]
    
    def __len__(self):
        return len(self.prompts)

def get_datasets(ravel_path, tokenizer, model: HookedTransformer, entity_type="nobel_prize_winner", attribute="Field"):
    assert (entity_type, attribute) in [("nobel_prize_winner", "Field"), ("physical_object", "Color")]
    print_("making dataset for", (entity_type, attribute))
    ravel_path = Path(ravel_path)
    with open(ravel_path / f"ravel_{entity_type}_entity_attributes.json") as f:
        entity_attr = json.load(f)
    
    if entity_type == "nobel_prize_winner":
        template = " %s won the Nobel Prize in %s. A. Michael Spence won the Nobel Prize in Economics. %s won the Nobel Prize in"
        test_template = "A. Michael Spence won the Nobel Prize in Economics. %s won the Nobel Prize in"
        to_remove = ["A. Michael Spence"]
    elif entity_type == "physical_object":
        template = "The color of %s is usually %s. The color of apple is usually red. The color of %s is usually"
        test_template = "The color of apple is usually red. The color of %s is usually"
        to_remove = ["apple"]

    for entity in to_remove:
        if entity in entity_attr:
            del entity_attr[entity]

    # filter known entity
    prompts = []
    for entity in entity_attr:
        target = " " + entity_attr[entity][attribute]
        text = test_template%entity + target

        prompts.append({"text": text, "target": target, "entity": entity})

    entity_set = set(entity_attr.keys())
    print_("before filtering", len(entity_set))
    batch_size = 8
    for i in tqdm(range(0, len(prompts), batch_size)):
        input_ids = tokenizer([p["text"] for p in prompts[i: i+batch_size]], return_tensors="pt", padding=True)["input_ids"]
        input_lengths = (input_ids.size(1) - (input_ids == tokenizer.pad_token_id).sum(dim=1)).tolist()
        targets = tokenizer([p["target"] for p in prompts[i: i+batch_size]], add_special_tokens=False)["input_ids"]
        target_lengths = [len(t) for t in targets]

        if i == 0:
            print_(tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))
            print_(tokenizer.convert_ids_to_tokens(targets[0]))

        logits = model(input_ids)
        pred = logits.argmax(dim=-1)

        for j in range(pred.size(0)):
            correct = (pred[j, input_lengths[j]-target_lengths[j]-1:input_lengths[j]-1] == input_ids[j, input_lengths[j]-target_lengths[j]:input_lengths[j]]).all().item()
            if not correct or target_lengths[j] > 1:
                if target_lengths[j] > 1:
                    print_("remove", prompts[i+j]["entity"], prompts[i+j]["target"], tokenizer.convert_ids_to_tokens(targets[j]))
                try:
                    entity_set.remove(prompts[i+j]["entity"])
                except:
                    pass
    print_("after filtering", len(entity_set))

    entity_set = list(entity_set)
    # make length - name mapping
    if entity_type == "nobel_prize_winner":
        madeups = MADEUP_NAMES
    elif entity_type == "physical_object":
        madeups = MADEUP_OBJECTS
    
    len_to_name = defaultdict(list)
    name_lengths = [len(n) for n in tokenizer([" " + n for n in madeups], add_special_tokens=False)["input_ids"]]
    for l, n in zip(name_lengths, madeups):
        len_to_name[l].append(n)
    print_("len to names", len_to_name)

    name_to_len = {}
    name_lengths = [len(n) for n in tokenizer([" " + n for n in entity_set], add_special_tokens=False)["input_ids"]]
    for l, n in zip(name_lengths, entity_set):
        name_to_len[n] = l


    # construct dataset
    prompts = []
    for entity in entity_set:
        attr = entity_attr[entity][attribute]
        while (wrong := entity_attr[random.choice(entity_set)][attribute]) == attr:
            pass
        try:
            random_name = random.choice(len_to_name[name_to_len[entity]])
        except:
            print(entity, name_to_len[entity])
            exit()

        param_target = " " + attr
        context_target = " " + wrong
        clean_prompt = template%(entity, wrong, entity)
        corrupt_param_prompt = template%(random_name, wrong, random_name)
        corrupt_context_prompt = template%(random_name, wrong, entity)

        param_target, context_target = tokenizer([param_target, context_target], add_special_tokens=False)["input_ids"]
        clean_prompt, corrupt_param_prompt, corrupt_context_prompt = tokenizer([clean_prompt, corrupt_param_prompt, corrupt_context_prompt])["input_ids"]
        assert len(param_target) == len(context_target) == 1
        assert len(clean_prompt) == len(corrupt_param_prompt) == len(corrupt_context_prompt)

        for j, (c1, c2) in enumerate(zip(clean_prompt[::-1], corrupt_param_prompt[::-1])):
            if c1 != c2:
                break
        entity_pos2 = len(clean_prompt) - (j+1)

        for j, (c1, c2) in enumerate(zip(clean_prompt[::-1], corrupt_context_prompt[::-1])):
            if c1 != c2:
                break
        entity_pos1 = len(clean_prompt) - (j+1)

        prompt = {
            "param_target": param_target[0],
            "context_target": context_target[0],
            "clean_prompt": clean_prompt,
            "corrupt_param_prompt": corrupt_param_prompt,
            "corrupt_context_prompt": corrupt_context_prompt,
            "entity_pos1": entity_pos1,
            "entity_pos2": entity_pos2,
            "last_pos": len(clean_prompt)-1,
        }
        prompts.append(prompt)
    
    datasets = InterventionDataset(prompts)

    return datasets

def load_R(exp_dir, model_name, layer):
    site_name = site_name_to_short_name(f"blocks.{layer[0]}.hook_resid_{layer[1]}")
    R_path = exp_dir / f"R-{model_name}-{site_name}.pt"
    if R_path.exists():
        print("build index using R from", R_path)
        R = torch.load(R_path, map_location=device)["R.parametrizations.weight.0.base"]

        with open(exp_dir / f"R_config-{model_name}-{site_name}.json") as f:
            partition = json.load(f)["partition"]

        return R, partition
    else:
        return None, None

def get_activations(model: HookedTransformer, layer: tuple[int,str]):
    data = load_dataset("JeanKaddour/minipile", split="train", streaming=True, trust_remote_code=True)    # may be changed if not gpt2
    data = data.shuffle(buffer_size=100_000)
    data = map(lambda x:x["text"], data)
    model.reset_hooks()
    layer, sublayer = layer
    cache = model.add_caching_hooks(f"blocks.{layer}.hook_resid_{sublayer}")

    caching_batch_size = 4
    all_acts = []
    for i in range(8):
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

        acts = cache[f"blocks.{layer}.hook_resid_{sublayer}"].flatten(end_dim=1)
        all_acts.append(acts)
    all_acts = torch.cat(all_acts, dim=0)

    model.reset_hooks()

    return all_acts

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

def collate(examples, pad_token_id):
    output = defaultdict(list)
    for example in examples:
        for k, v in example.items():
            output[k].append(v)
    
    for k in output:
        if type(output[k][0]) == list:
            max_len = max(len(l) for l in output[k])
            output[k] = torch.tensor([l + [pad_token_id] * (max_len-len(l)) for l in output[k]], dtype=torch.long)
            
        else:
            output[k] = torch.tensor(output[k], dtype=torch.long)
    
    return output

# examples = []
def evaluate_orig(data_loader, hooked_model: HookedTransformer):
    # original model prediction
    context_rate = 0
    param_rate = 0

    for i, batch in enumerate(tqdm(data_loader)):
        logits = hooked_model(batch["clean_prompt"])
        pred = logits.argmax(dim=-1)
        arange_idx = torch.arange(pred.size(0), dtype=torch.long)
        pred = pred[arange_idx, batch["last_pos"]]
        
        context_rate += (pred == batch["context_target"]).sum().item()
        param_rate += (pred == batch["param_target"]).sum().item()

    context_rate /= len(data_loader.dataset)
    param_rate /= len(data_loader.dataset)

    return context_rate, param_rate

def evaluate(R: torch.FloatTensor, partition: list[int], subspace_idx: int,
                data_loader, 
                hooked_model: HookedTransformer, 
                layer: tuple[int, str], corrupt_type: str, patch_pos: str):

    assert corrupt_type in ["param", "context"]
    assert patch_pos in ["entity_pos1", "entity_pos2", "all_entity_pos", "last_pos", "all"]
    tokenizer = hooked_model.tokenizer
    act_site = f"blocks.{layer[0]}.hook_resid_{layer[1]}"
    layer = layer[0]

    partition_edges = list(accumulate(partition, initial=0))
    if subspace_idx is not None:
        s, e = partition_edges[subspace_idx], partition_edges[subspace_idx+1]
    else:
        s, e = partition_edges[0], partition_edges[-1]

    context_rate = 0
    param_rate = 0
    for i, batch in enumerate(tqdm(data_loader)):

        # save corrupt act
        corp_act = None
        def save_hook(x, hook):
            nonlocal corp_act
            if patch_pos == "all":
                corp_act = x.clone()
            elif patch_pos == "all_entity_pos":
                idx = torch.arange(x.size(0), dtype=torch.long)
                corp_act = (x[idx, batch["entity_pos1"]].clone(), x[idx, batch["entity_pos2"]].clone())
            else:
                idx = torch.arange(x.size(0), dtype=torch.long)
                corp_act = x[idx, batch[patch_pos]].clone()

        hooked_model.run_with_hooks(
            batch[f"corrupt_{corrupt_type}_prompt"],
            stop_at_layer=layer+1,
            fwd_hooks=[(act_site, save_hook)])
        
        assert corp_act is not None

        def patch_hook(x, hook):
            if patch_pos == "all":
                rotated_x = x @ R.unsqueeze(0)
                rotated_x[:, :, s: e] = (corp_act @ R.unsqueeze(0))[:, :, s: e]
                new_x = rotated_x @ R.T.unsqueeze(0)
                x = new_x
            elif patch_pos == "all_entity_pos":
                idx = torch.arange(x.size(0), dtype=torch.long)
                rotated_x = x[idx, batch["entity_pos1"]] @ R
                rotated_x[:, s: e] = (corp_act[0] @ R)[:, s: e]
                new_x = rotated_x @ R.T
                x[idx, batch["entity_pos1"]] = new_x

                rotated_x = x[idx, batch["entity_pos2"]] @ R
                rotated_x[:, s: e] = (corp_act[1] @ R)[:, s: e]
                new_x = rotated_x @ R.T
                x[idx, batch["entity_pos2"]] = new_x

            else:
                idx = torch.arange(x.size(0), dtype=torch.long)
                rotated_x = x[idx, batch[patch_pos]] @ R
                rotated_x[:, s: e] = (corp_act @ R)[:, s: e]
                new_x = rotated_x @ R.T
                x[idx, batch[patch_pos]] = new_x
            return x
        # run subspace patching
        corp_logits = hooked_model.run_with_hooks(
            batch["clean_prompt"],
            fwd_hooks=[(act_site, patch_hook)]
        )
        pred = corp_logits.argmax(dim=-1)
        arange_idx = torch.arange(pred.size(0), dtype=torch.long)
        pred = pred[arange_idx, batch["last_pos"]]
        
        context_rate += (pred == batch["context_target"]).sum().item()
        param_rate += (pred == batch["param_target"]).sum().item()


    context_rate /= len(data_loader.dataset)
    param_rate /= len(data_loader.dataset)
    return context_rate, param_rate


if __name__ == "__main__":
    set_seed(0)
    torch.set_grad_enabled(False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_device(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--entity_type", type=str, default="nobel_prize_winner")
    parser.add_argument("--attribute", type=str, default="Field")
    parser.add_argument("--method", type=str, default="trainedR", choices=["trainedR", "identity", "random", "PCA", "PCA2"])
    args = parser.parse_args()

    assert os.getcwd().endswith("preimage")
    exp_dir = Path("..") / "trainedRs" / args.exp_name
    assert exp_dir.exists()

    log_path = exp_dir / (f"eval_log_{args.entity_type}-{args.attribute}.txt" if args.method == "trainedR" else f"eval_log_{args.entity_type}-{args.attribute}_{args.method}.txt")
    f = open(log_path, "w")
    print_ = partial(print_to_both, f=f)
    # print_ = print

    model_name = args.exp_name.split("-")[0]
    hooked_model = HookedTransformer.from_pretrained(
        to_valid_model_name(model_name),
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        device=device,
    )

    dataset = get_datasets("../dataset/ravel_data", hooked_model.tokenizer, hooked_model, entity_type=args.entity_type, attribute=args.attribute)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=partial(collate, pad_token_id=hooked_model.tokenizer.pad_token_id))

    context_rate, param_rate = evaluate_orig(data_loader, hooked_model)
    print_(f"original model behavior \tcontext rate: {context_rate:.3f} \tparam rate: {param_rate:.3f}")

    for layer in [9, 11]:
        layer = (layer, "mid")
        patch_pos = "all"  
        R, partition = load_R(exp_dir, model_name, layer)
        if R is None:
            continue
        R = convert_to_baseline_if_necessary(R, args, hooked_model, layer)

        top_context_rate = None
        top_param_rate = None
        for subspace_idx in [None,] + list(range(len(partition))):
            if subspace_idx is not None:
                print_(f"layer {layer} subspace {subspace_idx} (d={partition[subspace_idx]}): ")
            else:
                print_(f"layer {layer} whole space (d={R.size(0)}):")
            context_rate, param_rate = evaluate(R, partition, subspace_idx, data_loader, hooked_model, layer, "param", patch_pos)
            print_(f"corrupt param behavior \tcontext rate: {context_rate:.3f} \tparam rate: {param_rate:.3f}")
            if subspace_idx is not None and (top_context_rate is None or context_rate > top_context_rate[0]):
                top_context_rate = (context_rate, param_rate, subspace_idx)

            context_rate, param_rate = evaluate(R, partition, subspace_idx, data_loader, hooked_model, layer, "context", patch_pos)
            print_(f"corrupt context behavior \tcontext rate: {context_rate:.3f} \tparam rate: {param_rate:.3f}")
            if subspace_idx is not None and (top_param_rate is None or param_rate > top_param_rate[1]):
                top_param_rate = (context_rate, param_rate, subspace_idx)

        print_(f"\ntop context rate: space {top_context_rate[2]}, context rate {top_context_rate[0]:.3f}, param rate {top_context_rate[1]:.3f}")
        print_(f"top param rate: space {top_param_rate[2]}, context rate {top_param_rate[0]:.3f}, param rate {top_param_rate[1]:.3f}\n")