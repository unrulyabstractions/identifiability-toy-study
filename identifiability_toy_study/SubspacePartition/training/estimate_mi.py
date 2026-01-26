from data import *
from model import *
from utils import *
from functools import partial
from matplotlib import pyplot as plt
from collections import defaultdict
import argparse


torch.set_printoptions(sci_mode=False, precision=5)
set_seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str)
parser.add_argument("--method", type=str, default="trained", choices=["trained", "random", "identity"])
args = parser.parse_args()

exp_dir = Path("../trainedRs") / args.exp_name
with open(exp_dir / "training_args.json") as f:
    cfg = easydict.EasyDict(json.load(f))

if cfg.model_name == "gpt2": 
    act_sites = ["blocks.4.hook_resid_post", "blocks.6.hook_resid_post", "blocks.8.hook_resid_post", "blocks.9.hook_resid_post"]
elif cfg.model_name == "qwen2.5":   # layer: 28  d_model: 1536    head: 12
    act_sites = ["blocks.11.hook_resid_mid", ]
elif cfg.model_name == "gemma2": # layer: 26  d_model: 2304  head: 8
    act_sites = ["blocks.9.hook_resid_mid"]

cfg.refresh_block_num=2048 * 2048 // cfg.block_len
cfg.caching_batch_size=16 if cfg.model_name != "gemma2" else 2  # because 8192 n_ctx
cfg.device = device
test_search_steps = 200 * 2048 // cfg.block_len
if cfg.unit_size <= 4:
    mi_search_steps = 5 * 2048 // cfg.block_len
else:
    mi_search_steps = 50 * 2048 // cfg.block_len

hooked_model = HookedTransformer.from_pretrained(
    to_valid_model_name(cfg.model_name),
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device=device,
)

log_path = exp_dir / f"mi_log-{args.method}-{cfg.model_name}.txt"
f = open(log_path, "w")
print_ = partial(print_to_both, f=f)

for act_site in act_sites:
    print("estimating for", act_site)
    cfg.act_site = act_site
    site_name = site_name_to_short_name(act_site)
    
    buffer = BufferReuse(cfg, hooked_model)

    suffix = f"-{cfg.model_name}-{site_name}"
    with open(exp_dir / f"R_config{suffix}.json") as f:
        partition = json.load(f)["partition"]
    h_dim = sum(partition)

    if args.method == "trained":
        previous_R = torch.load(exp_dir / f"R{suffix}.pt", map_location="cpu")["R.parametrizations.weight.0.base"]
    elif args.method == "identity":
        previous_R = torch.eye(h_dim)
    elif args.method == "random":
        previous_R, _, _ = torch.linalg.svd(torch.randn(h_dim, h_dim))

    R = NewUnevenRTrainer(h_dim, partition, cfg, buffer, previous_R).to(cfg.device)
    print_("model loaded", exp_dir)

    print_("computing merge metric")
    mi = 0
    subspace_var = R.compute_subspace_var(num=2000)

    step = max(1, 100 * 128 // cfg.test_batch_size)
    for j in tqdm(range(step)):
        if cfg.merge_metric == "mi":
            mi_batch, entropy_batch = R.compute_MI_step(metric="euclidean", num_steps=mi_search_steps, batch_size=cfg.test_batch_size, subspace_var=subspace_var)
        mi += mi_batch

    mi /= step
    print_("******** Estimated MI ********")
    print_(mi.cpu())