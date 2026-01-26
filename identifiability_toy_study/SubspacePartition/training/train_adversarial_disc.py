import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from datasets import load_dataset
from collections import deque, defaultdict
import random
import gc
import re
from tqdm import tqdm
from utils import *
from pathlib import Path
import json
from functools import partial
from itertools import chain
from model import NewUnevenRTrainer
from data import BufferReuse
import time
import math
from train_adversarial_recon import ActivationBuffer, matrixHolder
    
class Discriminator(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.mlp0 = nn.Sequential(
            nn.Linear(dim0, dim0*4),
            nn.ReLU(),
            nn.Linear(dim0*4, dim0),
            nn.ReLU(),
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(dim1, dim1*4),
            nn.ReLU(),
            nn.Linear(dim1*4, dim1),
            nn.ReLU(),
        )
        self.bl = nn.Bilinear(dim0, dim1, 1, bias=False)
    
    def forward(self, h0, h1):
        h0 = self.mlp0(h0)
        h1 = self.mlp1(h1)
        out = self.bl(h0, h1).squeeze(-1)
        return out
    
# class Discriminator(nn.Module):
#     def __init__(self, dim0, dim1):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(dim0+dim1, (dim0+dim1)*4),
#             nn.ReLU(),
#             nn.Linear((dim0+dim1)*4, 1, bias=False),
#         )
    
#     def forward(self, h0, h1):
#         out = self.mlp(torch.cat([h0, h1], dim=-1)).squeeze(-1)
#         return out


if __name__ == "__main__":

    torch.set_printoptions(sci_mode=False, precision=5)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    default_cfg = dict(
        exp_name="adversarial_exp",
        batch_size=512,  
        acc_steps=1,
        inner_steps=10,
        outer_steps=20_000,   # for training R
        size_search_steps=200,
        unit_size=32,
        model_name = "gpt2",
        lr_mlp = 3e-4,
        lr_R = 3e-4,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        clip_grad_mlp = 100.0,
        clip_grad_R = 100.0,
        cache_size = 2048 * 2048,
        num_repeat = 3,     # each data point is used for # times
        obj_type = "GAN",   # GAN, MINE, LD (logit diff)
        normalize = False,
        lookahead_k = 0,
    )
    cfg = arg_parse_update_cfg(default_cfg)
    set_seed(0)

    assert os.getcwd().endswith("noSAE")
    output_dir: Path = Path("../trainedRs") / cfg.exp_name
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    config_path = output_dir / "training_args.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f)

    cfg.caching_batch_size=16
    cfg.device = device

    hooked_model = HookedTransformer.from_pretrained(
        to_valid_model_name(cfg.model_name),
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        device=device,
    )

    h_dim = hooked_model.cfg.d_model

    act_site = "blocks.8.hook_resid_post"

    print("training for", act_site)
    cfg.act_site = act_site
    site_name = site_name_to_short_name(act_site)

    log_path = output_dir / f"train_log-{cfg.model_name}-{site_name}.txt"
    f = open(log_path, "w")
    print_ = partial(print_to_both, f=f)

    buffer = ActivationBuffer(cfg, hooked_model)

    partition = [h_dim//2, h_dim//2]
    R = matrixHolder(torch.eye(h_dim), partition).to(device)
    R_optimizer = torch.optim.Adam(R.parameters(), lr=cfg.lr_R, betas=(cfg.adam_beta1, cfg.adam_beta2))

    disc = Discriminator(partition[0], partition[1]).to(device)
    disc_optimizer = torch.optim.AdamW(disc.parameters(), lr=cfg.lr_mlp, betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=0.01)

    log_metrics = defaultdict(list)

    cfg.block_len = 1024
    cfg.refresh_block_num = 100
    buffer_for_eval = BufferReuse(cfg, hooked_model)
    
    def derangement(lst):
        while True:
            perm = random.sample(lst, len(lst))
            if all(x != y for x, y in zip(lst, perm)):
                return perm

    param_snapshot = [p.detach().clone() for p in chain(R.parameters(), disc.parameters())]

    for i in tqdm(range(cfg.outer_steps)):

        if (i+1) % 1000 == 0:
            cfg.metric = "euclidean"
            temp_holder = NewUnevenRTrainer(h_dim, partition, cfg, buffer_for_eval, R.R.weight.data).to(device)
            print_("computing mi")

            mi = 0
            subspace_var = temp_holder.compute_subspace_var(num=2000)
            for j in tqdm(range(200)):
                mi_batch = temp_holder.compute_MI_step(metric="euclidean", subspace_var=subspace_var) 
                mi += mi_batch
            mi /= 200
            print_("MI:\n", mi)

        for j in range(cfg.inner_steps):

            acts = buffer.pop_batch(cfg.batch_size)
            with torch.no_grad():
                part0, part1 = R(acts)
            
            num_pos = acts.size(0) // 2
            assert acts.size(0) == num_pos * 2
            indices = list(range(num_pos)) + list(range(num_pos))
            shuffled_indices = list(range(num_pos*2))

            part0, part1 = part0[indices], part1[shuffled_indices]
            logits = disc(part0, part1)

            if cfg.obj_type == "GAN":
                labels = torch.cat([torch.ones(num_pos, device=device, dtype=torch.float),
                                    -torch.ones(acts.size(0)-num_pos, device=device, dtype=torch.float)], dim=0)
                loss = -F.logsigmoid(logits * labels).mean()
                if j == 0:
                    log_metrics["unlearned_pos_prob"].append(F.sigmoid(logits[:num_pos]).mean().item())
                    log_metrics["unlearned_neg_prob"].append(F.sigmoid(logits[num_pos:]).mean().item())
            elif cfg.obj_type == "MINE":
                term1 = logits[:num_pos].mean()
                term2 = torch.logsumexp(logits[num_pos:], dim=0) - math.log(acts.size(0)-num_pos)
                loss = -(term1 - term2)
                if j == 0:
                    log_metrics["unlearned_mi"].append((term1-term2).item())
            elif cfg.obj_type == "LD":
                term1 = logits[:num_pos].mean()
                term2 = logits[num_pos:].mean()
                loss = -(term1 - term2).clamp(max=100)
                if j == 0:
                    log_metrics["unlearned_pos_logit"].append(term1.item())
                    log_metrics["unlearned_neg_logit"].append(term2.item())


            disc_optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=float('inf'))
            log_metrics["mlp_grad_norm"].append(grad_norm.item())
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=cfg.clip_grad_mlp)
            disc_optimizer.step()

            if (j+1) % 100 == 0:
                print_((term1-term2).item())

        acts = buffer.pop_batch(cfg.batch_size)
        part0, part1 = R(acts)

        num_pos = acts.size(0) // 2
        assert acts.size(0) == num_pos * 2
        indices = list(range(num_pos)) + list(range(num_pos))
        shuffled_indices = list(range(num_pos*2))

        part0, part1 = part0[indices], part1[shuffled_indices]
        logits = disc(part0, part1)

        if cfg.obj_type == "GAN":
            labels = torch.cat([-torch.ones(num_pos, device=device, dtype=torch.float),
                                torch.ones(acts.size(0)-num_pos, device=device, dtype=torch.float)], dim=0)
            loss = -F.logsigmoid(logits * labels).mean()

            log_metrics["learned_pos_prob"].append(F.sigmoid(logits[:num_pos]).mean().item())
            log_metrics["learned_neg_prob"].append(F.sigmoid(logits[num_pos:]).mean().item())
        elif cfg.obj_type == "MINE":
            term1 = logits[:num_pos].mean()
            term2 = torch.logsumexp(logits[num_pos:], dim=0) - math.log(acts.size(0)-num_pos)
            loss = term1 - term2
            log_metrics["learned_mi"].append((term1-term2).item())
        elif cfg.obj_type == "LD":
            term1 = logits[:num_pos].mean()
            term2 = logits[num_pos:].mean()
            loss = (term1 - term2)**2
            log_metrics["learned_pos_logit"].append(term1.item())
            log_metrics["learned_neg_logit"].append(term2.item())


        R_optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(R.parameters(), max_norm=float('inf'))
        log_metrics["R_grad_norm"].append(grad_norm.item())
        torch.nn.utils.clip_grad_norm_(R.parameters(), max_norm=cfg.clip_grad_R)
        R_optimizer.step()

        if (cfg.lookahead_k > 1) and ((i+1) % cfg.lookahead_k == 0):
            all_params = chain(R.parameters(), disc.parameters())
            for p_old, p_new in zip(param_snapshot, all_params):
                p_new.data = p_old + 0.5 * (p_new.data - p_old)
            param_snapshot = [p.detach().clone() for p in all_params]

        if (i+1) % 200 == 0:
            print_( {k: sum(v) / len(v) for k, v in log_metrics.items()} )
            log_metrics = defaultdict(list)