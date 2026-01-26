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

class rotateMatrix(nn.Module):
    def __init__(self, weight):
        super().__init__()
        # nn.init.orthogonal_(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, h):
        return h @ self.weight
    
class matrixHolder(nn.Module):
    def __init__(self, init_R, partition):
        super().__init__()
        assert len(partition) == 2
        self.partition = partition
        self.R = nn.utils.parametrizations.orthogonal(rotateMatrix(init_R))

    def forward(self, h):
        return self.R(h).split(self.partition, dim=-1)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        mid_dim = in_dim * 4
        self.l1 = nn.Linear(in_dim, mid_dim)
        self.l2 = nn.Linear(mid_dim, out_dim)
    
    def forward(self, h):
        return self.l2(F.relu(self.l1(h)))
    
class MLP2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.ReLU(),
            nn.Linear(in_dim * 4, out_dim * 4),
            nn.ReLU(),
            nn.Linear(out_dim * 4, out_dim)
        )
    
    def forward(self, h):
        return self.mlp(h)
    

class ActivationBuffer:
    def __init__(self, cfg, model: HookedTransformer):
        self.cfg = cfg

        self.cache_size = cfg.cache_size
        self.num_repeat = cfg.num_repeat    # expected
        self.normalize = cfg.normalize

        self.acts_list = deque()

        self.cfg = cfg
        self.model = model
        self.buffer_dtype = torch.float16   # may change to float16

        self.load_dataset()

        self.refresh()

        self.std = self.compute_std()

    def load_dataset(self):
        data = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        data = data.shuffle(buffer_size=200_000)
        self.data = map(lambda x:x["text"], data)

    def compute_std(self, num=2000):
        num = min(len(self.acts_list), num)
        acts = []
        for i in range(num):
            acts.append(self.acts_list[i])
        acts = torch.stack(acts).float()
        return acts.std().item()

    def pop_batch(self, batch_size):
        if len(self.acts_list) <= self.cache_size * (1 - 1 / self.num_repeat):
            self.refresh()

        output_lis = []
        recyle_lis = []
        for i in range(batch_size):
            a = self.acts_list.popleft()
            output_lis.append(a)
            if random.random() < (1 - 1 / self.num_repeat):
                recyle_lis.append(a)
        
        self.acts_list.extend(recyle_lis)
        out = torch.stack(output_lis).float()
        if self.normalize:
            out /= self.std
        return out

    def token_batch(self):
        try:
            tokens = []
            while True:
                input_ids = self.model.tokenizer(next(self.data))["input_ids"]
                tokens.extend(input_ids)
                if len(tokens) >= self.cfg.caching_batch_size * self.model.cfg.n_ctx:
                    break
                tokens.append(self.model.tokenizer.eos_token_id)
            tokens = tokens[:self.cfg.caching_batch_size * self.model.cfg.n_ctx]
            return tokens
        
        except StopIteration:
            print("End of data stream reached")
            self.load_dataset()
            return self.token_batch()

    @torch.no_grad()
    def refresh(self):
        gc.collect()
        torch.cuda.empty_cache()

        self.model.reset_hooks()
        cache = self.model.add_caching_hooks(self.cfg.act_site)
        stop_at_layer = int(re.search(r"blocks\.(\d+)\.", self.cfg.act_site).group(1)) + 1

        pbar = tqdm(total=self.cache_size, initial=len(self.acts_list), desc="Refreshing activations", disable=False)
        while len(self.acts_list) < self.cache_size:
            input_batch = self.token_batch()
            input_batch = torch.tensor(input_batch, device=self.cfg.device, dtype=torch.long).view(self.cfg.caching_batch_size, -1)

            with torch.autocast("cuda"):
                self.model(input_batch, stop_at_layer=stop_at_layer)
                acts = cache[self.cfg.act_site]
                # acts = F.layer_norm(acts, [acts.size(-1)])
                
            acts = acts.flatten(end_dim=1).to(self.buffer_dtype)
            self.acts_list.extend(torch.unbind(acts))
            pbar.update(acts.size(0))

        pbar.close()

        # self.acts_list = deque(random.sample(self.acts_list, len(self.acts_list)))
        # random.shuffle(self.acts_list)     # too slow
        self.acts_list = deque(sorted(self.acts_list, key=lambda _: random.random()))


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
        clip_grad_mlp = 10.0,
        clip_grad_R = 100.0,
        cache_size = 2048 * 2048,
        num_repeat = 3,     # each data point is used for # times
        normalize = False,
        arch = "small",
        lookahead_k = 0,
        part0_dim = 384,
    )
    cfg = arg_parse_update_cfg(default_cfg)
    set_seed(0)

    assert os.getcwd().endswith("training")
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

    partition = sorted([cfg.part0_dim, h_dim-cfg.part0_dim], reverse=True)
    R = matrixHolder(torch.eye(h_dim), partition).to(device)
    R_optimizer = torch.optim.Adam(R.parameters(), lr=cfg.lr_R, betas=(cfg.adam_beta1, cfg.adam_beta2))

    if cfg.arch == "small":
        MLP_class = MLP
    elif cfg.arch == "medium":
        MLP_class = MLP2
    mlp01 = MLP_class(partition[0], partition[1]).to(device)
    mlp10 = MLP_class(partition[1], partition[0]).to(device)
    mlp_optimizer = torch.optim.AdamW(chain(mlp01.parameters(), mlp10.parameters()), lr=cfg.lr_mlp, betas=(cfg.adam_beta1, cfg.adam_beta2))

    loss_func = nn.MSELoss(reduction="none")
    log_metrics = defaultdict(list)

    cfg.block_len = 1024
    cfg.refresh_block_num = 100
    buffer_for_eval = BufferReuse(cfg, hooked_model)
    
    param_snapshot = [p.detach().clone() for p in chain(R.parameters(), mlp01.parameters(), mlp10.parameters())]
    for i in tqdm(range(cfg.outer_steps)):

        if i % 1000 == 0:
            cfg.metric = "euclidean"
            cfg.search_steps = 100
            temp_holder = NewUnevenRTrainer(h_dim, partition, cfg, buffer_for_eval, R.R.weight.data).to(device)
            print_("computing mi")

            mi = 0
            subspace_var = temp_holder.compute_subspace_var(num=2000)
            for j in tqdm(range(200)):
                mi_batch = temp_holder.compute_MI_step(metric="euclidean", subspace_var=subspace_var) 
                mi += mi_batch
            mi /= 200
            print_("MI:\n", mi)

        mlp01.train()
        mlp10.train()
        for j in range(cfg.inner_steps):

            acts = buffer.pop_batch(cfg.batch_size)
            with torch.no_grad():
                part0, part1 = R(acts)
                if j == 0:
                    part0_var = part0.var(dim=0).clamp(min=1e-5)
                    part1_var = part1.var(dim=0).clamp(min=1e-5)
            
            recon_loss1 = loss_func(mlp01(part0), part1)
            recon_loss1 = (recon_loss1.mean(dim=0) / part1_var).sum()
            recon_loss0 = loss_func(mlp10(part1), part0)
            recon_loss0 = (recon_loss0.mean(dim=0) / part0_var).sum()
            loss = recon_loss0 + recon_loss1

            if j == 0:
                log_metrics["unlearned_mlp_part0"].append(recon_loss0.item() / partition[0])
                log_metrics["unlearned_mlp_part1"].append(recon_loss1.item() / partition[1])

            mlp_optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(chain(mlp01.parameters(), mlp10.parameters()), max_norm=float('inf'))
            log_metrics["mlp_grad_norm"].append(grad_norm.item())
            torch.nn.utils.clip_grad_norm_(chain(mlp01.parameters(), mlp10.parameters()), max_norm=cfg.clip_grad_mlp)
            mlp_optimizer.step()

        mlp01.eval()
        mlp10.eval()
        acts = buffer.pop_batch(cfg.batch_size)
        part0, part1 = R(acts)

        recon_loss1 = loss_func(mlp01(part0), part1)
        recon_loss1 = (recon_loss1.mean(dim=0) / part1_var).sum()
        recon_loss0 = loss_func(mlp10(part1), part0)
        recon_loss0 = (recon_loss0.mean(dim=0) / part0_var).sum()
        loss = - (recon_loss0 + recon_loss1)

        log_metrics["learned_mlp_part0"].append(recon_loss0.item() / partition[0])
        log_metrics["learned_mlp_part1"].append(recon_loss1.item() / partition[1])

        R_optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(R.parameters(), max_norm=float('inf'))
        log_metrics["R_grad_norm"].append(grad_norm.item())
        torch.nn.utils.clip_grad_norm_(R.parameters(), max_norm=cfg.clip_grad_R)
        R_optimizer.step()
        
        if (cfg.lookahead_k > 1) and ((i+1) % cfg.lookahead_k == 0):
            all_params = chain(R.parameters(), mlp01.parameters(), mlp10.parameters())
            for p_old, p_new in zip(param_snapshot, all_params):
                p_new.data = p_old + 0.5 * (p_new.data - p_old)
            param_snapshot = [p.detach().clone() for p in all_params]


        if (i+1) % 200 == 0:
            print_( {k: sum(v) / len(v) for k, v in log_metrics.items()} )
            log_metrics = defaultdict(list)
            # print("compare weight and base", torch.allclose(R.R.weight, R.R.parametrizations.weight[0].base), (R.R.weight.data - R.R.parametrizations.weight[0].base).abs().mean())

    print_("training complete")
    cfg.metric = "euclidean"
    cfg.search_steps = 100
    mi = 0
    subspace_var = temp_holder.compute_subspace_var(num=2000)
    for j in tqdm(range(200)):
        mi_batch = temp_holder.compute_MI_step(metric="euclidean", subspace_var=subspace_var) 
        mi += mi_batch
    mi /= 200
    print_("final mi:", mi)