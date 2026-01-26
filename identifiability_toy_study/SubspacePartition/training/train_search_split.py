from data import *
from model import *
from utils import *
from functools import partial
from matplotlib import pyplot as plt
from collections import defaultdict


torch.set_printoptions(sci_mode=False, precision=5)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

default_cfg = dict(
    exp_name="search6_split_exp",
    batch_size=128,  # for query
    acc_steps=1,
    metric="euclidean",
    max_steps=10_000,   # for training R
    max_rounds=4,
    stop_mi_thr=0.04,  
    num_split_trial=3,  
    num_random_init=1,
    mi_metric="mi",
    search_steps=5,
    model_name = "gpt2",
    lr = 3e-4,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    weight_type = "none",
    block_len = 16384,
    clip_grad = 100.0,
    symmetric = True,   # 
    data_source = "minipile",    # minipile
)
cfg = arg_parse_update_cfg(default_cfg)
set_seed(0)
assert os.getcwd().endswith("training")
output_dir: Path = Path("../trainedRs") / cfg.exp_name
if not output_dir.exists():
    output_dir.mkdir(parents=True)

config_path = output_dir / "training_args.json"
if config_path.exists():
    with open(config_path, "r") as f:
        old_cfg = json.load(f)
    if cfg != old_cfg:
        print("warning: overwriting old config")
with open(config_path, "w") as f:
    json.dump(cfg, f)

cfg.refresh_block_num=2048 * 2048 // cfg.block_len
cfg.caching_batch_size=16
cfg.device = device
test_search_steps = 200 * 2048 // cfg.block_len
mi_search_steps = 50 * 2048 // cfg.block_len

hooked_model = HookedTransformer.from_pretrained(
    to_valid_model_name(cfg.model_name),
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device=device,
)

h_dim = hooked_model.cfg.d_model

act_sites = ["blocks.4.hook_resid_post", "blocks.6.hook_resid_post", "blocks.8.hook_resid_post", "blocks.9.hook_resid_post"]

for act_site in act_sites:
    print("training for", act_site)
    cfg.act_site = act_site
    site_name = site_name_to_short_name(act_site)

    if (output_dir / f"R-{cfg.model_name}-{site_name}.pt").exists():
        continue
    log_path = output_dir / f"train_log-{cfg.model_name}-{site_name}.txt"
    f = open(log_path, "w")
    print_ = partial(print_to_both, f=f)
    
    buffer = BufferReuse(cfg, hooked_model)

    prev_partition = [h_dim]
    prev_R = torch.eye(h_dim).to(device)
    for round_i in range(cfg.max_rounds):
        
        new_partition = [[] for _ in range(cfg.num_split_trial * cfg.num_random_init)]
        for d in prev_partition:
            if not cfg.symmetric:
                interval = d // 2 // cfg.num_split_trial
                for i in range(cfg.num_split_trial):
                    d0 = d // 2 + i * interval
                    d0 = min(d-1, max(1, d0))
                    d1 = d - d0
                    for j in range(cfg.num_random_init):
                        new_partition[i*cfg.num_random_init + j].append((d0, d1))
            else:
                interval = d // (cfg.num_split_trial + 1)   # symmetric
                for i in range(cfg.num_split_trial):
                    d0 = (i+1) * interval
                    d0 = min(d-1, max(1, d0))
                    d1 = d - d0
                    for j in range(cfg.num_random_init):
                        new_partition[i*cfg.num_random_init + j].append((d0, d1))

        print_("training with new partition:\n", new_partition)
        R = NewSplitRTrainer(h_dim, prev_partition, prev_R, new_partition, cfg, buffer).to(cfg.device)

        optimizer = torch.optim.Adam(R.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2))
        log_metrics = defaultdict(list)

        for i in tqdm(range(cfg.max_steps)):
            
            loss = R.step()

            loss.backward()

            if (i+1) % cfg.acc_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(R.parameters(), max_norm=float('inf'))
                log_metrics["R_grad_norm"].append(grad_norm.item())
                torch.nn.utils.clip_grad_norm_(R.parameters(), max_norm=cfg.clip_grad)
                optimizer.step()
                optimizer.zero_grad()

            log_metrics["training_loss"].append(loss.item() / cfg.num_split_trial / cfg.num_random_init)

            if (i+1) % 200 == 0:
                print_( {k: sum(v) / len(v) for k, v in log_metrics.items()} )
                log_metrics = defaultdict(list)
                if cfg.neighbor_thr > 0:
                    print_("search num", R.search_num)

            if ((i+1) % 1000 == 0) and (i+1) < (cfg.max_steps - 100):
                print_("computing mi")
                subspace_var = R.compute_subspace_var(num=2000)

                mi = 0
                for j in tqdm(range(50)):
                    if cfg.mi_metric == "mi":
                        mi_batch = R.compute_MI_step(metric="euclidean", num_steps=mi_search_steps, subspace_var=subspace_var)
                    mi += mi_batch 
                mi /= 50
                
                print_("mi:\n", mi)

                eval_result = []
                for j in range(50):
                    eval_result.append( R.evaluate_step(num_steps=test_search_steps) )
                eval_result = torch.stack(eval_result).mean(dim=0)
                print_("eval result:\n", eval_result)

        subspace_var = R.compute_subspace_var(num=2000)
        mi = 0
        for j in tqdm(range(200)):
            if cfg.mi_metric == "mi":
                mi_batch = R.compute_MI_step(metric="euclidean", num_steps=mi_search_steps, subspace_var=subspace_var)
            mi += mi_batch  
        mi /= 200
        print_("mi:\n", mi)

        normalized_mi = mi / torch.tensor([prev_partition], dtype=torch.float, device=device)
        print_("normed mi:\n", normalized_mi)
        print_(f"normed mi thr={cfg.stop_mi_thr}")
        for i in range(len(new_partition)):
            for j in range(len(prev_partition)):
                d0, d1 = new_partition[i][j]
                if d0 < 4 or d1 < 4:
                    print_(f"trial {i} space {j} candidate {d0}, {d1} is masked")
                    normalized_mi[i, j] = 1000
        if normalized_mi.min() > cfg.stop_mi_thr:
            print_(f"no good partition (min={normalized_mi.min().item()}), exit..")
            break

        best_split_idx = normalized_mi.argmin(dim=0).tolist()
        new_R = torch.zeros_like(prev_R)
        temp_idx = list(accumulate(prev_partition, initial=0))
        next_prev_partition = []
        for i in range(len(prev_partition)):
            s, e = temp_idx[i], temp_idx[i+1]
            new_R[s:e, s:e] = R.Rs[best_split_idx[i]][i].weight.data
            if normalized_mi[best_split_idx[i], i].item() < cfg.stop_mi_thr:
                d0, d1 = new_partition[best_split_idx[i]][i]
                next_prev_partition.extend([d0, d1])
                print_(f"space {i} (d={prev_partition[i]}) is split into {d0}, {d1} (mi={mi[best_split_idx[i], i].item()}, normed mi={normalized_mi[best_split_idx[i], i].item()})")
            else:
                next_prev_partition.append(prev_partition[i])
                print_(f"space {i} (d={prev_partition[i]}) remains the same (mi={mi[best_split_idx[i], i].item()}, normed mi={normalized_mi[best_split_idx[i], i].item()})")
        
        prev_R = prev_R @ new_R
        prev_partition = next_prev_partition
        print_(f"******* finish round {round_i}, updated partition: ******* \n", prev_partition, "\n")

        # save
        print("saving..", output_dir)
        suffix=f"-{cfg.model_name}-{site_name}"
        with open(output_dir / f"R_config{suffix}.json", "w") as f:
            json.dump({"partition": prev_partition}, f)
        
        obj = {"R.parametrizations.weight.0.base": prev_R}
        torch.save(obj, output_dir / f"R{suffix}.pt")

    print_(f"\n ******* finish training ({cfg.max_steps} x {round_i} (+1)) ******* ")
    print_("final partition", prev_partition)

    print_(f"evaluating ({test_search_steps} steps)...")
    eval_result = []
    for j in range(200):
        eval_result.append( R.evaluate_step_given_partition(prev_R, prev_partition, num_steps=test_search_steps) )
    eval_result = torch.stack(eval_result).mean(dim=0)
    print_(f" ******* eval result *******")
    print_("mean (weighted)", (eval_result * torch.tensor(prev_partition, device=device)).sum().item() / h_dim)
    print_("mean (unweighted)", eval_result.mean().item())
    print_(eval_result)

    f.close()