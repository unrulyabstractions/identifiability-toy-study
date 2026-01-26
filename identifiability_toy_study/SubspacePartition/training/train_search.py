from data import *
from model import *
from utils import *
from functools import partial
from matplotlib import pyplot as plt
from collections import defaultdict

torch.set_printoptions(sci_mode=False, precision=5)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

default_cfg = dict(
    exp_name="search6_merge_exp",
    batch_size=128,  # for query
    test_batch_size=128,   # 128*512/block_len when unit_size=4
    acc_steps=1,
    metric="euclidean",
    max_steps=50_000,   # for training R
    merge_interval=3_000,
    merge_start=10_000,
    merge_thr=0.04,
    merge_metric="mi",
    search_steps=25,
    unit_size=32,
    model_name = "gpt2",
    lr = 3e-4,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    weight_type = "none",
    block_len = 16384,
    clip_grad = 100.0,
    data_source = "minipile",    # minipile, openwebtext
    double_q = False,
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

h_dim = hooked_model.cfg.d_model

if cfg.model_name == "gpt2": 
    act_sites = ["blocks.4.hook_resid_post", "blocks.6.hook_resid_post", "blocks.8.hook_resid_post", "blocks.9.hook_resid_post"]
elif cfg.model_name == "qwen2.5":   # layer: 28  d_model: 1536    head: 12
    act_sites = ["blocks.11.hook_resid_mid"]
elif cfg.model_name == "gemma2": # layer: 26  d_model: 2304  head: 8
    act_sites = ["blocks.9.hook_resid_mid"]

for act_site in act_sites:
    print("training for", act_site)
    cfg.act_site = act_site
    site_name = site_name_to_short_name(act_site)

    if (output_dir / f"R-{cfg.model_name}-{site_name}.pt").exists():
        continue
    log_path = output_dir / f"train_log-{cfg.model_name}-{site_name}.txt"
    f = open(log_path, "w")
    print_ = partial(print_to_both, f=f)
    
    if not cfg.double_q:
        buffer = BufferReuse(cfg, hooked_model)
    else:
        buffer = BufferReuseDoubleQueue(cfg, hooked_model)
    R = NewUnevenRTrainer(h_dim, [cfg.unit_size] * (h_dim // cfg.unit_size), cfg, buffer).to(cfg.device)

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

        log_metrics["training_loss"].append(loss.item())

        if (i+1) % 200 == 0:
            print_( {k: sum(v) / len(v) for k, v in log_metrics.items()} )
            log_metrics = defaultdict(list)
      
        if (i+1) >= cfg.merge_start and ((i+1-cfg.merge_start) % cfg.merge_interval == 0) and (i+1) < (cfg.max_steps - 100):

            eval_result = []
            for j in tqdm(range(max(1, 50 * 128 // cfg.test_batch_size))):
                eval_result.append( R.evaluate_step(num_steps=test_search_steps, batch_size=cfg.test_batch_size) )
            eval_result = torch.stack(eval_result).mean(dim=0)
            print_("eval result", eval_result)
            
            pairs = list(combinations(range(len(R.partition)), 2))

            print_("computing merge metric")
            mi = 0
            subspace_var = R.compute_subspace_var(num=2000)

            step = max(1, 100 * 128 // cfg.test_batch_size)
            for j in tqdm(range(step)):
                if cfg.merge_metric == "mi":
                    mi_batch = R.compute_MI_step(metric="euclidean", pairs=pairs, num_steps=mi_search_steps, batch_size=cfg.test_batch_size, subspace_var=subspace_var) 
                mi += mi_batch

            mi /= step
            metric = {}
            for pair_idx, (j, k) in enumerate(pairs):
                metric[(j,k)] = mi[pair_idx].item() / (R.partition[j] + R.partition[k])
            
            lis = sorted([(k, v) for k, v in metric.items()], key=lambda x: -x[1])
            if len(lis) > 300:
                print_("sorted normed mi top 10", lis[:10])
                print_("sorted normed mi last 10", lis[-10:])
            else:
                print_("normed mi", lis)

            covered = set()
            pairs_to_merge = []
            for k, v in lis:
                if v > cfg.merge_thr and k[0] not in covered and k[1] not in covered:
                    pairs_to_merge.append(k)
                    covered.add(k[0])
                    covered.add(k[1])
            pairs_to_merge = pairs_to_merge[:max(1, len(R.partition)//8)]

            if pairs_to_merge:
                """ ********* merge ********* """
                temp = [j for p in pairs_to_merge for j in p]
                clusters = pairs_to_merge.copy()
                for j in range(len(R.partition)):
                    if j not in temp:
                        clusters.append((j,))
                clusters_sizes = []
                for c in clusters:
                    clusters_sizes.append( (c, sum(R.partition[j] for j in c)) )
                clusters_sizes.sort(key=lambda x: -x[1])

                R_chunks = R.R.weight.data.split(R.partition, dim=1)
                new_R = []
                new_partition = []
                for c, s in clusters_sizes:
                    new_R.extend([R_chunks[j] for j in c])
                    new_partition.append(s)
                new_R = torch.cat(new_R, dim=1)

                R = NewUnevenRTrainer(h_dim, new_partition, cfg, buffer, previous_R=new_R).to(cfg.device)
                assert torch.allclose(R.R.weight.data, new_R), (R.R.weight.data - new_R).abs().mean().item()
                optimizer = torch.optim.Adam(R.parameters(), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2))

                print_(f"******* after merging ({cfg.merge_thr}):", clusters_sizes)
            
            else:
                break

    print_(f"finish training ({i+1})")
    R.save(output_dir, suffix=f"-{cfg.model_name}-{site_name}")

    print_(f"evaluating ({test_search_steps} steps)...")
    eval_result = []
    for j in range(max(1, 100 * 128 // cfg.test_batch_size)):
        eval_result.append( R.evaluate_step(num_steps=test_search_steps, batch_size=cfg.test_batch_size) )
    eval_result = torch.stack(eval_result).mean(dim=0)
    print_(f" ******* eval result *******")
    print_("mean (weighted)", (eval_result * torch.tensor(R.partition, device=device)).sum().item() / sum(R.partition))
    print_("mean (unweighted)", eval_result.mean().item())
    print_(eval_result)

    f.close()