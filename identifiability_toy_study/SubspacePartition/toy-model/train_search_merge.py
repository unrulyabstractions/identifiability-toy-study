from model_train import *
from easydict import EasyDict
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from matplotlib.colors import Normalize
from functools import partial


class rotateMatrix(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, h, dtype=torch.float32):
        return h.to(dtype) @ self.weight.to(dtype)

class RTrainer(nn.Module):
    def __init__(self, partition: list[int], metric="cosine", previous_R=None):
        super().__init__()
        h_dim = sum(partition)
        self.partition = partition
        if previous_R is None:
            previous_R = torch.eye(h_dim)
        else:
            assert previous_R.size(0) == previous_R.size(1) == h_dim
        self.R = nn.utils.parametrizations.orthogonal(rotateMatrix(previous_R))
        assert metric in ["cosine", "euclidean"]
        self.metric = metric

    def forward(self, h, separate_loss=False, search_N=None):
        # h: batch_size, h_dim

        h = self.R(h)

        if self.metric == "cosine":
            raise NotImplementedError
            
        elif self.metric == "euclidean":
            loss = []
            for subspace_h in h.split(self.partition, dim=-1):
                if search_N is not None:
                    num_group = subspace_h.size(0) // search_N
                    subspace_h = subspace_h.view(num_group, search_N, subspace_h.size(-1))
                else:
                    subspace_h = subspace_h.unsqueeze(0)
                with torch.no_grad():
                    dist = torch.cdist(subspace_h, subspace_h)
                    dist.masked_fill_(torch.eye(dist.size(-1), device=dist.device, dtype=torch.bool).unsqueeze(0), torch.finfo(dist.dtype).max)
                    most_similar = dist.argmin(dim=-1, keepdim=True).expand_as(subspace_h)

                most_sim_vec = torch.gather(subspace_h, dim=1, index=most_similar)
                loss.append( torch.linalg.vector_norm(subspace_h - most_sim_vec, dim=-1).mean() )

            loss = torch.stack(loss)
            if separate_loss:
                return loss
            return loss.mean()
        
        else:
            raise NotImplementedError
        
    @torch.no_grad()
    def evaluate(self, h):
        h = self.R(h)
        if self.metric == "cosine":
            pass
        elif self.metric == "euclidean":
            neighbors = []
            for subspace_h in h.split(self.partition, dim=1):
                with torch.no_grad():
                    dist = torch.cdist(subspace_h.unsqueeze(0), subspace_h.unsqueeze(0)).squeeze(0)
                    dist.masked_fill_(torch.eye(dist.size(0), device=dist.device, dtype=torch.bool), torch.finfo(dist.dtype).max)
                    most_similar = dist.argmin(dim=1)

                neighbors.append( subspace_h[most_similar] )

            unexplained_v = ((h - torch.cat(neighbors, dim=1))**2).mean(dim=0)
            return torch.stack([v.sum() for v in unexplained_v.split(self.partition)])

        else:
            raise NotImplementedError
        
    @torch.no_grad()
    def compute_MI_step(self, h, subspace_var=None, pairs=None):
        h = self.R(h)

        all_dist = []
        for i, subspace_h in enumerate(h.split(self.partition, dim=1)):
            dist = torch.cdist(subspace_h.unsqueeze(0), subspace_h.unsqueeze(0)).squeeze(0)
            if subspace_var is not None:
                dist /= subspace_var[i].sqrt()
            dist.masked_fill_(torch.eye(dist.size(0), device=h.device, dtype=torch.bool), torch.finfo(dist.dtype).max)
            all_dist.append(dist)
        all_dist = torch.stack(all_dist)

        if pairs is None:
            pairs = list(combinations(range(len(self.partition)), 2))
        mi = {}
        constant = torch.special.digamma(torch.ones((), device=h.device)) + torch.special.digamma(torch.tensor(h.size(0), device=h.device))
        for j, k in pairs:
            dist_joint = torch.max(all_dist[j], all_dist[k])
            min_d, _ = dist_joint.min(dim=1, keepdim=True)
            n_x = (all_dist[j] < min_d).sum(dim=1)
            n_y = (all_dist[k] < min_d).sum(dim=1)
            mi[(j, k)] = (constant - torch.special.digamma(n_x+1) - torch.special.digamma(n_y+1)).mean()

        return mi
    
    def save(self, path, suffix=""):
        print_("saving..", path)
        path = Path(path)
        if not path.exists():
            path.mkdir()
        
        config = {"partition": self.partition}
        with open(path / f"R_config{suffix}.json", "w") as f:
            json.dump(config, f)
        
        torch.save(self.state_dict(), path / f"R{suffix}.pt")

    @classmethod
    def from_trained(cls, path, suffix=""):
        path = Path(path)
        with open(path / f"R_config{suffix}.json") as f:
            config = json.load(f)
        
        trainer = cls(**config)
        trainer.load_state_dict(torch.load(path / f"R{suffix}.pt", map_location="cpu"))
        print_("model loaded", path)

        return trainer

@torch.no_grad()
def compute_MI(R: RTrainer, model: toyModel, dataset, pairs=None):
    dataloader = DataLoader(dataset, batch_size=128)

    all_h = []
    for i, x in enumerate(dataloader):
        if i == 20:
            break
        h = x @ model.W
        all_h.append(R.R(h))
    all_h = torch.cat(all_h, dim=0)
    subspace_var = torch.stack([v.sum() for v in all_h.var(dim=0).split(R.partition)])

    avg_mi = defaultdict(list)
    for i, x in enumerate(dataloader):
        if i == 100:
            break
        h = x @ model.W
        mi = R.compute_MI_step(h, subspace_var=subspace_var, pairs=pairs)
        for k in mi:
            avg_mi[k].append(mi[k])
    for k in avg_mi:
        avg_mi[k] = torch.stack(avg_mi[k]).mean().item()
    return avg_mi

@torch.no_grad()
def evaluate_per_space(R: RTrainer, model: toyModel, dataset):
    dataloader = DataLoader(dataset, batch_size=1024)

    loss = []
    for i, x in enumerate(dataloader):
        if i == 50:
            break
        h = x @ model.W
        loss.append(R(h, separate_loss=True, search_N=1024))

    loss = torch.stack(loss).mean(dim=0)

    print_("\n", loss, "\n", loss.mean())
    return loss


    
def show_fig(W, label=""):
    height, width = W.size()
    height_inch = 4
    width_inch = max(2, height_inch * width / height)
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    max_abs = W.abs().max().item()
    heatmap = ax.imshow(W.cpu().numpy(), cmap="bwr", aspect="equal", norm=Normalize(vmin=-max_abs, vmax=max_abs))

    fig.colorbar(heatmap, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
    plt.tight_layout()
    
    plt.savefig(output_dir / f"fig_{label}.png", dpi=200)
    plt.close()

if __name__ == "__main__":

    unit_size = 2

    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_device(device)
    
    def print_to_both(*args, f=None):
        print(*args)
        print(*args, file=f, flush=True)

    output_dir = Path("./exp1")
    if not output_dir.exists():
        output_dir.mkdir()
    log_path = output_dir / "train_log.txt"
    f = open(log_path, "w")
    print_ = partial(print_to_both, f=f)

    with open("config.json", "r") as f:
        config = EasyDict(json.load(f))
    config.x_dim = sum(config.n_feature)

    model = toyModel(config.x_dim, config.h_dim)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    show_fig(model.W.data.T, "orig_W")

    n_spaces = config.h_dim // unit_size
    partition = [unit_size] * n_spaces
    if n_spaces * unit_size < config.h_dim:
        partition.append( config.h_dim - n_spaces * unit_size )
    
    R = RTrainer(partition, "euclidean")

    best_loss = 1000
    merge_start = 60_000
    best_R = None
    indices = []
    dataset = toyData(config.n_feature, config.sparsity)
    dataloader = DataLoader(dataset, batch_size=128)  #4
    optimizer = torch.optim.Adam(R.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=3e-4, max_lr=3e-3, step_size_up=1, step_size_down=10_000)
    loss_lis = []
    for i, x in enumerate(dataloader):

        with torch.no_grad():
            h = x @ model.W

        loss = R(h, search_N=4)

        loss_lis.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 1000 == 0:
            print_(sum(loss_lis) / len(loss_lis))
            loss_lis = []
        
        if (i+1) <= merge_start and ((i+1) % 3_000 == 0):

            loss_per_space = evaluate_per_space(R, model, dataset)

            show_fig((model.W.data @ R.R.weight.data).T, "first_3000" if i+1 == 3_000 else "")

            if loss_per_space.mean().item() < best_loss:
                best_loss = loss_per_space.mean().item()
                best_R = R.R.weight.detach().clone()

            max_idx = loss_per_space.argmax().item()
            pairs = [(max_idx, j) for j in range(len(R.partition)) if j != max_idx]
            mi = compute_MI(R, model, dataset, pairs)
            sorted_mi = sorted([(k[1], v) for k, v in mi.items()], key=lambda x: -x[1])
            print_("sorted mi\n", sorted_mi)

            mi_keys = list(mi.keys())
            mi_values = torch.tensor([mi[k] for k in mi_keys]).clamp(min=0)
            loss_values = loss_per_space[[k[1] for k in mi_keys]]
            probs = loss_values * mi_values
            probs = probs / probs.sum()
            sorted_prob = sorted([(k[1], probs[j].item()) for j, k in enumerate(mi_keys)], key=lambda x: -x[1])
            print_("sorted prob\n", sorted_prob)
            selected_idx = torch.multinomial(probs, 1).item()
            indices = mi_keys[selected_idx]

            print_("re init subspace", indices)

            R_chunks = list(R.R.weight.data.split(R.partition, dim=1))
            R_chunk = torch.cat([R_chunks[idx] for idx in indices], dim=1)
            rand, _, _ = torch.linalg.svd(torch.randn(R_chunk.size(1), R_chunk.size(1)))
            new_R_chunks = (R_chunk @ rand).split([R.partition[idx] for idx in indices], dim=1)
            for j, idx in enumerate(indices):
                R_chunks[idx] = new_R_chunks[j]
            new_R = torch.cat(R_chunks, dim=1)

            R = RTrainer(partition, "euclidean", previous_R=new_R)
            assert torch.allclose(R.R.weight.data, new_R), (R.R.weight.data - new_R).abs().mean().item()
            optimizer = torch.optim.Adam(R.parameters(), lr=1e-3)

        if (i+1) > merge_start and ((i+1) % 3_000 == 0):

            if best_R is not None:
                R = RTrainer(partition, "euclidean", previous_R=best_R)
                best_R = None

                show_fig((model.W.data @ R.R.weight.data).T, "before_merge")

            mi = compute_MI(R, model, dataset)
            metric = {}
            for j, k in mi:
                metric[(j,k)] = mi[(j,k)] / (R.partition[j] + R.partition[k]) #/ max(entropy[j].item(), entropy[k].item(), 1e-1)
            
            lis = sorted([(k, v) for k, v in metric.items()], key=lambda x: -x[1])
            print_("sorted normed mi", lis)
            thr = 0.04

            covered = set()
            pairs_to_merge = []
            for k, v in lis:
                if v > thr and k[0] not in covered and k[1] not in covered:
                    pairs_to_merge.append(k)
                    covered.add(k[0])
                    covered.add(k[1])
            pairs_to_merge = pairs_to_merge[:max(1, len(partition) // 8)]
            
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

                R = RTrainer(new_partition, "euclidean", previous_R=new_R)
                assert torch.allclose(R.R.weight.data, new_R), (R.R.weight.data - new_R).abs().mean().item()
                optimizer = torch.optim.Adam(R.parameters(), lr=1e-3)

                show_fig((model.W.data @ R.R.weight.data).T)

                print_(f"******* after merging ({thr}):", clusters_sizes)
            else:
                break

    show_fig((model.W.data @ R.R.weight.data).T, "final")

    f.close()