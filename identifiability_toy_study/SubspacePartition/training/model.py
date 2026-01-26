import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import random
from data import *
from itertools import combinations, permutations, accumulate
import math
from collections import OrderedDict

class rotateMatrix(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.d = weight.size(0)

    def forward(self, h, dtype=torch.float32):
        return h.to(dtype) @ self.weight.to(dtype)


class NewUnevenRTrainer(nn.Module):
    def __init__(self, h_dim, partition: list[int], cfg, buffer: BufferReuse, previous_R=None):
        super().__init__()

        self.h_dim = h_dim
        if previous_R is None:
            previous_R = torch.eye(h_dim)
        self.R = nn.utils.parametrizations.orthogonal(rotateMatrix(previous_R))

        self.partition = partition
        assert sum(partition) == h_dim
        assert sorted(partition, reverse=True) == partition, "sort to increase parallel"
        _partition = OrderedDict()
        for p in partition:
            _partition[p] = _partition.get(p, 0) + 1
        self._partition = list(_partition.items())

        self.metric = cfg.metric
        assert self.metric in ["cosine", "euclidean"]
        self.buffer = buffer
        self.cfg = cfg
        self.num_steps = cfg.search_steps

    def _get_subspace_h(self, h, dtype=torch.float32):
        bz = h.size(0)
        h = self.R(h, dtype)
        temp = [d*num for d, num in self._partition]
        h = list(h.split(temp, dim=1))
        for i in range(len(h)):
            h[i] = h[i].view(bz, self._partition[i][1], self._partition[i][0]).transpose(0, 1)  # num_unit, bz, unit_size

        return h
    
    def step(self):

        query_h = self.buffer.pop_one(self.cfg.batch_size)

        query_h = query_h[:self.cfg.batch_size]
        bz = query_h.size(0)
        device = query_h.device
        search_dtype = torch.float32

        query_h = self._get_subspace_h(query_h, torch.float32)
        num_unit = len(self.partition)

        if self.metric == "cosine":
            with torch.no_grad():

                query_h_normed = [(h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=1e-8)).to(search_dtype) for h in query_h]
                most_sim_value = -torch.ones(num_unit, bz, dtype=search_dtype, device=device)
                most_sim_vec = torch.zeros(num_unit, bz, self.h_dim, dtype=self.buffer.buffer_dtype, device=device)

                for i in range(self.num_steps):
                    key_h = next(self.buffer)
                    subspace_key_h = self._get_subspace_h(key_h, search_dtype)
                    subspace_key_h = [h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=1e-3) for h in subspace_key_h]

                    sim = []
                    for q, k in zip(query_h_normed, subspace_key_h):
                        sim.append(torch.bmm(q, k.transpose(1,2)))
                    sim = torch.cat(sim, dim=0)


                    max_v, max_idx = sim.max(dim=-1)
                    m = max_v > most_sim_value
                    most_sim_value[m] = max_v[m]
                    most_sim_vec[m] = key_h[max_idx[m]]
                

            most_sim_vec = self.R(most_sim_vec.flatten(end_dim=1), torch.float32).view(num_unit, bz, self.h_dim)

            temp = [d*num for d, num in self._partition]
            temp = list(accumulate(temp, initial=0))
        
            sim = []
            for i, chunk in enumerate(most_sim_vec.split([num for _, num in self._partition], dim=0)):
                d, n = self._partition[i]
                chunk = chunk[:, :, temp[i]: temp[i+1]].view(n, bz, n, d).transpose(1, 2)
                m = torch.eye(n, dtype=torch.bool, device=device)
                chunk = chunk[m]

                sim_chunk = F.cosine_similarity(query_h[i], chunk, dim=-1)

                mean_sim_per_space = sim_chunk.mean(dim=1)
                
                sim.append( mean_sim_per_space * d )

            sim = torch.cat(sim, dim=0).sum() / sum(self.partition)
            loss = -sim

        elif self.metric == "euclidean":
            with torch.no_grad():
                smallest_dist = torch.full((num_unit, bz), torch.finfo(search_dtype).max, dtype=search_dtype, device=device)
                most_sim_vec = torch.zeros(num_unit, bz, self.h_dim, dtype=self.buffer.buffer_dtype, device=device)

                query_h_search = [h.to(search_dtype) for h in query_h]
                for i in range(self.num_steps):
                    key_h = next(self.buffer)
                    subspace_key_h = self._get_subspace_h(key_h, search_dtype)

                    dist = []
                    for q, k in zip(query_h_search, subspace_key_h):
                        dist.append(torch.cdist(q, k))
                    dist = torch.cat(dist, dim=0)

                    min_v, min_idx = dist.min(dim=-1)
                    m = min_v < smallest_dist
                    smallest_dist[m] = min_v[m]
                    most_sim_vec[m] = key_h[min_idx[m]]

            most_sim_vec = self.R(most_sim_vec.flatten(end_dim=1), torch.float32).view(num_unit, bz, self.h_dim)

            temp = [d*num for d, num in self._partition]
            temp = list(accumulate(temp, initial=0))

            dist = []
            for i, chunk in enumerate(most_sim_vec.split([num for _, num in self._partition], dim=0)):
                d, n = self._partition[i]
                chunk = chunk[:, :, temp[i]: temp[i+1]].view(n, bz, n, d).transpose(1, 2)
                m = torch.eye(n, dtype=torch.bool, device=device)
                chunk = chunk[m]

                dist_chunk = torch.linalg.vector_norm(query_h[i] - chunk, dim=-1)
                
                mean_dist_per_space = dist_chunk.mean(dim=1)

                dist.append(mean_dist_per_space)

            dist = torch.cat(dist, dim=0).mean()
            loss = dist
            
 
        return loss
    


    @torch.no_grad()
    def evaluate_step(self, num_steps=None, batch_size=None):
        if num_steps is None:
            num_steps = self.num_steps

        if batch_size is None:
            batch_size = self.cfg.batch_size
        query_h = self.buffer.pop_one(batch_size)
        query_h = query_h[:batch_size] # smaller bz when num_unit big
        bz = query_h.size(0)
        device = query_h.device

        query_h = self._get_subspace_h(query_h)
        num_unit = len(self.partition)

        if self.metric == "cosine":
            query_h = [h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=1e-8) for h in query_h]
            most_sim_value = -torch.ones(num_unit, bz, dtype=torch.float, device=device)

            for i in range(num_steps):
                key_h = next(self.buffer)
                key_h = self._get_subspace_h(key_h)
                key_h = [h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=1e-3) for h in key_h]

                sim = []
                for q, k in zip(query_h, key_h):
                    sim.append(torch.bmm(q, k.transpose(1,2)))
                sim = torch.cat(sim, dim=0)

                max_v, _ = sim.max(dim=-1)

                most_sim_value = torch.max(most_sim_value, max_v)

            result = most_sim_value.mean(dim=1)
        
        elif self.metric == "euclidean":
            smallest_dist = torch.full((num_unit, bz), torch.finfo(torch.float).max, dtype=torch.float, device=device)
            
            for i in range(num_steps):
                key_h = next(self.buffer)
                key_h = self._get_subspace_h(key_h)

                dist = []
                for q, k in zip(query_h, key_h):
                    dist.append(torch.cdist(q, k))
                dist = torch.cat(dist, dim=0)

                min_v, _ = dist.min(dim=-1)

                smallest_dist = torch.min(smallest_dist, min_v)

            result = smallest_dist.mean(dim=1)

        return result

    @torch.no_grad()
    def compute_subspace_var(self, num=2000):
        self.buffer.cursor = 0
        total_count = 0
        h = []
        while total_count < num:
            h.append(next(self.buffer))
            total_count += h[-1].size(0)
        h = torch.cat(h, dim=0)
        h = self._get_subspace_h(h)

        all_var = []
        for subspace_h in h:
            all_var.append(subspace_h.var(dim=1).sum(dim=-1))
        all_var = torch.cat(all_var, dim=0)
        return all_var
    
    @torch.no_grad()
    def compute_MI_step(self, metric="euclidean", pairs=None, num_steps=None, batch_size=None, subspace_var=None):

        if pairs is None:
            pairs = list(combinations(range(len(self.partition)), 2))
        if num_steps is None:
            max_num_steps = - (self.search_num.max().item() // (-self.buffer.block_len))
            num_steps = min(max_num_steps, self.num_steps)

        if batch_size is None:
            batch_size = self.cfg.batch_size
        query_h = self.buffer.pop_one(batch_size)
        query_h = query_h[:batch_size]
        batch_size = query_h.size(0)
        device = query_h.device

        query_h = self._get_subspace_h(query_h)
        num_unit = len(self.partition)

        assert metric == "euclidean"

        idx1, idx2 = [], []
        for j, k in pairs:
            idx1.append(j)
            idx2.append(k)
        idx1 = torch.tensor(idx1, dtype=torch.long, device=device)
        idx2 = torch.tensor(idx2, dtype=torch.long, device=device)


        smallest_dist = torch.full((len(pairs), batch_size), torch.finfo(torch.float).max, dtype=torch.float, device=device)

        for i in range(num_steps):
            key_h = next(self.buffer)
            key_h = self._get_subspace_h(key_h)

            dist = []
            for q, k in zip(query_h, key_h):
                dist.append(torch.cdist(q, k))
            dist = torch.cat(dist, dim=0)
            if subspace_var is not None:    
                dist /= subspace_var.view(-1, 1, 1).sqrt()

            min_v, _ = torch.max(dist[idx1], dist[idx2]).min(dim=-1)
            smallest_dist = torch.min(smallest_dist, min_v)

        self.buffer.cursor = 0
        n_x = 0
        n_y = 0

        for i in range(num_steps):
            key_h = next(self.buffer)
            key_h = self._get_subspace_h(key_h)

            dist = []
            for q, k in zip(query_h, key_h):
                dist.append(torch.cdist(q, k))
            dist = torch.cat(dist, dim=0)
            if subspace_var is not None:
                dist /= subspace_var.view(-1, 1, 1).sqrt()

            n_x += (dist[idx1] < smallest_dist.unsqueeze(-1)).sum(dim=-1)
            n_y += (dist[idx2] < smallest_dist.unsqueeze(-1)).sum(dim=-1)

        N = num_steps * self.buffer.block_len + 1
        constant = torch.special.digamma(torch.ones((), device=device)) + torch.special.digamma(torch.tensor(N, device=device))
        I = (constant - torch.special.digamma(n_x+1) - torch.special.digamma(n_y+1)).mean(dim=-1)

        return I


    def save(self, path, suffix=""):
        print("saving..", path)
        path = Path(path)
        if not path.exists():
            path.mkdir()
        
        config = {"partition": self.partition}
        with open(path / f"R_config{suffix}.json", "w") as f:
            json.dump(config, f)
        
        obj = {"R.parametrizations.weight.0.base": self.R.weight.data}
        torch.save(obj, path / f"R{suffix}.pt")

    @classmethod
    def from_trained(cls, path, cfg, buffer, suffix=""):
        path = Path(path)
        with open(path / f"R_config{suffix}.json") as f:
            config = json.load(f)
        
        config["h_dim"] = sum(config["partition"])
        config["cfg"] = cfg
        config["buffer"] = buffer
        config["previous_R"] = torch.load(path / f"R{suffix}.pt", map_location="cpu")["R.parametrizations.weight.0.base"]
        trainer = cls(**config)
        print("model loaded", path)

        return trainer
    

class NewSplitRTrainer(nn.Module):
    def __init__(self, h_dim, prev_partition: list[int], previous_R: torch.FloatTensor, new_partition: list[list[tuple[int]]], cfg, buffer: BufferReuse):
        super().__init__()
 
        def get_random_init_matrix(d):
            if cfg.symmetric:
                return torch.eye(d)
            else:
                return torch.eye(d)[torch.randperm(d)].clone()

        self.h_dim = h_dim
        self.register_buffer("previous_R", previous_R)
        self.num_split_trial = cfg.num_split_trial * cfg.num_random_init
        assert len(new_partition) == self.num_split_trial
        assert all(len(prev_partition) == len(new_partition[i]) for i in range(self.num_split_trial))
        assert sum(prev_partition) == h_dim
        self.prev_partition = prev_partition
        self.new_partition = new_partition

        self.Rs = nn.ModuleList()
        for i in range(self.num_split_trial):
            self.Rs.append(
                nn.ModuleList([nn.utils.parametrizations.orthogonal(rotateMatrix(get_random_init_matrix(d))) for d in prev_partition])
            )

        self.metric = cfg.metric
        assert self.metric in ["cosine", "euclidean"]
        self.buffer = buffer
        self.cfg = cfg
        self.num_steps = cfg.search_steps

    def _get_subspace_h(self, h, trial_idx, dtype=torch.float32):
        h = (h.to(dtype) @ self.previous_R.to(dtype)).split(self.prev_partition, dim=-1)
        subspace_h = []
        for h_chunk, R, split_dim in zip(h, self.Rs[trial_idx], self.new_partition[trial_idx]):
            subspace_h.extend(list(R(h_chunk, dtype).split(split_dim, dim=-1)))
        return subspace_h
    
    def step(self):

        h = self.buffer.pop_one(self.cfg.batch_size)

        h = h[:self.cfg.batch_size]
        bz = h.size(0)
        device = h.device

        total_loss = 0
        for trial_idx in range(self.num_split_trial):
            query_h = self._get_subspace_h(h, trial_idx)
            num_unit = len(query_h)

            if self.metric == "cosine":
                with torch.no_grad():

                    query_h_normed = [h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=1e-8) for h in query_h]
                    most_sim_value = -torch.ones(num_unit, bz, dtype=torch.float, device=device)
                    most_sim_vec = torch.zeros(num_unit, bz, self.h_dim, dtype=self.buffer.buffer_dtype, device=device)

                    self.buffer.cursor = 0
                    for i in range(self.num_steps):
                        key_h = next(self.buffer)
                        subspace_key_h = self._get_subspace_h(key_h, trial_idx)
                        subspace_key_h = [h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=1e-3) for h in subspace_key_h]

                        sim = []
                        for q, k in zip(query_h_normed, subspace_key_h):
                            sim.append(q @ k.T)
                        sim = torch.stack(sim, dim=0)

                        max_v, max_idx = sim.max(dim=-1)
                        m = max_v > most_sim_value
                        most_sim_value[m] = max_v[m]
                        most_sim_vec[m] = key_h[max_idx[m]]

                most_sim_h = self._get_subspace_h(most_sim_vec.flatten(end_dim=1), trial_idx)
                sim = 0
                for i in range(len(most_sim_h)):
                    subspace_sim = F.cosine_similarity(query_h[i], most_sim_h[i].view(num_unit, bz, most_sim_h[i].size(-1))[i], dim=-1)
                
                    mean_sim_per_space = subspace_sim.mean()

                    sim = sim + mean_sim_per_space * query_h[i].size(-1)
                sim = sim / self.h_dim
                loss = -sim

            elif self.metric == "euclidean":
                with torch.no_grad():
                    smallest_dist = torch.full((num_unit, bz), torch.finfo(torch.float).max, dtype=torch.float, device=device)
                    most_sim_vec = torch.zeros(num_unit, bz, self.h_dim, dtype=self.buffer.buffer_dtype, device=device)
                    
                    self.buffer.cursor = 0
                    for i in range(self.num_steps):
                        key_h = next(self.buffer)
                        subspace_key_h = self._get_subspace_h(key_h, trial_idx)

                        dist = []
                        for q, k in zip(query_h, subspace_key_h):
                            dist.append(torch.cdist(q.unsqueeze(0), k.unsqueeze(0)))
                        dist = torch.cat(dist, dim=0)

                        min_v, min_idx = dist.min(dim=-1)
                        m = min_v < smallest_dist
                        smallest_dist[m] = min_v[m]
                        most_sim_vec[m] = key_h[min_idx[m]]
        
                   
                most_sim_h = self._get_subspace_h(most_sim_vec.flatten(end_dim=1), trial_idx)
                dist = 0
                for i in range(len(most_sim_h)):
                    subspace_dist = torch.linalg.vector_norm(query_h[i] - most_sim_h[i].view(num_unit, bz, most_sim_h[i].size(-1))[i], dim=-1)

                    mean_dist_per_space = subspace_dist.mean()

                    dist = dist + mean_dist_per_space
                dist = dist / len(most_sim_h)
                loss = dist
            
            total_loss = total_loss + loss
 
        return total_loss
    
    @torch.no_grad()
    def evaluate_step(self, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps

        h = self.buffer.pop_one(self.cfg.batch_size)
        h = h[:self.cfg.batch_size] # smaller bz when num_unit big
        bz = h.size(0)
        device = h.device

        all_result = []
        for trial_idx in range(self.num_split_trial):
            query_h = self._get_subspace_h(h, trial_idx)
            num_unit = len(query_h)

            if self.metric == "cosine":
                query_h = [h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=1e-8) for h in query_h]
                most_sim_value = -torch.ones(num_unit, bz, dtype=torch.float, device=device)

                self.buffer.cursor = 0
                for i in range(num_steps):
                    key_h = next(self.buffer)
                    key_h = self._get_subspace_h(key_h, trial_idx)
                    key_h = [h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=1e-3) for h in key_h]

                    sim = []
                    for q, k in zip(query_h, key_h):
                        sim.append(q @ k.T)
                    sim = torch.stack(sim, dim=0)

                    max_v, _ = sim.max(dim=-1)

                    most_sim_value = torch.max(most_sim_value, max_v)

                result = most_sim_value.mean(dim=1)
            
            elif self.metric == "euclidean":
                smallest_dist = torch.full((num_unit, bz), torch.finfo(torch.float).max, dtype=torch.float, device=device)
                
                self.buffer.cursor = 0
                for i in range(num_steps):
                    key_h = next(self.buffer)
                    key_h = self._get_subspace_h(key_h, trial_idx)

                    dist = []
                    for q, k in zip(query_h, key_h):
                        dist.append(torch.cdist(q.unsqueeze(0), k.unsqueeze(0)))
                    dist = torch.cat(dist, dim=0)

                    min_v, _ = dist.min(dim=-1)

                    smallest_dist = torch.min(smallest_dist, min_v)

                result = smallest_dist.mean(dim=1)
            
            all_result.append(result)

        return torch.stack(all_result)  # num_trial, num_spaces

    @torch.no_grad()
    def compute_subspace_var(self, num=2000):
        self.buffer.cursor = 0
        total_count = 0
        h = []
        while total_count < num:
            h.append(next(self.buffer))
            total_count += h[-1].size(0)
        h = torch.cat(h, dim=0)

        all_var = []
        for trial_idx in range(self.num_split_trial):
            all_subspace_h = self._get_subspace_h(h, trial_idx)

            var = torch.stack([subspace_h.var(dim=0).sum() for subspace_h in all_subspace_h])
            all_var.append(var)
        
        all_var = torch.stack(all_var)
        return all_var
    

    @torch.no_grad()
    def compute_MI_step(self, metric="euclidean", num_steps=None, subspace_var=None):

        pairs = [(i*2, i*2+1) for i in range(len(self.prev_partition))]

        h = self.buffer.pop_one(self.cfg.batch_size)
        h = h[:self.cfg.batch_size]
        bz = h.size(0)
        device = h.device

        all_I = []
        for trial_idx in range(self.num_split_trial):

            query_h = self._get_subspace_h(h, trial_idx)
            num_unit = len(query_h)

            if metric == "cosine":
                query_h = [h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=1e-8) for h in query_h]

            smallest_dist = {pair: torch.full((bz,), torch.finfo(torch.float).max, dtype=torch.float, device=device) for pair in pairs}
            smallest_dist_per_space = torch.full((num_unit, bz), torch.finfo(torch.float).max, dtype=torch.float, device=device)

            if num_steps is None:
                max_num_steps = - (self.search_num[trial_idx].max().item() // (-self.buffer.block_len))
                num_steps = min(max_num_steps, self.num_steps)
            
            self.buffer.cursor = 0
            for i in range(num_steps):
                key_h = next(self.buffer)
                key_h = self._get_subspace_h(key_h, trial_idx)

                if metric == "euclidean":
                    dist = []
                    for q, k in zip(query_h, key_h):
                        dist.append(torch.cdist(q.unsqueeze(0), k.unsqueeze(0)))
                    dist = torch.cat(dist, dim=0)
                    if subspace_var is not None:    # H is not correct in this case
                        dist /= subspace_var[trial_idx].view(-1, 1, 1).sqrt()
                elif metric == "cosine":
                    key_h = [h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=1e-8) for h in key_h]
                    sim = []
                    for q, k in zip(query_h, key_h):
                        sim.append(q @ k.T)
                    sim = torch.stack(sim, dim=0)
                    dist = 1 - sim

                for j, k in pairs:
                    min_v, _ = torch.max(dist[j], dist[k]).min(dim=-1)
                    smallest_dist[(j,k)] = torch.min(smallest_dist[(j,k)], min_v)
   
                min_v, _ = dist.min(dim=-1)
                smallest_dist_per_space = torch.min(smallest_dist_per_space, min_v)

            self.buffer.cursor = 0
            n_x_y = {pair: {"n_x": 0, "n_y":0} for pair in pairs}
            for i in range(num_steps):
                key_h = next(self.buffer)
                key_h = self._get_subspace_h(key_h, trial_idx)

                if metric == "euclidean":
                    dist = []
                    for q, k in zip(query_h, key_h):
                        dist.append(torch.cdist(q.unsqueeze(0), k.unsqueeze(0)))
                    dist = torch.cat(dist, dim=0)
                    if subspace_var is not None:    # H is not correct in this case
                        dist /= subspace_var[trial_idx].view(-1, 1, 1).sqrt()
                elif metric == "cosine":
                    key_h = [h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=1e-8) for h in key_h]
                    sim = []
                    for q, k in zip(query_h, key_h):
                        sim.append(q @ k.T)
                    sim = torch.stack(sim, dim=0)
                    dist = 1 - sim

                for j, k in pairs:
                    n_x_y[(j,k)]["n_x"] += (dist[j] < smallest_dist[(j,k)].unsqueeze(-1)).sum(dim=-1)
                    n_x_y[(j,k)]["n_y"] += (dist[k] < smallest_dist[(j,k)].unsqueeze(-1)).sum(dim=-1)

            I = []
            N = num_steps * self.buffer.block_len + 1
            constant = torch.special.digamma(torch.ones((), device=device)) + torch.special.digamma(torch.tensor(N, device=device))
            for j, k in pairs:
                I.append( (constant - torch.special.digamma(n_x_y[(j,k)]["n_x"]+1) - torch.special.digamma(n_x_y[(j,k)]["n_y"]+1)).mean() )

            all_I.append(torch.stack(I))

        return torch.stack(all_I)
    
    @torch.no_grad()
    def evaluate_step_given_partition(self, prev_R, prev_partition, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps

        h = self.buffer.pop_one(self.cfg.batch_size)
        h = h[:self.cfg.batch_size] # smaller bz when num_unit big
        bz = h.size(0)
        device = h.device

        query_h = (h.to(prev_R.dtype) @ prev_R).split(prev_partition, dim=-1)
        num_unit = len(query_h)

        if self.metric == "cosine":
            query_h = [h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=1e-8) for h in query_h]
            most_sim_value = -torch.ones(num_unit, bz, dtype=torch.float, device=device)

            for i in range(num_steps):
                key_h = next(self.buffer)
                key_h = (key_h.to(prev_R.dtype) @ prev_R).split(prev_partition, dim=-1)

                key_h = [h / torch.linalg.vector_norm(h, dim=-1, keepdim=True).clamp(min=1e-3) for h in key_h]

                sim = []
                for q, k in zip(query_h, key_h):
                    sim.append(q @ k.T)
                sim = torch.stack(sim, dim=0)

                max_v, _ = sim.max(dim=-1)

                most_sim_value = torch.max(most_sim_value, max_v)

            result = most_sim_value.mean(dim=1)
        
        elif self.metric == "euclidean":
            smallest_dist = torch.full((num_unit, bz), torch.finfo(torch.float).max, dtype=torch.float, device=device)
            
            for i in range(num_steps):
                key_h = next(self.buffer)
                key_h = (key_h.to(prev_R.dtype) @ prev_R).split(prev_partition, dim=-1)

                dist = []
                for q, k in zip(query_h, key_h):
                    dist.append(torch.cdist(q.unsqueeze(0), k.unsqueeze(0)))
                dist = torch.cat(dist, dim=0)

                min_v, _ = dist.min(dim=-1)

                smallest_dist = torch.min(smallest_dist, min_v)

            result = smallest_dist.mean(dim=1)
        
        return result