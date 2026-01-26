import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import math
import random
import json
import numpy as np
import os

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class toyModel(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()
        W = torch.rand(x_dim, h_dim)
        k = math.sqrt(1 / x_dim)
        W = W * 2 * k - k
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(torch.zeros(x_dim))

    def forward(self, x):
        assert x.dim() == 2
        return F.relu(x @ self.W @ self.W.T + self.b.unsqueeze(0))
    
class toyData(IterableDataset):
    def __init__(self, n_feature, sparsity):
        super().__init__()
        self.n_feature = n_feature
        self.sparsity = sparsity
        self.x_dim = sum(n_feature)

    def __iter__(self):
        while True:
            data = []
            for group in self.n_feature:
                if random.random() > self.sparsity:
                    idx = random.randint(0, group-1)
                    data.append(F.one_hot(torch.tensor(idx), group).float() * random.random())
                else:
                    data.append(torch.zeros(group))
            data = torch.cat(data, dim=0)
            yield data

if __name__ == "__main__":
    set_seed(0)
    n_feature = [20, 20]
    # n_feature = [5, 15, 5, 15]
    # n_feature = [80, 80]
    # n_feature = [40, 80, 120, 160]
    x_dim = sum(n_feature)
    h_dim = 12
    # h_dim= 32
    sparsity = 0.25  # prob of a whole group not occur
    print(sparsity)
    importance = torch.linspace(2, 0, x_dim+2)[1:-1][torch.randperm(x_dim)]
    importance = torch.ones(x_dim)
    print(importance)

    model = toyModel(x_dim, h_dim)
    dataset = toyData(n_feature, sparsity)
    dataloader = DataLoader(dataset, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10000)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=3e-4, max_lr=3e-3, step_size_up=2000)

    try:
        for i, x in enumerate(dataloader):
            if i == 10000:
                break
            
            pred = model(x)
            loss = (importance.unsqueeze(0) * (x - pred)**2).sum(dim=1).mean()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                print(loss.item())
            
            scheduler.step()
    except KeyboardInterrupt:
        pass

    with torch.no_grad():
        all_x = []
        all_pred = []
        for i, x in enumerate(dataloader):
            if i == 100:
                break
            pred = model(x)
            all_x.append(x)
            all_pred.append(pred)
        all_x = torch.cat(all_x,dim=0)
        all_pred = torch.cat(all_pred,dim=0)
    
        total_v = all_x.var(dim=0, correction=0).mean()
        unexplained_v = ((all_x - all_pred)**2).mean(dim=0).mean()
        fvu = (unexplained_v / total_v).item()
        print("\nfvu", fvu, "\n")

    print("saving...")
    config = {"n_feature": n_feature, "h_dim": h_dim, "sparsity": sparsity, "importance": importance.tolist()}
    with open("config.json", "w") as f:
        json.dump(config, f)
    torch.save(model.state_dict(), "model.pt")