from transformer_lens import HookedTransformer
from datasets import load_dataset, concatenate_datasets, load_from_disk
import torch
from tqdm import tqdm
import gc
import random
import re
from collections import deque
import torch.nn.functional as F


class BufferReuse:
    # a buffer designed for extended search range
    def __init__(self, cfg, model: HookedTransformer, normalize=False):
        self.cfg = cfg

        self.load_dataset()

        self.block_len = cfg.block_len
        self.refresh_block_num = cfg.refresh_block_num

        self.blocks = deque()
        self.cursor = 0

        self.cfg = cfg
        self.model = model
        self.buffer_dtype = torch.float16
        assert not normalize


        self.refresh()

        self.mean = self.compute_mean()

        self.to_pop = None
    
    def load_dataset(self):
        if self.cfg.data_source == "openwebtext":
            data = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
            data = data.shuffle(buffer_size=200_000)
            self.data = map(lambda x:x["text"], data)
        elif self.cfg.data_source == "minipile":
            data = load_dataset("JeanKaddour/minipile", split="train")
            data = data.shuffle()
            self.data = map(lambda x:x["text"], data)
        else:
            raise NotImplementedError
    
    @torch.no_grad()
    def compute_mean(self):
        s = 0
        for block in self.blocks:
            s += block.float().mean(dim=0, keepdim=True)  # important: first float() then mean()
        return s / len(self.blocks)
    
    def __iter__(self):
        return self
    
    def pop_one(self, bz=None):
        if len(self.blocks) == 0:
            self.refresh()

        self.cursor = 0
        if bz is None or bz == self.block_len:
            acts = self.blocks.popleft()
        elif bz > self.block_len:
            n = -(bz // -self.block_len)
            acts = torch.cat([self.blocks.popleft() for _ in range(n)], dim=0)
        elif self.to_pop is None or self.to_pop.size(0) < bz:
            acts = self.blocks.popleft()
            acts, self.to_pop = acts.split([bz, acts.size(0)-bz], dim=0)
        else:
            acts, self.to_pop = self.to_pop.split([bz, self.to_pop.size(0)-bz], dim=0)

        return acts.contiguous()

    def __next__(self):
        """
        Return a batch of activations
        """
        with torch.no_grad():
            if self.cursor >= len(self.blocks):
                self.refresh()

            out = self.blocks[self.cursor]
            self.cursor += 1

            return out
    

    def token_batch(self):
        """
        Return a batch of tokens (flattened)
        """
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

        if self.cfg.act_site != "blocks.0.hook_resid_pre":
            self.model.reset_hooks()
            cache = self.model.add_caching_hooks(self.cfg.act_site)
            stop_at_layer = int(re.search(r"blocks\.(\d+)\.", self.cfg.act_site).group(1)) + 1
        else:
            stop_at_layer = 0

        buffer = []

        pbar = tqdm(total=self.block_len * self.refresh_block_num, desc="Refreshing activations", disable=True)
        while len(buffer) < self.block_len * self.refresh_block_num:
            # inside no_grad()
            input_batch = self.token_batch()
            input_batch = torch.tensor(input_batch, device=self.cfg.device, dtype=torch.long).view(self.cfg.caching_batch_size, -1)

            with torch.autocast("cuda"):
                acts = self.model(input_batch, stop_at_layer=stop_at_layer)
                if self.cfg.act_site != "blocks.0.hook_resid_pre":
                    acts = cache[self.cfg.act_site]
                # acts = F.layer_norm(acts, [acts.size(-1)])
                
            acts = acts.flatten(end_dim=1).to(self.buffer_dtype)
            buffer.extend(list(torch.unbind(acts)))
            pbar.update(acts.size(0))

        pbar.close()

        random.shuffle(buffer)

        for i in range(self.refresh_block_num):
            block = torch.stack([buffer.pop() for j in range(self.block_len)])
            self.blocks.append(block)
        
        assert self.blocks[-1].dtype == self.buffer_dtype



class BufferReuseDoubleQueue:
    def __init__(self, cfg, model: HookedTransformer, normalize=False):
        self.cfg = cfg

        self.load_dataset()

        self.block_len = cfg.block_len
        self.refresh_block_num = cfg.refresh_block_num
        assert self.refresh_block_num % 2 == 0

        self.blocks1 = deque()
        self.blocks2 = deque()
        self.next_q_at_que1 = True
        self.cursor = 0

        self.cfg = cfg
        self.model = model
        self.buffer_dtype = torch.float16  
        assert not normalize

        self.refresh()

        self.mean = self.compute_mean()

        self.to_pop = None

    def load_dataset(self):
        if self.cfg.data_source == "openwebtext":
            data = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
            data = data.shuffle(buffer_size=200_000)
            self.data = map(lambda x:x["text"], data)
        elif self.cfg.data_source == "minipile":
            data = load_dataset("JeanKaddour/minipile", split="train")
            data = data.shuffle()
            self.data = map(lambda x:x["text"], data)
        else:
            raise NotImplementedError
    
    @torch.no_grad()
    def compute_mean(self):
        s = 0
        for block in self.blocks1:
            s += block.float().mean(dim=0, keepdim=True)  # important: first float() then mean()
        return s / len(self.blocks1)
    
    def __iter__(self):
        return self

    def pop_one(self, bz=None):
        if len(self.blocks1) == 0 or len(self.blocks2) == 0:
            self.refresh()

        self.cursor = 0
        if bz is None or bz == self.block_len:
            if self.next_q_at_que1:
                q_act = self.blocks1.popleft()
                self.next_q_at_que1 = False
            else:
                q_act = self.blocks2.popleft()
                self.next_q_at_que1 = True
        elif bz > self.block_len:
            n = -(bz // -self.block_len)
            if self.next_q_at_que1:
                q_act = torch.cat([self.blocks1.popleft() for _ in range(n)], dim=0)
                self.next_q_at_que1 = False
            else:
                q_act = torch.cat([self.blocks2.popleft() for _ in range(n)], dim=0)
                self.next_q_at_que1 = True
        elif self.to_pop is None or self.to_pop.size(0) < bz:
            if self.next_q_at_que1:
                q_act = self.blocks1.popleft()
                self.next_q_at_que1 = False
            else:
                q_act = self.blocks2.popleft()
                self.next_q_at_que1 = True
            q_act, self.to_pop = q_act.split([bz, q_act.size(0)-bz], dim=0)
        else:
            q_act, self.to_pop = self.to_pop.split([bz, self.to_pop.size(0)-bz], dim=0)

        return q_act

    def __next__(self):
        """
        Return a batch of activations
        """
        with torch.no_grad():
            if self.next_q_at_que1:
                blocks = self.blocks1
            else:
                blocks = self.blocks2

            if self.cursor >= len(blocks):
                self.refresh()
            
            out = blocks[self.cursor]
            self.cursor += 1

            return out
    

    def token_batch(self):
        """
        Return a batch of tokens (flattened)
        """
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

        if self.cfg.act_site != "blocks.0.hook_resid_pre":
            self.model.reset_hooks()
            cache = self.model.add_caching_hooks(self.cfg.act_site)
            stop_at_layer = int(re.search(r"blocks\.(\d+)\.", self.cfg.act_site).group(1)) + 1
        else:
            stop_at_layer = 0

        buffer = []

        pbar = tqdm(total=self.block_len * self.refresh_block_num, desc="Refreshing activations", disable=True)
        while len(buffer) < self.block_len * self.refresh_block_num:
            # inside no_grad()
            input_batch = self.token_batch()
            input_batch = torch.tensor(input_batch, device=self.cfg.device, dtype=torch.long).view(self.cfg.caching_batch_size, -1)

            with torch.autocast("cuda"):
                acts = self.model(input_batch, stop_at_layer=stop_at_layer)
                if self.cfg.act_site != "blocks.0.hook_resid_pre":
                    acts = cache[self.cfg.act_site]
                
            acts = acts.flatten(end_dim=1).to(self.buffer_dtype)
            buffer.extend(list(torch.unbind(acts)))
            pbar.update(acts.size(0))

        pbar.close()

        buffer = buffer[:self.block_len * self.refresh_block_num]
        buffer1, buffer2 = buffer[:len(buffer)//2], buffer[len(buffer)//2:]
        random.shuffle(buffer1)
        random.shuffle(buffer2)

        for i in range(self.refresh_block_num//2):
            block = torch.stack([buffer1.pop() for j in range(self.block_len)])
            self.blocks1.append(block)

        for i in range(self.refresh_block_num//2):
            block = torch.stack([buffer2.pop() for j in range(self.block_len)])
            self.blocks2.append(block)
        
        assert self.blocks1[-1].dtype == self.buffer_dtype

