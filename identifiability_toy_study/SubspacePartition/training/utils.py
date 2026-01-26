
import argparse
import torch
import numpy as np
import random
import easydict
import numpy as np
import os
import re

def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.
    """
    
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    return easydict.EasyDict(cfg)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def print_to_both(*args, f=None):
    print(*args)
    print(*args, file=f, flush=True)


def site_name_to_short_name(act_site):
    block_idx = int(re.search(r"blocks\.(\d+)\.", act_site).group(1))
    if "hook_result" in act_site:
        head_idx = int(re.search(r"hook_result\.(\d+)", act_site).group(1))
        return f"a{block_idx}.{head_idx}"
    elif "hook_resid" in act_site:
        layer_idx = re.search(r"hook_resid_([a-z]+)", act_site).group(1)
        return f"x{block_idx}.{layer_idx}"
    elif "mlp_out" in act_site:
        return f"m{block_idx}"
    else:
        raise NotImplementedError()
    
def short_name_to_site_name(act_name):
    # a0.1   x0.mid  m0
    if act_name is None:
        return None
    elif act_name.startswith("a"):
        block_idx, head_idx = act_name.lstrip("a").split(".")
        return f"blocks.{block_idx}.attn.hook_result.{head_idx}"
    elif act_name.startswith("x"):
        block_idx, layer_idx = act_name.lstrip("x").split(".")
        return f"blocks.{block_idx}.hook_resid_{layer_idx}"
    elif act_name.startswith("m"):
        block_idx = act_name.lstrip("m")
        return f"blocks.{block_idx}.hook_mlp_out"
    else:
        raise NotImplementedError()

 
def to_valid_model_name(model_name):
    if model_name == "gpt2":
        return "gpt2"
    elif model_name == "qwen2.5":
        return "qwen2.5-1.5b"
    elif model_name == "gemma2":
        return "gemma-2-2b"
    else:
        raise NotImplementedError
