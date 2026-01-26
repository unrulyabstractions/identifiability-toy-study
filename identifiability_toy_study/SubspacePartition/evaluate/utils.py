import re
import bisect

# duplicated code
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
    
def locate_str_tokens(act_idx, seq_edges):
    seq_idx = bisect.bisect(seq_edges, act_idx) - 1
    pos_idx = act_idx - seq_edges[seq_idx]
    return seq_idx, pos_idx

def print_to_both(*args, f=None):
    print(*args)
    print(*args, file=f, flush=True)


