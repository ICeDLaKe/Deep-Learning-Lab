import random
import numpy as np
import torch
import os
import json

def set_seed(seed=79):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_config(cfg, out_dir):
    ensure_dir(out_dir)
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
