import os
from .models import CNNClassifier, init_weights
from .train import run_training
import torch
import csv

def run_block_experiment(train_ds, val_ds, cfg, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    results = []
    for nb in [1,2,3]:
        print(f"Running {nb}-block model...")
        cfg_local = dict(cfg)
        cfg_local["n_blocks"] = nb
        model = CNNClassifier(n_blocks=nb,
                              activation=cfg_local["activation"],
                              dropout_conv=cfg_local["dropout_conv"],
                              dropout_fc=cfg_local["dropout_fc"])
        init_weights(model, mode=cfg_local.get("init","he"))
        out_dir = os.path.join(base_dir, f"blocks_{nb}")
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg_local["batch_size"], shuffle=True,
                                                   num_workers=cfg_local.get("num_workers",4), pin_memory=cfg_local.get("pin_memory",True))
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg_local["batch_size"], shuffle=False,
                                                 num_workers=cfg_local.get("num_workers",4), pin_memory=cfg_local.get("pin_memory",True))
        history, best_val = run_training(model, train_loader, val_loader, cfg_local, out_dir)
        results.append([nb, best_val])
    with open(os.path.join(base_dir, "block_results.csv"), "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["n_blocks","best_val_acc"]); writer.writerows(results)
    return dict(results)

def run_dropout_experiment(train_ds, val_ds, cfg, base_dir, ps=[0.2,0.5,0.8]):
    os.makedirs(base_dir, exist_ok=True)
    results=[]
    for p in ps:
        print(f"Running dropout p={p} ...")
        cfg_local = dict(cfg)
        cfg_local["dropout_fc"]=p
        model = CNNClassifier(n_blocks=cfg_local["n_blocks"], activation=cfg_local["activation"],
                              dropout_conv=cfg_local["dropout_conv"], dropout_fc=p)
        init_weights(model, mode=cfg_local.get("init","he"))
        out_dir = os.path.join(base_dir, f"dropout_{str(p).replace('.','')}")
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg_local["batch_size"], shuffle=True,
                                                   num_workers=cfg_local.get("num_workers",4), pin_memory=cfg_local.get("pin_memory",True))
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg_local["batch_size"], shuffle=False,
                                                 num_workers=cfg_local.get("num_workers",4), pin_memory=cfg_local.get("pin_memory",True))
        history, best_val = run_training(model, train_loader, val_loader, cfg_local, out_dir)
        results.append([p, best_val])
    with open(os.path.join(base_dir,"dropout_results.csv"), "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["p","best_val_acc"]); writer.writerows(results)
    return dict(results)

def run_init_experiment(train_ds, val_ds, cfg, base_dir, inits=["zero","random","he"]):
    os.makedirs(base_dir, exist_ok=True)
    results=[]
    for mode in inits:
        print(f"Running init={mode} ...")
        cfg_local = dict(cfg)
        cfg_local["init"]=mode
        model = CNNClassifier(n_blocks=cfg_local["n_blocks"], activation=cfg_local["activation"],
                              dropout_conv=cfg_local["dropout_conv"], dropout_fc=cfg_local["dropout_fc"])
        init_weights(model, mode=mode)
        out_dir = os.path.join(base_dir, f"init_{mode}")
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg_local["batch_size"], shuffle=True,
                                                   num_workers=cfg_local.get("num_workers",4), pin_memory=cfg_local.get("pin_memory",True))
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg_local["batch_size"], shuffle=False,
                                                 num_workers=cfg_local.get("num_workers",4), pin_memory=cfg_local.get("pin_memory",True))
        history, best_val = run_training(model, train_loader, val_loader, cfg_local, out_dir)
        results.append([mode, best_val])
    with open(os.path.join(base_dir,"init_results.csv"), "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["init","best_val_acc"]); writer.writerows(results)
    return dict(results)
