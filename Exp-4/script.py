import argparse
from ann_cnn import config as cfg_module
from ann_cnn.utils import set_seed, save_config, ensure_dir
from ann_cnn.dataset import load_pickle, create_split, PickleImageDataset
from ann_cnn.transforms import Normalize, RandomCropFlip
from ann_cnn.models import CNNClassifier, init_weights
from ann_cnn.train import run_training
from ann_cnn import experiments, viz
import torch
import os

def parse_args():
    p = argparse.ArgumentParser(description="Run ANN CNN experiments")
    p.add_argument("--data_path", type=str, default=cfg_module.DEFAULTS["data_path"])
    p.add_argument("--base_dir", type=str, default=cfg_module.DEFAULTS["base_dir"])
    p.add_argument("--epochs", type=int, default=cfg_module.DEFAULTS["epochs"])
    p.add_argument("--batch_size", type=int, default=cfg_module.DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=cfg_module.DEFAULTS["lr"])
    p.add_argument("--n_blocks", type=int, default=cfg_module.DEFAULTS["n_blocks"])
    p.add_argument("--activation", type=str, default=cfg_module.DEFAULTS["activation"])
    p.add_argument("--dropout_conv", type=float, default=cfg_module.DEFAULTS["dropout_conv"])
    p.add_argument("--dropout_fc", type=float, default=cfg_module.DEFAULTS["dropout_fc"])
    p.add_argument("--init", type=str, default=cfg_module.DEFAULTS["init"])
    p.add_argument("--seed", type=int, default=cfg_module.DEFAULTS["seed"])
    p.add_argument("--num_workers", type=int, default=cfg_module.DEFAULTS["num_workers"])
    p.add_argument("--run_all", action="store_true", help="Run all experiments (blocks, dropout, init, best, no-activation)")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = {
        "data_path": args.data_path,
        "base_dir": args.base_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_blocks": args.n_blocks,
        "activation": args.activation,
        "dropout_conv": args.dropout_conv,
        "dropout_fc": args.dropout_fc,
        "init": args.init,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "early_stop_patience": 5
    }
    set_seed(cfg["seed"])
    ensure_dir(cfg["base_dir"])
    save_config(cfg, cfg["base_dir"])

    images, labels = load_pickle(cfg["data_path"])
    (train_x, train_y), (val_x, val_y) = create_split(images, labels)
    # transforms: normalize and augmentation on train
    train_transform = lambda x: RandomCropFlip()(Normalize().__call__(x))
    val_transform = Normalize()

    train_ds = PickleImageDataset(train_x, train_y, transform=train_transform)
    val_ds = PickleImageDataset(val_x, val_y, transform=val_transform)

    # create loaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                                               num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                                             num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])

    # visualize samples
    viz_path = viz.visualize_10_per_class(train_ds, save_dir=os.path.join(cfg["base_dir"], "visualizations"))
    print("Saved samples at", viz_path)

    if args.run_all:
        print("Running block experiment...")
        experiments.run_block_experiment(train_ds, val_ds, cfg, os.path.join(cfg["base_dir"], "blocks"))
        print("Running dropout experiment...")
        experiments.run_dropout_experiment(train_ds, val_ds, cfg, os.path.join(cfg["base_dir"], "dropout"))
        print("Running initialization experiment...")
        experiments.run_init_experiment(train_ds, val_ds, cfg, os.path.join(cfg["base_dir"], "init"))

    # run best model as default single run
    print("Training final model (best setting)...")
    model = CNNClassifier(n_blocks=3, activation=cfg["activation"],
                          dropout_conv=cfg["dropout_conv"],
                          dropout_fc=cfg["dropout_fc"])
    init_weights(model, mode=cfg["init"])
    out_dir = os.path.join(cfg["base_dir"], "best_model")
    history, best_val = run_training(model, train_loader, val_loader, cfg, out_dir)
    print("Best validation acc (this run) =", best_val)
    # plot history
    hist_csv = os.path.join(out_dir, "history.csv")
    if os.path.exists(hist_csv):
        viz.plot_history(hist_csv, save_prefix=os.path.join(out_dir, "best"))
        print("Saved best model plots.")
    # run no-activation experiment
    print("Running no-activation experiment...")
    model_no_act = CNNClassifier(n_blocks=3, activation="none", dropout_conv=0.0, dropout_fc=0.0)
    init_weights(model_no_act, mode=cfg["init"])
    out_dir_no = os.path.join(cfg["base_dir"], "no_activation")
    history_no, best_no = run_training(model_no_act, train_loader, val_loader, dict(cfg, epochs=cfg["epochs"]), out_dir_no)
    print("No-activation best val acc:", best_no)

if __name__ == "__main__":
    main()
