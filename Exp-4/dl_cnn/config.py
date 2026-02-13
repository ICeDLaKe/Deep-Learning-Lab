DEFAULTS = {
    "data_path": r"C:\Users\hritt\Documents\Coding\ANN\ANN_A2_Group01\code\dataset_assignment2_afi507",  # override via CLI
    "base_dir": r"C:\Users\hritt\Documents\Coding\ANN\ANN_A2_Group01\code\results",
    "seed": 42,
    "batch_size": 64,
    "epochs": 30,
    "lr": 1e-3,
    "n_blocks": 3,
    "activation": "relu",   # relu | tanh | none
    "dropout_conv": 0.2,
    "dropout_fc": 0.5,
    "init": "he",           # zero | random | he
    "device": None,         # if None it will auto-detect the device
    "num_workers": 0,
    "pin_memory": True,
    "save_best_only": True,
    "early_stop_patience": 5
}
