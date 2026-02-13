import numpy as np
import pickle
from torch.utils.data import Dataset
import torch

class PickleImageDataset(Dataset):
    """
    Expect: data array shaped (N, 3072) or (N, 3, 32, 32)
    labels: list/array length N
    transform: callable on torch.FloatTensor image (C,H,W)
    """
    def __init__(self, data_np, labels, transform=None):
        # Accept both flattened or channel-first arrays
        self._raw = np.array(data_np)
        if self._raw.ndim == 2 and self._raw.shape[1] == 3072:
            self.data = self._raw.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        else:
            self.data = self._raw.astype(np.float32)
            if self.data.shape[1] != 3:
                # try channel-first reformat
                self.data = self.data.reshape(-1, 3, 32, 32)
            self.data /= 255.0
        self.labels = np.array(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx]).float()
        lbl = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl


def load_pickle(path):
    """
    Loads pickle file saved in original assignment format (dict with 'data' and 'labels' or 'train').
    """
    with open(path, 'rb') as f:
        raw = pickle.load(f, encoding='bytes')
    # decode keys
    keys = { (k.decode() if isinstance(k, bytes) else k): v for k, v in raw.items() }

    if "data" in keys:
      images = np.array(keys["data"])
    elif "images" in keys:
      images = np.array(keys["images"])
    else:
      raise KeyError("No 'data' or 'images' key found in pickle file.")

    # labels
    if "labels" in keys:
      labels = keys["labels"]
    elif "train" in keys:
      labels = keys["train"]
    else:
      raise KeyError("No 'labels' or 'train' key found in pickle file.")
    return images, labels


def create_split(images, labels, train_n=8000):
    images = np.array(images)
    labels = list(labels)
    assert len(labels) >= train_n, "Not enough samples in dataset"
    train_x = images[:train_n]
    train_y = labels[:train_n]
    val_x = images[train_n:train_n + (len(labels)-train_n)]
    val_y = labels[train_n:train_n + (len(labels)-train_n)]
    return (train_x, train_y), (val_x, val_y)
