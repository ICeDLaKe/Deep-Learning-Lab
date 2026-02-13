import torch
import torch.nn.functional as F
import random

class Normalize:
    def __init__(self, mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616)):
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std = torch.tensor(std).view(3,1,1)
    def __call__(self, x):
        # x expected [C,H,W] float in [0,1]
        return (x - self.mean) / self.std

class RandomCropFlip:
    def __init__(self, pad=4, prob_hflip=0.5):
        self.pad = pad
        self.prob_hflip = prob_hflip

    def __call__(self, x):
        # x: tensor [C,H,W]
        c,h,w = x.shape
        x = F.pad(x.unsqueeze(0), (self.pad, self.pad, self.pad, self.pad), mode="reflect").squeeze(0)
        top = random.randint(0, 2*self.pad)
        left = random.randint(0, 2*self.pad)
        x = x[:, top:top+h, left:left+w]
        if random.random() < self.prob_hflip:
            x = torch.flip(x, dims=[2])
        return x
