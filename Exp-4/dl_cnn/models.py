import torch.nn as nn
import torch

def get_activation(name):
    name = (name or "relu").lower()
    if name == "relu": return nn.ReLU(inplace=True)
    if name == "tanh": return nn.Tanh()
    if name == "none": return nn.Identity()
    raise ValueError("Unknown activation")

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, activation="relu", use_pool=True):
        super().__init__()
        act = get_activation(activation)
        layers = [
            nn.Conv2d(in_c, out_c, 3, padding=1),
            act,
            nn.Conv2d(out_c, out_c, 3, padding=1),
            act
        ]
        if use_pool:
            layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

class CNNClassifier(nn.Module):
    def __init__(self, n_blocks=1, activation="relu", dropout_conv=0.0, dropout_fc=0.5, num_classes=10):
        super().__init__()
        C = [16, 32, 64]
        in_c = 3
        blocks = []
        for i in range(n_blocks):
            blocks.append(ConvBlock(in_c, C[i], activation=activation, use_pool=True))
            in_c = C[i]
        self.conv = nn.Sequential(*blocks)
        self.drop2d = nn.Dropout2d(dropout_conv) if dropout_conv > 0 else nn.Identity()

        spatial = 32 // (2 ** n_blocks)
        flat_dim = C[n_blocks-1] * spatial * spatial

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 128),
            get_activation(activation),
            nn.Dropout(dropout_fc),
            nn.Linear(128, 64),
            get_activation(activation),
            nn.Dropout(dropout_fc),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, return_probs=False):
        x = self.conv(x)
        x = self.drop2d(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return nn.functional.softmax(logits, dim=1) if return_probs else logits


def init_weights(model, mode="he"):
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if mode == "zero":
                nn.init.zeros_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif mode == "random":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif mode == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            else:
                raise ValueError("Unknown init mode")
