import torch.nn as nn

class ReLuActivation(nn.Module):

    def forward(self, x):
        return nn.ReLU()(x)