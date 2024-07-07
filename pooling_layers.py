import torch.nn as nn

class PoolingLayer(nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0):
        super(PoolingLayer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)
    