"""The original model: a small ReLU network with a [batch, 2] output."""

from torch import nn


class ReviewNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8, bias=False)
        self.activation = nn.ReLU()
        self.output = nn.Linear(8, 2, bias=False)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        return self.output(x)
