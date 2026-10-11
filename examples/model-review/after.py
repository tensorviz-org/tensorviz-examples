"""The edited model: GELU and LayerNorm, with the same output shape."""

from torch import nn


class ReviewNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8, bias=False)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(8)
        self.output = nn.Linear(8, 2, bias=False)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.norm(x)
        return self.output(x)
