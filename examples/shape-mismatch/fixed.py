"""The fixed-resolution correction: 8 channels times 8 times 8 spatial values."""

from torch import nn


class FixedClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(8 * 8 * 8, 5)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)
