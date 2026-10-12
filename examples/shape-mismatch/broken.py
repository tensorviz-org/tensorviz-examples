"""A deliberately wrong Linear width after flattening a small image model."""

from torch import nn


class BrokenClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(128, 5)  # Wrong: a 16 x 16 input produces 512 features.

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)
