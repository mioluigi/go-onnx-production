"""The PyTorch model, kept apart from training so that timing it does not also time torchvision's import."""

import torch
from torch import nn


class DigitClassifier(nn.Module):
    """Multilayer network 784 -> 128 -> 64 -> 10. Outputs logits, as CrossEntropyLoss expects for training."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class WithProbabilities(nn.Module):
    """The exported model: the classifier followed by softmax, so its output is class probabilities."""

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, inputs):
        return torch.softmax(self.classifier(inputs), dim=1)
