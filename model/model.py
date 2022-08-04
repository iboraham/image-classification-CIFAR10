import torch
import torch.nn as nn
from torchvision import models


class CIFAR10Model(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        # self.model.fc = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.Linear(1024, 512),
        #     nn.Linear(512, 256),
        #     nn.Linear(256, 128),
        #     nn.Linear(128, 64),
        #     nn.Linear(64, 32),
        #     nn.Linear(32, 16),
        #     nn.Linear(16, num_classes),
        #     )
        self.model.fc = nn.Sequential(nn.Linear(2048, num_classes))

    def forward(self, x):
        x = self.model(x)
        return x
