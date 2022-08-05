import numpy as np
import PIL
import torch


class cifar10_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        X = PIL.Image.fromarray(img).convert("RGB")
        label = self.labels[idx]

        if self.transforms:
            X = self.transforms(X)
        if label is None:
            return torch.tensor(X, dtype=torch.float)
        return torch.tensor(X, dtype=torch.float), torch.tensor(label, dtype=torch.long)
