import timm
import torch
import torch.nn as nn

import tez
from sklearn import metrics


class cifar10_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("resnet18", pretrained=True)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)

    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        f1 = metrics.f1_score(targets, outputs, average="macro")
        return {"f1": torch.tensor(f1, device=device)}

    def optimizer_scheduler(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.5,
            patience=2,
            verbose=True,
            mode="max",
            threshold=1e-4,
        )
        return opt, sch

    def forward(self, image, label=None):
        outputs = self.model(image)
        if label is not None:
            loss = nn.CrossEntropyLoss()(outputs, label.type(torch.LongTensor))
            metrics = self.monitor_metrics(outputs, label)
            return outputs, loss, metrics
        return outputs, 0, {}
