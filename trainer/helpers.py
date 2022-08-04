import numpy as np
import torch


def save_model(path, epoch, model, optimizer, loss, acc):
    """Saves Pytorch model.

    Args:
        Path (str): Path to save model in.
        Epoch (int): Current epoch in model training.
        Model (obj): Pytorch model.
        Optimizer (obj): Pytorch optimizer used for 'model' arg.
        Loss, acc (float): Current epoch loss/accuracy value.

    Returns:
        A dictionary saved at the specified path with 'epoch',
        'model_state_dict', 'optimizer_state_dict', 'loss', and
        'acc' as keys, and the args as values.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "acc": acc,
        },
        path,
    )
    return


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.

    Developed by: https://github.com/Bjarten/early-stopping-pytorch

    Attributes:
      patience (int): How long to wait after last time validation
        loss improved. Default: 7
      delta (float): Minimum change in the monitored quantity to
        qualify as an improvement. Default: 0
      trace_func (function): trace print function.
                    Default: print
    """

    def __init__(self, patience=7, delta=0, trace_func=print):
        """Inits class with patience, verbose, delta, and trace_func."""
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
