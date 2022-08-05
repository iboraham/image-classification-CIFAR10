import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .helpers import save_model


def train_model(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    save_path,
    num_epochs=25,
    scheduler=None,
    early_stopping=None,
):
    """Trains a Pytorch model.

    Args:
        Model (obj): Pytorch model.
        Dataloader (dict): A dictionary with 'train' and 'val'
          as keys and Pytorch dataloader classes as values.
        Criterion (obj): A Pytorch loss function.
        Optimizer (obj): A Pytorch optimizer.
        Save_path (str): Path to save model checkpoint.
        Num_epochs (int): Number of epochs to train. Default: 25.
        Scheduler (obj): A Pytorch learning rate scheduler. Default: None.
        Early_stopping (obj): An instance of the 'EarlyStopping' class.
          Default: None.

    Returns:
        The model with the lowest validation loss across all epochs.
    """

    log_df = pd.DataFrame(
        {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    )
    log_filename = f'{save_path.split(".")[0]}_logs.csv'

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    best_acc = 0.0
    best_loss = np.inf

    if early_stopping:
        early_stopping = early_stopping

    tbar_epoch = tqdm(range(num_epochs), desc="Epochs")
    for epoch in tbar_epoch:

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            tbar_train = tqdm(
                dataloader[phase], position=1, leave=False, desc=f"{phase}"
            )
            accs = []
            for inputs, labels in tbar_train:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                temp = loss.item() * inputs.size(0)
                # tbar_train.set_postfix({"Loss": f"{temp:.3f}"})
                running_loss += temp
                temp = torch.sum(preds == labels.data)
                accs.append(temp.item() / inputs.size(0))
                tbar_train.set_postfix({"Acc": f"{np.mean(accs):.3%}"})
                running_corrects += temp

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)

            # tbar_epoch.set_postfix(
            #     {f"{phase}_loss": epoch_loss, f"{phase}_acc": epoch_acc}
            # )

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                # print("saving model - best loss")
                tbar_epoch.write(f"Saving model - best loss")
                tbar_epoch.set_postfix(
                    {f"{phase}_loss": epoch_loss, f"{phase}_acc": epoch_acc}
                )
                save_model(save_path, epoch, model, optimizer, epoch_loss, epoch_acc)
            # save metrics
            if phase == "train":
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
            if phase == "val":
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

                if scheduler:
                    scheduler.step(epoch_loss)

                if early_stopping:
                    early_stopping(epoch_loss, model)
                    if early_stopping.early_stop:
                        print("Early stopping triggered!")
                        break
                # early stopping
                log_df = log_df.append(
                    pd.DataFrame(
                        {
                            "epoch": [epoch],
                            "train_loss": [train_loss_history[-1]],
                            "val_loss": [val_loss_history[-1]],
                            "train_acc": [train_acc_history[-1].cpu()],
                            "val_acc": [val_acc_history[-1].cpu()],
                        }
                    )
                )
                log_df.to_csv(log_filename)

    checkpoint = torch.load(save_path)
    print("Best val Acc: {:4f}".format(checkpoint["acc"]))
    print(f"Logs saved to {log_filename}")

    model.load_state_dict(checkpoint["model_state_dict"])

    return model
