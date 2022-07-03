import pickle

import pandas as pd
import numpy as np


def unpickle(filePath: str) -> dict:
    """
    Unpickle a file from the CIFAR-10 dataset.

    Args:
        filePath (str): Path to the file to unpickle.

    Returns:
        dict: A dictionary containing the unpickled data.

    Ref:
        https://www.cs.toronto.edu/~kriz/cifar.html
    """

    with open(filePath, "rb") as fo:
        resDict = pickle.load(fo, encoding="bytes")
    return resDict


def unpickle_cifar10() -> None:
    """Unpickle the CIFAR-10 dataset's all batches and create a dataset."""
    images = []
    labels = []
    for i in range(1, 6):
        dataBatch = unpickle(f"./cifar-10-batches-py/data_batch_{i}")
        imagesBatch = dataBatch[b"data"].copy()
        labelBatch = dataBatch[b"labels"].copy()
        imagesBatch = imagesBatch.reshape(10000, 3, 32, 32)
        images = np.concatenate(images)
        labels = np.concatenate(labels)


if __name__ == "__main__":
    unpickle_cifar10()
