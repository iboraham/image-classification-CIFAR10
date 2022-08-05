import pickle

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


def unpickle_cifar10(verbose: bool = False) -> None:
    """Unpickle the CIFAR-10 dataset's all batches and create a dataset."""
    images = np.empty((50000, 32, 32, 3), dtype=np.uint8)
    labels = np.empty((50000,))
    counter = 0
    for i in range(1, 6):
        dataBatch = unpickle(f"./cifar-10-batches-py/data_batch_{i}")
        imagesBatch = dataBatch[b"data"].copy()
        labelBatch = dataBatch[b"labels"].copy()
        imagesBatch = imagesBatch.reshape(-1, 3, 32, 32)
        imagesBatch = imagesBatch.transpose(0, 2, 3, 1)
        images[counter : counter + 10000] = imagesBatch
        labels[counter : counter + 10000] = labelBatch
        counter += 1
    if verbose:
        print(f"{counter} batches unpickled")
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
    return images, labels


def unpickle_cifar10_test(verbose: bool = False) -> None:
    """Unpickle the CIFAR-10 test dataset and create a dataset."""
    dataBatch = unpickle(f"./cifar-10-batches-py/test_batch")
    images = dataBatch[b"data"].copy()
    labels = dataBatch[b"labels"].copy()
    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    if verbose:
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
    return images, labels


if __name__ == "__main__":
    # continue from this if you want to unpickle the train, test set
    unpickle_cifar10()
