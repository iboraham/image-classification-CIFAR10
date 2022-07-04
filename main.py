from data import cifar10_dataset
from data.transformations import get_train_transformations, get_test_transformations
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from download.unpickle_cifar10 import unpickle_cifar10
from model import cifar10_model


def main():
    # Unpickle the CIFAR-10 dataset
    images, labels = unpickle_cifar10()

    # Split the data into training and validation sets
    train_dataset, test_dataset = random_split(dataset, [40000, 10000])

    # Create a dataset object
    dataset = cifar10_dataset(images, labels, transforms=get_train_transformations())

    # Create a dataloader object
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create a dataloader object for the training set
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Iterate over the data
    for batch in dataloader:
        print(batch["image"].shape)
        print(batch["label"].shape)
        break

    # Get the transformations for the training and test sets
    # train_transformations = get_train_transformations()
    # test_transformations = get_test_transformations()

    # Train the model
    # cifar10_model.fit(
    #     train_dataset,
    #     valid_dataset=valid_dataset,
    #     config=config,
    #     callbacks=[es],
    # )
