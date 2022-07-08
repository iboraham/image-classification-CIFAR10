import argparse
from distutils.command.config import config

from torch.utils.data import DataLoader, random_split

from data import cifar10_dataset
from data.transformations import get_test_transformations, get_train_transformations
from download.unpickle_cifar10 import unpickle_cifar10, unpickle_cifar10_test
from model import cifar10_model
import tez
from tez.callbacks import EarlyStopping

ES = EarlyStopping(
    monitor="valid_rmse",
    model_path=f"saved_model/model_f{0}.bin",
    patience=3,
    mode="min",
    save_weights_only=True,
)

CONFIG = tez.TezConfig(
    training_batch_size=32,
    validation_batch_size=64,
    epochs=10,
    device="cpu",
    step_scheduler_after="epoch",
    step_scheduler_metric="valid_rmse",
    fp16=True,
)


def main(args):
    # Unpickle the CIFAR-10 dataset
    images, labels = unpickle_cifar10()

    # Create a dataset object
    train_dataset = cifar10_dataset(
        images, labels, transforms=get_train_transformations()
    )

    # Unpickle test dataset
    test_images, test_labels = unpickle_cifar10_test()
    valid_dataset = cifar10_dataset(
        test_images, test_labels, transforms=get_test_transformations()
    )

    model = cifar10_model(num_classes=10)
    model = tez.Tez(model)

    # Train the model
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        config=CONFIG,
        callbacks=[ES],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args)
