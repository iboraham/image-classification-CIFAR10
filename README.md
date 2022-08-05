# Image Classification with CIFAR 10 Dataset

## Introduction

This very beginner friendly deep learning project for those trying to get into the world of deep learning. We are using the CIFAR 10 dataset to train a convolutional neural network to classify images.

## Dataset

Cifar 10 is a dataset of 60,000 32x32 color images in 10 classes. Each image is labeled with one of the 10 digits. It is a subset of the [ImageNet dataset](http://www.image-net.org/).

### Downloading the dataset

To download the dataset, you can use the following command:

```bash
sh download/download_cifar10.sh
```

### Unpickling the dataset

To unpickle the dataset, you can use the following command:

```bash
python data/unpickle_cifar10.py
```

## Usage
---
### Training

To train the network, you can use the following command and retrain model:

```bash
python main.py
```

Model will be saved as `model.pth` in the current directory. 

Ps. There is no current support for training model on starting from current checkpoint, but it'll be easy to add.

### Testing

To test the network and reproduce the results, you can use the following command:

```bash
python test.py
```

- Results:


## :construction: !Work in progress!