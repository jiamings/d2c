import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from torch.utils.data import Subset, TensorDataset
import numpy as np


def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    train_samples = np.load(args.train_fname)
    train_labels = np.zeros(len(train_samples))

    data_mean = np.mean(train_samples, axis=(0, 2, 3), keepdims=True)
    data_std = np.std(train_samples, axis=(0, 2, 3), keepdims=True)

    train_samples = (train_samples - data_mean)/data_std
    print("train data shape are - ", train_samples.shape, train_labels.shape)
    print("train data stats are - ", np.mean(train_samples), np.std(train_samples), 
        np.min(train_samples), np.max(train_samples))

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_samples).float(), torch.from_numpy(train_labels).float())

    return dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
