from __future__ import print_function, absolute_import

from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from .tiny_imagenet import TinyImageNet


__factory = {
    'mnist': MNIST,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'tiny_imagenet': TinyImageNet,
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)
