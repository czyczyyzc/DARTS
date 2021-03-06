from __future__ import print_function, absolute_import

from modeling.models.cnn import CNN_CIFAR, CNN_ImageNet
from modeling.models.cnn_search import CNN_Search

__factory = {
    'cnn_cifar': CNN_CIFAR,
    'cnn_search': CNN_Search,
    'cnn_imagenet': CNN_ImageNet,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
