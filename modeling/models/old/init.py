import math
import torch.nn as nn


def normal(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def orthogonal(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def kaiming_normal_in(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def kaiming_uniform_in(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def kaiming_normal_out(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def kaiming_uniform_out(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def xavier_normal(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def xavier_uniform(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def reset_parameters(module, init_mode='kaiming_normal_out'):
    if init_mode == 'normal':
        module.apply(normal)
    elif init_mode == 'orthogonal':
        module.apply(orthogonal)
    elif init_mode == 'kaiming_normal_in':
        module.apply(kaiming_normal_in)
    elif init_mode == 'kaiming_uniform_in':
        module.apply(kaiming_uniform_in)
    elif init_mode == 'kaiming_normal_out':
        module.apply(kaiming_normal_out)
    elif init_mode == 'kaiming_uniform_out':
        module.apply(kaiming_uniform_out)
    elif init_mode == 'xavier_normal':
        module.apply(xavier_normal)
    elif init_mode == 'xavier_uniform':
        module.apply(xavier_uniform)
    else:
        raise KeyError("Unsupported initialize mode {}.".format(init_mode))
    print("Initialization is finished!")
