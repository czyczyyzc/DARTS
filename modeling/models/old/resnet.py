import torch
import torchvision
import torch.nn as nn
from modeling.models.old.init import reset_parameters


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class ResNet(nn.Module):
    __factory = {
        18:  torchvision.models.resnet18,
        34:  torchvision.models.resnet34,
        50:  torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, num_classes=1000, stride=32, init_mode='kaiming_normal_out', **kwargs):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.stride = stride

        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained, num_classes=num_classes)

        if self.stride <= 16:
            self.base.conv1.stride = (1, 1)

        if not self.pretrained:
            reset_parameters(self, init_mode)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)

        if self.stride >= 16:
            x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.fc(x)
        return x


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
