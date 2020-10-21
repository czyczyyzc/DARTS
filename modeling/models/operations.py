import torch
import torch.nn as nn


OPS = {
    'none': lambda C, stride, affine, norm_layer: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine, norm_layer: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine, norm_layer: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine, norm_layer: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine, norm_layer=norm_layer),
    'sep_conv_3x3': lambda C, stride, affine, norm_layer: SepConv(C, C, 3, stride, 1, affine=affine, norm_layer=norm_layer),
    'sep_conv_5x5': lambda C, stride, affine, norm_layer: SepConv(C, C, 5, stride, 2, affine=affine, norm_layer=norm_layer),
    'sep_conv_7x7': lambda C, stride, affine, norm_layer: SepConv(C, C, 7, stride, 3, affine=affine, norm_layer=norm_layer),
    'dil_conv_3x3': lambda C, stride, affine, norm_layer: DilConv(C, C, 3, stride, 2, 2, affine=affine, norm_layer=norm_layer),
    'dil_conv_5x5': lambda C, stride, affine, norm_layer: DilConv(C, C, 5, stride, 4, 2, affine=affine, norm_layer=norm_layer),
    'conv_7x1_1x7': lambda C, stride, affine, norm_layer: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        norm_layer(C, affine=affine)
    ),
}


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, norm_layer=nn.BatchNorm2d):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            norm_layer(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, norm_layer=nn.BatchNorm2d):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            norm_layer(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, norm_layer=nn.BatchNorm2d):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            norm_layer(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            norm_layer(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True, norm_layer=nn.BatchNorm2d):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = norm_layer(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

