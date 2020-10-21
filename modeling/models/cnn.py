import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import OPS, FactorizedReduce, ReLUConvBN, Identity
from .genotypes import Genotype


class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, norm_layer=nn.BatchNorm2d):
        super(Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, norm_layer=norm_layer)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, norm_layer=norm_layer)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, norm_layer=norm_layer)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction, norm_layer)

    def _compile(self, C, op_names, indices, concat, reduction, norm_layer):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True, norm_layer=norm_layer)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = F.dropout(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = F.dropout(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes, norm_layer=nn.BatchNorm2d):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            norm_layer(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, C, num_classes, norm_layer=nn.BatchNorm2d):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            norm_layer(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class CNN_CIFAR(nn.Module):
    def __init__(self, num_classes, init_channels, layers, genotype, auxiliary, drop_prob=0.2, norm_layer=nn.BatchNorm2d):
        super(CNN_CIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_prob = drop_prob

        stem_multiplier = 3
        C_curr = stem_multiplier * init_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            norm_layer(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, init_channels
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, norm_layer=norm_layer)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes, norm_layer=norm_layer)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, data):
        logits_aux = None
        s0 = s1 = self.stem(data)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class CNN_ImageNet(nn.Module):
    def __init__(self, num_classes, init_channels, layers, genotype, auxiliary, drop_prob=0.2, norm_layer=nn.BatchNorm2d):
        super(CNN_ImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_prob = drop_prob

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, init_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(init_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_channels // 2, init_channels, 3, stride=2, padding=1, bias=False),
            norm_layer(init_channels),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(init_channels, init_channels, 3, stride=2, padding=1, bias=False),
            norm_layer(init_channels),
        )

        C_prev_prev, C_prev, C_curr = init_channels, init_channels, init_channels

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, norm_layer=norm_layer)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes, norm_layer=norm_layer)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, data):
        logits_aux = None
        s0 = self.stem0(data)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

