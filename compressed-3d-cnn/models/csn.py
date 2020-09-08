'''Channel Separated Convolutional Network (CSN) in PyTorch.

See the paper "Video Classification with Channel-Separated Convolutional Networks" for more details.
'''


import math
import torch
import torch.nn as nn


class CSNBottleneck(nn.Module):
    expansion = 4

    # NOTE: could add a downsample arg to this init? - same as in ResNet
    def __init__(self, in_channels, channels, stride=1, mode='ip'):
        super().__init__()

        assert mode in ['ip', 'ir']
        self.mode = mode

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

        conv2 = []
        # if mode = 'ip' then replace Conv3d with Conv1d + DepthwiseConv3d
        if self.mode == 'ip':
            conv2.append(nn.Conv3d(channels, channels, kernel_size=1, stride=1, bias=False))
        # if mode = 'ir' then just DepthwiseConv3d - same as seen in MobileNet
        conv2.append(nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=channels))
        # NOTE: added a nn.Flatten() here for testing
        # conv2.append(nn.Flatten())

        self.conv2 = nn.Sequential(*conv2)

        self.bn2 = nn.BatchNorm3d(channels)

        self.conv3 = nn.Conv3d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channels * self.expansion)

        # NOTE: having problems with nn.Sequential, do they all require this?
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels * self.expansion),
                nn.Flatten()
            )


    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CSN(nn.Module):
    def __init__(self, block, layers, num_classes, sample_size, sample_duration,
                 mode='ip'):

        super().__init__()

        assert mode in ['ip', 'ir']
        self.mode = mode

        self.in_channels = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # self.avgpool = nn.AdaptiveAvgPool3d(1)
        # NOTE: adaptive pool replaced with args definition below
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = []
        layers.append(block(self.in_channels, channels, stride, mode=self.mode))
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels, mode=self.mode))

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def csn26(num_classes, sample_size, sample_duration, mode='ip'):
    return CSN(CSNBottleneck, [1,2,4,1], num_classes=num_classes,
     sample_size=sample_size, sample_duration=sample_duration, mode=mode)


def csn50(num_classes, sample_size, sample_duration, mode='ip'):
    return CSN(CSNBottleneck, [3,4,6,3], num_classes=num_classes,
     sample_size=sample_size, sample_duration=sample_duration, mode=mode)


def csn101(num_classes, sample_size, sample_duration, mode='ip'):
    return CSN(CSNBottleneck, [3,4,23,3], num_classes=num_classes,
     sample_size=sample_size, sample_duration=sample_duration, mode=mode)


def csn152(num_classes, sample_size, sample_duration, mode='ip'):
    return CSN(CSNBottleneck, [3,8,36,3], num_classes=num_classes,
     sample_size=sample_size, sample_duration=sample_duration, mode=mode)
