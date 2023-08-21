"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math

__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

def movilenetv2_layer(input_channel, output_channel, s, n, t):
    block = InvertedResidual
    layers = []
    for i in range(n):
        layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
        input_channel = output_channel
    return nn.Sequential(*layers)

class featureExtract_MobilenetV2(nn.Module):
    def __init__(self, input_channels=16, width_mult=1.):
        super(featureExtract_MobilenetV2, self).__init__()

        # building first layer
        output_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        self.movilenetv2_layer_0 = conv_3x3_bn(input_channels, output_channel, 2)
        input_channel = output_channel
        # building inverted residual blocks

        output_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
                                                                                   #s, n, t
        self.movilenetv2_layer_1 = movilenetv2_layer(input_channel, output_channel, 1, 1, 1)
        input_channel = output_channel

        output_channel = _make_divisible(64 * width_mult, 4 if width_mult == 0.1 else 8)
                                                                                   #s, n, t
        self.movilenetv2_layer_2 = movilenetv2_layer(input_channel, output_channel, 2, 2, 6)
        input_channel = output_channel

        output_channel = _make_divisible(128 * width_mult, 4 if width_mult == 0.1 else 8)
                                                                                   #s, n, t
        self.movilenetv2_layer_3 = movilenetv2_layer(input_channel, output_channel, 2, 3, 6)
        input_channel = output_channel

        output_channel = _make_divisible(256 * width_mult, 4 if width_mult == 0.1 else 8)
                                                                                   #s, n, t
        self.movilenetv2_layer_4 = movilenetv2_layer(input_channel, output_channel, 2, 4, 6)
        input_channel = output_channel

        output_channel = _make_divisible(256 * width_mult, 4 if width_mult == 0.1 else 8)
                                                                                   #s, n, t
        self.movilenetv2_layer_5 = movilenetv2_layer(input_channel, output_channel, 1, 3, 6)
        input_channel = output_channel

        output_channel = _make_divisible(512 * width_mult, 4 if width_mult == 0.1 else 8)
                                                                                   #s, n, t
        self.movilenetv2_layer_6 = movilenetv2_layer(input_channel, output_channel, 2, 3, 6)
        input_channel = output_channel

        output_channel = _make_divisible(512 * width_mult, 4 if width_mult == 0.1 else 8)
                                                                                   #s, n, t
        self.movilenetv2_layer_7 = movilenetv2_layer(input_channel, output_channel, 1, 1, 6)
        input_channel = output_channel

        # building last several layers
        # output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        output_channel = _make_divisible(512 * width_mult, 4 if width_mult == 0.1 else 8)
        self.conv = conv_1x1_bn(input_channel, output_channel)

        self._initialize_weights()

    def forward(self, x):
        # input has 7 channels, by concatting 1 image and 4 gray images
        # data size is [batch, channel, height, width]
        # output0_size = input_size/2
        out0 = self.movilenetv2_layer_0(x)

        # output1_size = input_size/2
        out1 = self.movilenetv2_layer_1(out0)

        # output2_size = input_size/4
        out2 = self.movilenetv2_layer_2(out1)

        # output3_size = input_size/8
        out3 = self.movilenetv2_layer_3(out2)

        # output4_size = input_size/16
        out4 = self.movilenetv2_layer_4(out3)

        # output5_size = input_size/16
        out5 = self.movilenetv2_layer_5(out4)

        # output6_size = input_size/32
        out6 = self.movilenetv2_layer_6(out5)

        # output7_size = input_size/32
        out7 = self.movilenetv2_layer_7(out6)

        out8 = self.conv(out7)
        return out1, out2, out3, out5, out8

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def feature_extract_mobilenetv2(input_channels=16, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    input_channels=input_channels
    return featureExtract_MobilenetV2(input_channels, **kwargs)

