import torch.nn as nn
from collections import OrderedDict

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

class MultiInputAggregation(nn.Module):
    def __init__(self, input_img_num, output_channels=16, input_mode='GRG'):
        super(MultiInputAggregation, self).__init__()
        # input_mode: "RGB" or "GRG". "RGB" means all the image is rgb mode. "GRG" means that the middle image remains RGB,
        # and the others will be coverted to gray. 
        if input_mode == "RGB":
            input_channels = input_img_num * 3
        elif input_mode == "GRG":
            input_channels = input_img_num + 2
        else:
            raise print("input_mode error!")
        self.conv = conv2d(input_channels, output_channels, 3)
        

    def forward(self, x):
        out = self.conv(x)
        return out