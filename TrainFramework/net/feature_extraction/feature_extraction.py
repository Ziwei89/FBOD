import sys
sys.path.append("..")
import torch
import torch.nn as nn
from collections import OrderedDict
from .feature_extract_CSPdarknet import feature_extract_darknet53
from .feature_extract_MobilenetV2 import feature_extract_mobilenetv2
from ..module_net.SCM_after_cat import SCMLayer

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

class FusionModule(nn.Module):
    def __init__(self, channels, fusion_method="concat"):
        super(FusionModule, self).__init__()
        ### fusion_method: scm or concat
        semi_channels = int(channels/2)
        self.conv_for_px = conv2d(channels, semi_channels, 1)
        if not (fusion_method=="scm" or fusion_method=="concat"):
            raise("Error! fusion_method error.")
        self.fusion_method = fusion_method
        self.scm = SCMLayer(channels=channels)
        self.make_five_conv = make_five_conv([semi_channels, channels],channels)

    def forward(self, x, Pxp_upsampe):
        Px = self.conv_for_px(x)
        Px = torch.cat([Px,Pxp_upsampe],axis=1)
        if self.fusion_method=="scm":
            Px = self.scm(Px)
        Px = self.make_five_conv(Px)
        return Px

def build_backbone(backbone_name):
    if backbone_name == "cspdarknet53":
        backbone_net = feature_extract_darknet53
    elif backbone_name == "mobilenetv2":
        backbone_net = feature_extract_mobilenetv2
    else:
        raise print("Error! No such model structure:{}.".format(backbone_name))
    return backbone_net

class FeatureExtraction(nn.Module):
    def __init__(self, backbone_name, input_channels, fusion_method="concat"):
        super(FeatureExtraction, self).__init__()
        self.backbone = build_backbone(backbone_name)(input_channels=input_channels)
        self.conv1 = make_three_conv([512,1024],1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512,1024],2048)

        self.upsample1 = Upsample(512,256)
        self.fusion1 = FusionModule(512, fusion_method=fusion_method)

        self.upsample2 = Upsample(256,128)
        self.fusion2 = FusionModule(256, fusion_method=fusion_method)

        self.upsample3 = Upsample(128,64)
        self.fusion3 = FusionModule(128, fusion_method=fusion_method)

        self.upsample4 = Upsample(64,32)
        self.fusion4 = FusionModule(64, fusion_method=fusion_method)
        

    def forward(self, x):
        #  backbone
        ## x4 channels = 64, x3 channels = 128, x2 channels = 256, x1 channels = 512, x0 channels = 1024,
        c5, c4, c3, c2, c1 = self.backbone(x)

        P5 = self.conv1(c1)  # channels = 512
        P5 = self.SPP(P5)  # channels = 1024
        P5 = self.conv2(P5)  ## input_size/32  # channels = 512

        P5_upsample = self.upsample1(P5)  # channels = 256
        P4 = self.fusion1(c2, P5_upsample) # output channels = 256

        P4_upsample = self.upsample2(P4)  # channels = 128
        P3 = self.fusion2(c3, P4_upsample) # output channels = 128

        P3_upsample = self.upsample3(P3)  # channels = 64
        P2 = self.fusion3(c4, P3_upsample) # output channels = 64

        P2_upsample = self.upsample4(P2)  # channels = 32
        P1 = self.fusion4(c5, P2_upsample) # output channels = 32

        return P1

class FeatureExtraction_MultiOutput(nn.Module):
    def __init__(self, backbone_name, input_channels, fusion_method="concat"):
        super(FeatureExtraction_MultiOutput, self).__init__()
        if fusion_method != "concat":
            raise("Error! In multi-output, the fusion must be 'concat'.")
        self.backbone = build_backbone(backbone_name)(input_channels=input_channels)
        self.conv1 = make_three_conv([512,1024],1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512,1024],2048)

        self.upsample1 = Upsample(512,256)
        self.fusion1 = FusionModule(512, fusion_method=fusion_method)

        self.upsample2 = Upsample(256,128)
        self.fusion2 = FusionModule(256, fusion_method=fusion_method)

        self.upsample3 = Upsample(128,64)
        self.fusion3 = FusionModule(128, fusion_method=fusion_method)

        self.upsample4 = Upsample(64,32)
        self.fusion4 = FusionModule(64, fusion_method=fusion_method)
        

    def forward(self, x):
        #  backbone
        ## x4 channels = 64, x3 channels = 128, x2 channels = 256, x1 channels = 512, x0 channels = 1024,
        c5, c4, c3, c2, c1 = self.backbone(x)

        P5 = self.conv1(c1)  # channels = 512
        P5 = self.SPP(P5)  # channels = 1024
        P5 = self.conv2(P5)  ## input_size/32  # channels = 512

        P5_upsample = self.upsample1(P5)  # channels = 256
        P4 = self.fusion1(c2, P5_upsample) # output channels = 256

        P4_upsample = self.upsample2(P4)  # channels = 128
        P3 = self.fusion2(c3, P4_upsample) # output channels = 128

        P3_upsample = self.upsample3(P3)  # channels = 64
        P2 = self.fusion3(c4, P3_upsample) # output channels = 64

        P2_upsample = self.upsample4(P2)  # channels = 32
        P1 = self.fusion4(c5, P2_upsample) # output channels = 32

        return P5, P3, P1

class FeatureExtraction_old(nn.Module):
    def __init__(self, backbone_name, input_channels):
        super(FeatureExtraction, self).__init__()
        self.backbone = build_backbone(backbone_name)(input_channels=input_channels)
        self.conv1 = make_three_conv([512,1024],1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512,1024],2048)

        self.upsample1 = Upsample(512,256)
        self.conv_for_P4 = conv2d(512,256,1)
        self.make_five_conv1 = make_five_conv([256, 512],512)

        self.upsample2 = Upsample(256,128)
        self.conv_for_P3 = conv2d(256,128,1)
        self.make_five_conv2 = make_five_conv([128, 256],256)

        self.upsample3 = Upsample(128,64)
        self.conv_for_P2 = conv2d(128,64,1)
        self.make_five_conv3 = make_five_conv([64, 128],128)

        self.upsample4 = Upsample(64,32)
        self.conv_for_P1 = conv2d(64,32,1)
        self.make_five_conv4 = make_five_conv([32, 64],64)
        

    def forward(self, x):
        #  backbone
        ## x4 channels = 64, x3 channels = 128, x2 channels = 256, x1 channels = 512, x0 channels = 1024,
        x4, x3, x2, x1, x0 = self.backbone(x)

        P5 = self.conv1(x0)  # channels = 512
        P5 = self.SPP(P5)  # channels = 1024
        P5 = self.conv2(P5)  ## input_size/32  # channels = 512

        P5_upsample = self.upsample1(P5)  # channels = 256
        P4 = self.conv_for_P4(x1)  # channels = 256
        P4 = torch.cat([P4,P5_upsample],axis=1)  # channels = 512
        P4 = self.make_five_conv1(P4)  ## input_size/16  # channels = 256

        P4_upsample = self.upsample2(P4)  # channels = 128
        P3 = self.conv_for_P3(x2)  # channels = 128
        P3 = torch.cat([P3,P4_upsample],axis=1)  # channels = 256
        P3 = self.make_five_conv2(P3)  ## input_size/8  # channels = 128

        P3_upsample = self.upsample3(P3)  # channels = 64
        P2 = self.conv_for_P2(x3)  # channels = 64
        P2 = torch.cat([P2,P3_upsample],axis=1)  # channels = 128
        P2 = self.make_five_conv3(P2)  ## input_size/4  # channels = 64

        P2_upsample = self.upsample4(P2)  # channels = 32
        P1 = self.conv_for_P1(x4)  # channels = 32
        P1 = torch.cat([P1,P2_upsample],axis=1)  # channels = 64
        out = self.make_five_conv4(P1)  ## input_size/2  # channels = 32

        return out