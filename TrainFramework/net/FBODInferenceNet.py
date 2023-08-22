import torch.nn as nn
from collections import OrderedDict
import sys
sys.path.append("..")
from .feature_aggregation.feature_aggregation import ImagesAggregation
from .feature_extraction.feature_extraction import FeatureExtraction, FeatureExtraction_MultiOutput

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


#---------------------------------------------------#
#   output
#---------------------------------------------------#
def FBODetection_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   FBOInferenceNet
#---------------------------------------------------#
class FBODInferenceBody(nn.Module):
    def __init__(self, input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", ### Aggreagation parameters.
                       backbone_name="cspdarknet53", fusion_method="concat"): ### Extract parameters. input_channels equal to aggregation_output_channels.
        super(FBODInferenceBody, self).__init__()
        """
        aggregation_method: "multiinput" or "convlstm". "multiinput" means MultiInput, and "convlstm" means ConvLSTM.
        input_mode:         "RGB" or "GRG". "RGB" means all the image is rgb mode. "GRG" means that the middle image remains RGB,
                             and the others will be coverted to gray.
        backbone_name:      "cspdarknet53" or "mobilenetv2".
        fusion_method:      "concat" or "scm".
        """
        self.aggregate_features = ImagesAggregation(input_img_num=input_img_num, aggregation_output_channels=aggregation_output_channels,
                                                    aggregaton_method=aggregation_method, input_mode=input_mode)
        ### The input_channels of feature extraction net is equal to aggregation_output_channels.
        self.extract_features = FeatureExtraction(backbone_name=backbone_name, input_channels=aggregation_output_channels, fusion_method=fusion_method)

        self.FBODetection_head_conf = FBODetection_head([32, 1],32)
        self.FBODetection_head_pos = FBODetection_head([32, 4],32)


    def forward(self, x):

        out0 = self.aggregate_features(x) # output channels = 16
        out1 = self.extract_features(out0) # output channels = 32

        conf = self.FBODetection_head_conf(out1) # output channels = 1
        pos = self.FBODetection_head_pos(out1) # output channels = 4

        return conf, pos

#---------------------------------------------------#
#   FBOInferenceNet_MultiScale
#---------------------------------------------------#
class FBODInferenceBody_MultiScale(nn.Module):
    def __init__(self, input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", ### Aggreagation parameters.
                       backbone_name="cspdarknet53", fusion_method="concat"): ### Extract parameters. input_channels equal to aggregation_output_channels.
        super(FBODInferenceBody_MultiScale, self).__init__()
        """
        aggregation_method: "multiinput" or "convlstm". "multiinput" means MultiInput, and "convlstm" means ConvLSTM.
        input_mode:         "RGB" or "GRG". "RGB" means all the image is rgb mode. "GRG" means that the middle image remains RGB,
                             and the others will be coverted to gray.
        backbone_name:      "cspdarknet53" or "mobilenetv2".
        fusion_method:      In multi-scale only "concat".
        """
        self.aggregate_features = ImagesAggregation(input_img_num=input_img_num, aggregation_output_channels=aggregation_output_channels,
                                                    aggregaton_method=aggregation_method, input_mode=input_mode)
        ### The input_channels of feature extraction net is equal to aggregation_output_channels.
        self.extract_features = FeatureExtraction_MultiOutput(backbone_name=backbone_name, input_channels=aggregation_output_channels, fusion_method=fusion_method)

        self.FBODetection_head_conf1 = FBODetection_head([512, 1],512)
        self.FBODetection_head_pos1 = FBODetection_head([512, 4],512)

        self.FBODetection_head_conf2 = FBODetection_head([128, 1],128)
        self.FBODetection_head_pos2 = FBODetection_head([128, 4],128)

        self.FBODetection_head_conf3 = FBODetection_head([32, 1],32)
        self.FBODetection_head_pos3 = FBODetection_head([32, 4],32)


    def forward(self, x):

        out0 = self.aggregate_features(x) # output channels = 16
        P5, P3, P1 = self.extract_features(out0) # output channels = 512, 128, 32

        conf1 = self.FBODetection_head_conf1(P5) # output channels = 1, feature_size = input_size/32
        pos1 = self.FBODetection_head_pos1(P5) # output channels = 4, feature_size = input_size/32

        conf2 = self.FBODetection_head_conf2(P3) # output channels = 1, feature_size = input_size/8
        pos2 = self.FBODetection_head_pos2(P3) # output channels = 4, feature_size = input_size/8

        conf3 = self.FBODetection_head_conf3(P1) # output channels = 1, feature_size = input_size/2
        pos3 = self.FBODetection_head_pos3(P1) # output channels = 4, feature_size = input_size/2

        return [conf1, pos1], [conf2, pos2], [conf3, pos3]