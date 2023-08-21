import torch.nn as nn
import sys
sys.path.append("..")

from .convlstm_method import ConvLSTMAggregation
from .multiinput_method import MultiInputAggregation



class ImagesAggregation(nn.Module):
    def __init__(self, input_img_num=5, aggregation_output_channels=16, aggregaton_method="multiinput", input_mode="GRG"):
        super(ImagesAggregation, self).__init__()
        # aggregaton_method: "multiinput" or "convlstm". "multiinput" means MultiInput, and "convlstm" means ConvLSTM.
        # input_mode: "RGB" or "GRG". "RGB" means all the image is rgb mode. "GRG" means that the middle image remains RGB,
        # and the others will be coverted to gray.
        if aggregaton_method=="multiinput":
            self.images_fusion_module = MultiInputAggregation(input_img_num=input_img_num, output_channels=aggregation_output_channels, input_mode=input_mode)
        elif aggregaton_method=="convlstm":
            if input_mode=="GRG":
                raise("Error! When the aggregation methord is 'convlstm', the input mode must be 'RGB'.")
            self.images_fusion_module = ConvLSTMAggregation(input_img_num=input_img_num, output_channels=aggregation_output_channels)
        else:
            raise("fusion_method error!")
        

    def forward(self, x):#
        fusion_img = self.images_fusion_module(x)
        return fusion_img