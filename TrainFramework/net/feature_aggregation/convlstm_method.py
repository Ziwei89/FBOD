import torch.nn as nn
import sys
sys.path.append("..")
from ..module_net.ConvLSTM import ConvLSTM

class ConvLSTMAggregation(nn.Module):
    def __init__(self, input_img_num, output_channels=16):
        super(ConvLSTMAggregation, self).__init__()
        self.input_img_num = input_img_num
        self.convlstm = ConvLSTM(3,output_channels)

    def forward(self, x):
        # input has self.input_img_num * 3 channels, by concatting self.input_img_num images
        # data size is [batch, channels, height, width]
        # chunk across channels dimension
        x_ss = x.chunk(self.input_img_num,1)
        state = None
        for x_s in x_ss:
            state = self.convlstm(x_s,state)
        out =  state[0]
        return out