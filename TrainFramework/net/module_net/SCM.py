import torch
import torch.nn as nn

class SCMLayer(nn.Module):
    def __init__(self, channel1, channel2):
        super(SCMLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # channel attention H,W->1*1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        concat_channels = channel1 + channel2
        self.fc_layer = nn.Linear(concat_channels, concat_channels)
        self.conv1_1 = nn.Conv2d(concat_channels, concat_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3_3 = nn.Conv2d(concat_channels, concat_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, ll_feature, hl_feature):
        UP_feature = self.upsample(ll_feature)
        CONCAT_out = torch.cat([UP_feature,hl_feature],axis=1)
        GAP_out = self.avg_pool(CONCAT_out)
        FC_out = self.fc_layer(GAP_out)
        Mul_out = torch.mul(CONCAT_out, FC_out)
        out = self.conv1_1(Mul_out)
        out = self.conv3_3(out)
        return out