import torch
import torch.nn as nn

class SCMLayer(nn.Module):
    def __init__(self, channels):
        super(SCMLayer, self).__init__()
        # channel attention H,W->1*1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Linear(channels, channels)
        self.conv1_1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3_3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        GAP_out = self.avg_pool(x)
        GAP_out = GAP_out.view(GAP_out.size(0), -1)
        FC_out = self.fc_layer(GAP_out)
        FC_out = FC_out.view(FC_out.size(0), -1, 1, 1)
        
        Mul_out = torch.mul(x, FC_out)
        out = self.conv1_1(Mul_out)
        out = self.conv3_3(out)
        return out