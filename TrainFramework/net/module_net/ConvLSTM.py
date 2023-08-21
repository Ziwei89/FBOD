import torch
import torch.nn as nn
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
# cuda = False

class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ConvLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=1)
        self.weightC = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.weightI = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.weightF = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.weightO = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
 
    def forward(self, input_, prev_state):
 
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
 
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if cuda:
                prev_state = (
                    Variable(torch.zeros(state_size, device=torch.device('cuda'))),
                    Variable(torch.zeros(state_size, device=torch.device('cuda')))
            )
            else:
                prev_state = (
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                )
 
        prev_hidden, prev_cell = prev_state
 
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
 
        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
 
        # apply sigmoid non linearity
        in_gate = torch.sigmoid(self.weightI(in_gate))
        remember_gate = torch.sigmoid(self.weightF(remember_gate))
        out_gate = torch.sigmoid(self.weightO(out_gate))
 
        # apply tanh non linearity
        cell_gate = torch.tanh(self.weightC(cell_gate))
 
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
 
        return hidden, cell