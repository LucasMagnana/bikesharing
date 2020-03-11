import torch
from torch import nn
from torch.autograd import Variable


class NN(nn.Module):

    def __init__(self, d_in, d_out):
        super(NN, self).__init__()
        self.inp = nn.Linear(d_in, 32)
        self.int = nn.Linear(32, 16)
        self.out = nn.Linear(16, d_out)

    def forward(self, ob):
        return self.out(nn.functional.relu(self.int(nn.functional.relu(self.inp(ob)))))

