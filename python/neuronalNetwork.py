import torch
from torch import nn
from torch.autograd import Variable


class NN(nn.Module):

    def __init__(self, d_in, d_out):
        super(NN, self).__init__()
        self.inp = nn.Linear(d_in, 256)
        self.int = nn.Linear(256, 128)
        self.out = nn.Linear(128, d_out)
        self.softm = nn.LogSoftmax(dim=0)

    def forward(self, ob):
        out = self.inp(ob)
        out = self.int(out)
        out = self.out(out)
        return self.softm(out)





