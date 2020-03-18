import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = 16
        self.i2h = nn.Linear(input_size + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(input_size + self.hidden_size, output_size)

        self.i2h_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.i2o_1 = nn.Linear(self.hidden_size, output_size)

        self.i2h_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.i2o_2 = nn.Linear(self.hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)

        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)