import numpy as np
import torch
import torch.nn as nn

DROP_RATE = 0.1
DIM = 0

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_sie, word_embedding):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_embedding = word_embedding
        # print(input_size)
        self.U = nn.Linear(input_size, hidden_size)
        # self.U.weight = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.W = nn.Linear(hidden_size, hidden_size)
        # self.W.weight = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.V = nn.Linear(hidden_size, output_sie)
        # self.V.weight = nn.Parameter(torch.zeros(hidden_size, output_sie))
        self.dropout = nn.Dropout(DROP_RATE)
        self.softmax = nn.LogSoftmax(dim=DIM)

    def lookup_embedding(self, index):
        return self.word_embedding(index)

    def forward(self, input, hidden):
        # tens_input = torch.tensor(input, dtype=torch.float32)
        # print(type(torch.t(tens_input)), torch.t(tens_input).size())
        # print(input_ind.item())
        # input = self.lookup_embedding(input_ind)
        # print(type(input), input.size())
        hid1 = self.U(input)
        # print(type(hid), hid.size())
        # hidden1 = self.U(tens_input) + self.W(hidden)
        # hidden1 = torch.add(torch.matmul(torch.t(self.U.weight), tens_input), torch.matmul(torch.t(self.W.weight), hidden))
        # print(torch.t(hidden).size())
        hid2 = self.W(hidden)
        hid3 = torch.add(hid1, hid2)
        hidden2_St = torch.tanh(hid3)
        # print(type(hidden2_St), hidden2_St.size())

        # output = self.V(hidden2_St)
        output = self.V(hidden2_St)
        output = self.dropout(output)

        # output = self.softmax(output)
        # print(list(output), output.size())
        # output = torch.t(output)

        # outputoutput_ind = output[0].max(0)[1]

        # print(output_ind)
        return output, hidden2_St

    def initHidden(self):
        return torch.rand(1, self.hidden_size)

