import torch
from torch.autograd import Variable
import torch.nn.functional as F

# a way to define a network
class MLP(torch.nn.Module):
    def __init__(self, n_feature=5, n_hidden=200, n_output=1):
        super(NN, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
