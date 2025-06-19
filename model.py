import torch
import torch.nn as nn
from torch.optim import Adam


device = torch.device("cpu")

class model(nn.Module):
  def __init__(self):
    super(model, self).__init__()
    self.input_layer = nn.Linear(10, 10)
    self.linear = nn.Linear(10, 1)
    self.sigmoid = nn.Sigmoid()
  def forward(self, x):
    x = self.input_layer(x)
    x = self.linear(x)
    x = self.sigmoid(x)
    return x

#criterion = nn.BCELoss().to(device)
#parameters = model.parameters()
#optimizer = Adam(parameters, lr = 1e-4)