import torch
import random
import torch.nn as nn
import torch.nn.functional as F
class agent(nn.Module):
  def __init__(self):
    super(agent,self).__init__()
    self.input = nn.Linear(8,64)
    self.layer1 = nn.Linear(64,128)
    self.layer2 = nn.Linear(128, 64)
    self.output = nn.Linear(64,4)
  def forward(self,x):
    x = F.relu(self.input(x))
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = self.output(x)
    return x
  def choice(self,state,epsilon,temp):
    if random.random()>epsilon:
      with torch.no_grad():
        q_values = self.forward(state)
        # softmax with temperature
        probs = torch.softmax(q_values / temp, dim=-1)
        return torch.multinomial(probs, 1).item()
    else:
      return random.randint(0,3)