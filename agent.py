import torch.nn as nn

num_observations = 658
num_actions = 20
SHAPE = [num_observations, 30, 30, num_actions] # Initial 658, Ending 20 are fixed
ACTIVATION = nn.functional.relu

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.layers = nn.ModuleList()
        for a in range(len(SHAPE) - 1):
            self.layers.append(nn.Linear(SHAPE[a], SHAPE[a + 1]))

    def forward(self, x):
        ### x = input value array
        for layer in self.layers[:-1]:
            x = ACTIVATION(layer(x))
        return self.layers[-1](x)
