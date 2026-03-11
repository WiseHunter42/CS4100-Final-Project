from matplotlib.hatch import Shapes
from pettingzoo.classic import hanabi_v5
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

SHAPE = [658, 30, 30, 20] # Initial 658, Ending 20 are fixed
ACTIVATION = nn.functional.relu

env = hanabi_v5.env(
    render_mode="human"
    # additional parameters here if we want to change them
    # e.g. num_players=4, colors=5, ranks=5, hand_size=4, etc.
)
env.reset(seed=42)

class network(nn.Model):
    def __init__(self):
        super(network, self).__init__()
        self.layers = []
        for a in range(len(SHAPE) - 1):
            self.layers.append(nn.Linear(SHAPE[a], SHAPE[a + 1]))

    def forward(self, x):
        ### x = input value array
        for layer in self.layers:
            x = ACTIVATION(layer(x))
