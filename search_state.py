from numpy import dtype

import agent
import random
import math
import variables
import torch

EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500

def select_action(state):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * variables.epoch / EPS_DECAY)
    sample = random.random
    if (sample < eps_threshold) :
        # Return chosen state
        with torch.no_grad():
            return variables.policy_net(state).max(1).indices.view(1, 1)
    return torch.tensor([[agent.env.action_space.sample()]], device = variables.device, dtype = torch.long)

