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

def search_state():
    state, info = agent.env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=variables.device).unsqueeze(0)
    terminated = False
    truncated = False
    while terminated == False and truncated == False:
        action = select_action(state)
        observation, reward, terminated, truncated, _ = agent.env.step(action.item())
        reward = torch.tensor([reward], device = variables.device)
        if (terminated) :
            next_state = None
        else :
            next_state = torch.tensor(observation, dtype=torch.float32, device=variables.device).unsqueeze(0)
        variables.memory.append(variables.Transition(state, action, next_state, reward))
        state = next_state
