import agent
import random
import math
import variables
import torch

def select_action(state):
    eps_threshold = variables.eps_end + (variables.eps_start - variables.eps_end) * math.exp(-1. * variables.epoch / variables.eps_decay)
    sample = random.random()
    if (sample > eps_threshold) :
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
