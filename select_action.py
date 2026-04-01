
import random
import math
import variables
import torch

def select_action(state, action_mask, env, agent):
    eps_threshold = variables.eps_end + (variables.eps_start - variables.eps_end) * math.exp(-1. * variables.episode / variables.eps_decay)
    sample = random.random()
    if (sample > eps_threshold) :
        # Return chosen state
        with torch.no_grad():
            q_values = variables.policy_net(state)
            q_values[~action_mask.unsqueeze(0)] = -float('inf')
            return q_values.max(1).indices.view(1, 1)
    return torch.tensor([[env.action_space(agent).sample(mask=action_mask.cpu().numpy().astype('int8'))]], device=variables.device, dtype=torch.long)
