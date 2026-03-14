import random
import math
import variables
import torch

def select_action(state, action_mask):
    eps_threshold = variables.eps_end + (variables.eps_start - variables.eps_end) * math.exp(-1. * variables.epoch / variables.eps_decay)
    sample = random.random()
    if (sample > eps_threshold) :
        # Return chosen state, but mask illegal actions
        with torch.no_grad():
            q_values = variables.policy_net(state)
            # Set illegal actions to -inf so they won't be selected
            q_values = q_values.masked_fill(~action_mask, float('-inf'))
            return q_values.max(1).indices.view(1, 1)
    # Explore: sample from legal actions
    legal_actions = action_mask.nonzero(as_tuple=True)[0]
    action = random.choice(legal_actions.tolist())
    return torch.tensor([[action]], device=variables.device, dtype=torch.long)
