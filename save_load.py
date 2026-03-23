import torch

import variables
import agent

def save(policy_file="policy.pth", target_file="target.pth"):
    torch.save(variables.policy_net.state_dict(), policy_file)
    torch.save(variables.target_net.state_dict(), target_file)

def load(target_file="target.pth", weights_only = True):
    network = agent.Network().to(device=variables.device)
    network.load_state_dict(torch.load(target_file, weights_only=weights_only))
    return network

def resume(policy_file="policy.pth", target_file="target.pth"):
    variables.target_net.load_state_dict(torch.load(target_file, weights_only=True))
    variables.policy_net.load_state_dict(torch.load(policy_file, weights_only=True))

