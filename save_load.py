import torch
import os
import variables
import agent


def _run_dir(run_name):
    return os.path.join("Data", "runs", run_name)


def save(run_name):
    run_dir = _run_dir(run_name)
    os.makedirs(run_dir, exist_ok=True)
    torch.save(variables.policy_net.state_dict(), os.path.join(run_dir, "policy.pth"))
    torch.save(variables.loss_history, os.path.join(run_dir, "loss_history.pth"))
    torch.save(variables.episode_rewards, os.path.join(run_dir, "episode_rewards.pth"))


def save_params(run_name, episode_limit):
    run_dir = _run_dir(run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        f.write(f"episode_limit={episode_limit}\n")
        f.write(f"batch_size={variables.batch_size}\n")
        f.write(f"gamma={variables.gamma}\n")
        f.write(f"eps_start={variables.eps_start}\n")
        f.write(f"eps_end={variables.eps_end}\n")
        f.write(f"eps_decay={variables.eps_decay}\n")
        f.write(f"tau={variables.tau}\n")
        f.write(f"lr={variables.lr}\n")
        f.write(f"update_frequency={variables.update_frequency}\n")
        f.write(f"capacity={variables.CAPACITY}\n")
        f.write(f"network_shape={agent.SHAPE}\n")


def load(run_name):
    network = agent.Network().to(device=variables.device)
    path = os.path.join(_run_dir(run_name), "policy.pth")
    network.load_state_dict(torch.load(path, weights_only=True))
    return network


def save_checkpoint(run_name, episode_limit):
    checkpoint = {
        'policy_net': variables.policy_net.state_dict(),
        'target_net': variables.target_net.state_dict(),
        'optimizer': variables.optimizer.state_dict(),
        'step': variables.step,
        'episode': variables.episode,
        'loss_history': variables.loss_history,
        'episode_rewards': variables.episode_rewards,
        'eps_decay': variables.eps_decay,
        'episode_limit': episode_limit,
    }
    torch.save(checkpoint, os.path.join(_run_dir(run_name), "checkpoint.pth"))


def load_checkpoint(run_name):
    checkpoint = torch.load(
        os.path.join(_run_dir(run_name), "checkpoint.pth"),
        weights_only=False,
        map_location=variables.device
    )
    variables.policy_net.load_state_dict(checkpoint['policy_net'])
    variables.target_net.load_state_dict(checkpoint['target_net'])
    variables.optimizer.load_state_dict(checkpoint['optimizer'])
    variables.step = checkpoint['step']
    variables.episode = checkpoint['episode']
    variables.loss_history = checkpoint['loss_history']
    variables.episode_rewards = checkpoint['episode_rewards']
    variables.eps_decay = checkpoint['eps_decay']
    return checkpoint['episode_limit']
