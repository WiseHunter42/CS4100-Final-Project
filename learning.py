"""
Hanabi environment from PettingZoo:
https://pettingzoo.farama.org/environments/classic/hanabi/
"""

from pettingzoo.classic import hanabi_v5
from pettingzoo.utils.env_logger import EnvLogger
from select_action import select_action
from optimize import optimize
from tqdm import tqdm
import torch
import variables
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import save_load


EnvLogger.suppress_output()

def plot_loss(loss_history, run_dir):
    plt.figure()
    plt.plot(loss_history, label = "Step Loss")
    convFilter = []
    for i in range(1000):
        convFilter.append(1/1000.0)

    running_avgs = np.convolve(loss_history, convFilter, 'valid')
    plt.plot(running_avgs, label = "Moving Average Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss.png"))
    # plt.show()


def plot_rewards(episode_rewards, run_dir):
    plt.figure()
    plt.plot(episode_rewards, label = "Episodic Rewards")
    convFilter = []
    for i in range(1000):
        convFilter.append(1/1000.0)

    running_avgs = np.convolve(episode_rewards, convFilter, 'valid')
    plt.plot(running_avgs, label = "Moving Average Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "rewards.png"))
    # plt.show()


def learn():
    env = hanabi_v5.env(
        # render_mode="human"
        # additional parameters here if we want to change them
        # e.g. num_players=4, colors=5, ranks=5, hand_size=4, etc.
    )

    resuming = input("Resume from checkpoint? (y/n): ").strip().lower() == 'y'

    if resuming:
        runs = sorted(os.listdir(os.path.join("Data", "runs")))
        most_recent = runs[-1] if runs else None
        prompt = f"Enter run name to resume (default: {most_recent}): " if most_recent else "Enter run name to resume: "
        run_name = input(prompt).strip() or most_recent
        episode_limit = save_load.load_checkpoint(run_name)
        print(f"Resuming from episode {variables.episode}/{episode_limit}")
    else:
        episode_limit = int(input("Enter the episode limit for training: "))
        run_name = time.strftime('%Y-%m-%d_%H-%M-%S')
        variables.eps_decay = episode_limit // 2
        os.makedirs(os.path.join("Data", "runs", run_name), exist_ok=True)
        save_load.save_params(run_name, episode_limit)

    run_dir = os.path.join("Data", "runs", run_name)
    pbar = tqdm(total=episode_limit, initial=variables.episode, desc="Training", unit="ep")

    while True:
        if len(variables.episode_rewards) >= episode_limit:
            break

        # reset the environment at the start of each episode
        env.reset()
        episode_reward = 0

        for agent in env.agent_iter():
            # env.last() returns info for current agent
            observation, reward, terminated, truncated, _ = env.last()

            # If the agent is done, the game is over; break out of the loop early.
            if terminated or truncated:
                break

            # need to get the state from the observation,
            # which is a dictionary with keys "observation", "action_mask" which holds the legal moves
            state = torch.tensor(observation["observation"], dtype=torch.float32, device=variables.device).unsqueeze(0)
            action_mask = torch.tensor(observation["action_mask"], dtype=torch.bool, device=variables.device)

            # select and take an action for the current agent
            action = select_action(state, action_mask, env, agent)

            # save current agent since env.step moves to the next agent
            current_agent = agent
            env.step(action.item())

            # Get next_state from the current agent's PoV after the action
            next_observation = env.observe(current_agent)
            reward = torch.tensor([env.rewards[current_agent]], device=variables.device)

            if env.terminations[current_agent] or env.truncations[current_agent]:
                next_state = None
            else:
                next_state = torch.tensor(next_observation["observation"], dtype=torch.float32, device=variables.device).unsqueeze(0)

            episode_reward += reward.item()
            variables.memory.append(variables.Transition(state, action, next_state, reward))

            # update the step counter
            variables.step += 1

            # perform one step of the optimization on the policy network (every 4 steps)
            if variables.step % variables.train_frequency == 0:
                optimize()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            if variables.step % variables.update_frequency == 0:
                with torch.no_grad():
                    for p, tp in zip(variables.policy_net.parameters(), variables.target_net.parameters()):
                        tp.data.mul_(1 - variables.tau).add_(p.data, alpha=variables.tau)

        variables.episode += 1
        variables.episode_rewards.append(episode_reward)
        pbar.update(1)

        if variables.episode % variables.checkpoint_frequency == 0:
            save_load.save_checkpoint(run_name, episode_limit)

    pbar.close()
    env.close()
    return run_name, run_dir


if __name__ == "__main__":
    run_name, run_dir = learn()
    plot_loss(variables.loss_history, run_dir)
    plot_rewards(variables.episode_rewards, run_dir)
    save_load.save(run_name)
