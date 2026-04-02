from pettingzoo.classic import hanabi_v5
import torch
import os
import variables
from save_load import load
import matplotlib.pyplot as plt

def plot_reward_histogram(total_rewards, eval_dir):
    plt.figure()
    plt.hist(total_rewards, bins=26, range=(-0.5, 25.5), edgecolor='black')
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.title("Distribution of Total Rewards")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "eval_histogram.png"))
    plt.show()

def evaluate():
    run_name = input("Enter the run name to evaluate (folder name under Data/runs/): ")
    eval_dir = os.path.join("Data", "eval", run_name)
    os.makedirs(eval_dir, exist_ok=True)

    num_episodes = 10000
    total_rewards = []
    policy = load(run_name)
    policy.eval()
    env = hanabi_v5.env()

    for episode in range(num_episodes):
        env.reset()

        episode_reward = 0

        for agent in env.agent_iter():
            # env.last() returns info for current agent
            observation, reward, terminated, truncated, _ = env.last()

            # If the agent is done, the game is over; break out of the loop early.
            if terminated or truncated:
                break

            state = torch.tensor(observation["observation"], dtype=torch.float32, device=variables.device).unsqueeze(0)

            action_mask = torch.tensor(observation["action_mask"], dtype=torch.bool, device=variables.device)

            # select and take an action for the current agent
            with torch.no_grad():
                q_values = policy(state)
                q_values[~action_mask.unsqueeze(0)] = -float('inf')
                action = q_values.max(1).indices.view(1, 1)

            # save current agent since env.step moves to the next agent
            current_agent = agent
            env.step(action.item())

            # Get next_state from the current agent's PoV after the action
            reward = torch.tensor([env.rewards[current_agent]], device=variables.device)
            episode_reward += reward.item()
        total_rewards.append(episode_reward)
    plot_reward_histogram(total_rewards, eval_dir)

evaluate()
