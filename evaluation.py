from hanabi import plot_rewards
from pettingzoo.classic import hanabi_v5
import torch
import variables
from save_load import load

def evaluate():
    num_episodes = 20000
    total_rewards = []
    policy = load("file.path.here")
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

            # select and take an action for the current agent
            action = policy(state).max(1).indices.view(1, 1)
            
            # save current agent since env.step moves to the next agent
            current_agent = agent
            env.step(action.item())

            # Get next_state from the current agent's PoV after the action
            reward = torch.tensor([env.rewards[current_agent]], device=variables.device)
            episode_reward += reward.item()
        variables.episode_rewards.append(episode_reward)
    plot_rewards(total_rewards)