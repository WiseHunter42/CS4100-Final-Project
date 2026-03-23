"""
Hanabi environment from PettingZoo:
https://pettingzoo.farama.org/environments/classic/hanabi/
"""

from pettingzoo.classic import hanabi_v5
from select_action import select_action
from optimize import optimize
import torch
import variables
import matplotlib.pyplot as plt
import time

def plot_loss():
    plt.figure()
    plt.plot(variables.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(f"graphs/loss_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()

def plot_rewards():
    plt.figure()
    plt.plot(variables.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards")
    plt.tight_layout()
    plt.savefig(f"graphs/rewards_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()

def learn():
    env = hanabi_v5.env(
        # render_mode="human"
        # additional parameters here if we want to change them
        # e.g. num_players=4, colors=5, ranks=5, hand_size=4, etc.
    )

    if variables.device.type in ["cuda", "mps"]:
        num_episodes = 6000000
    else:
        num_episodes = 500000
    for i in range(num_episodes):
        # reset the environment at the start of each episode
        # I think the seeds needs to be random or every episode will be the same?
        # Someone could check this
        env.reset()
        episode_reward = 0
        
        # report time passed and estimated time remaining every 100 episodes
        if i % 100 == 0 and i > 0:
            print(f"Episode {i} / {num_episodes}")
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / i * num_episodes
            estimated_time_remaining = estimated_total_time - elapsed_time
            print(f"Time elapsed: {elapsed_time:.2f} seconds, Estimated time remaining: {estimated_time_remaining:.2f} seconds")
        if i == 0:
            start_time = time.time()

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
            action = select_action(state, env, agent)
            
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

            # perform one step of the optimization on the policy network
            optimize()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = variables.target_net.state_dict()
            policy_net_state_dict = variables.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*variables.tau + target_net_state_dict[key]*(1-variables.tau)
            variables.target_net.load_state_dict(target_net_state_dict)
        variables.episode_rewards.append(episode_reward)
    env.close()

if __name__ == "__main__":
    learn()
    plot_loss()
    plot_rewards()
