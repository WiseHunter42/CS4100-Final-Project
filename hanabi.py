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
import save_load

def plot_loss(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(f"graphs/loss_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()

def plot_rewards(episode_rewards):
    plt.figure()
    plt.plot(episode_rewards)
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
    training_limit = str(input("Enter training limit type ('time' or 'episodes'): ")).lower()
    
    if training_limit == 'time':
        # ask user for a time limit for training, and run the learning loop for that amount of time
        time_limit = float(input("Enter the time limit for training (in minutes): "))
    elif training_limit == 'episodes':
        # ask user for an episode limit for training, and run the learning loop for that amount of episodes
        episode_limit = int(input("Enter the episode limit for training: "))
    else:
        print("Invalid training limit. Please enter 'time' or 'episodes'.")
        return

    start_time = time.time()

    while True:
        if training_limit == 'time':
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit * 60:
                print(f"Time limit of {time_limit} minutes reached. Ending training.")
                break
        elif training_limit == 'episodes':
            if len(variables.episode_rewards) >= episode_limit:
                print(f"Episode limit of {episode_limit} reached. Ending training.")
                break  
        
        # reset the environment at the start of each episode
        # I think the seeds needs to be random or every episode will be the same?
        # Someone could check this
        env.reset()
        episode_reward = 0
        
        # report elapsed time every 100 episodes
        if len(variables.episode_rewards) % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode: {len(variables.episode_rewards)}, Elapsed Time: {elapsed_time:.2f} seconds")

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
    plot_loss(variables.loss_history())
    plot_rewards(variables.episode_rewards)
    save_load.save(f'policy_data/policy_{time.strftime("%Y%m%d_%H%M%S")}.pth', f'target_data/target_{time.strftime("%Y%m%d_%H%M%S")}.pth')
