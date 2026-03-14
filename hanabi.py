"""
Hanabi environment from PettingZoo:
https://pettingzoo.farama.org/environments/classic/hanabi/
"""

from pettingzoo.classic import hanabi_v5
from select_action import select_action
from optimize import optimize
import torch
import variables

def learn():
    env = hanabi_v5.env(
        render_mode="human"
        # additional parameters here if we want to change them
        # e.g. num_players=4, colors=5, ranks=5, hand_size=4, etc.
    )

    if variables.device.type in ["cuda", "mps"]:
        num_episodes = 600
    else:
        num_episodes = 50

    for i in range(num_episodes):
        # reset the environment at the start of each episode
        # I think the seeds needs to be random or every episode will be the same? 
        # Someone could check this
        env.reset() 
        
        for agent in env.agent_iter():
            # env.last() returns info for current agent
            observation, reward, terminated, truncated, _ = env.last()

            # need to get the state from the observation, 
            # which is a dictionary with keys "observation", "action_mask" which holds the legal moves
            state = torch.tensor(observation["observation"], dtype=torch.float32, device=variables.device).unsqueeze(0)
            action_mask = torch.tensor(observation["action_mask"], dtype=torch.bool, device=variables.device)

            # select and take an action for the current agent
            action = select_action(state, action_mask)
            
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
            
            variables.memory.append(variables.Transition(state, action, next_state, reward))

            # perform one step of the optimization on the policy network
            optimize()
            variables.epoch += 1

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = variables.target_net.state_dict()
            policy_net_state_dict = variables.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*variables.tau + target_net_state_dict[key]*(1-variables.tau)
            variables.target_net.load_state_dict(target_net_state_dict)
    env.close()

if __name__ == "__main__":
    learn()
    
#     env = hanabi_v5.env(
#         render_mode="human"
#         # additional parameters here if we want to change them
#         # e.g. num_players=4, colors=5, ranks=5, hand_size=4, etc.
#     )
#     env.reset(seed=42)

#     for agent in env.agent_iter():
#         observation, reward, termination, truncation, info = env.last()

#         if termination or truncation:
#             action = None
#         else:
#             # action_mask is a binary vector where each index of the vector represents whether the action is legal or not
#             # The action_mask will be all zeros for any agent whose turn it is not
#             # taking an illegal move will end the game; -1 reward for agent that moved illegaly (technically 0 for others, but we only care about -1)
#             mask = observation["action_mask"]
#             # insert policy here

#             # for now, sample a random legal action
#             action = env.action_space(agent).sample()

#         # the action is applied to the environment and the next agent's turn begins
#         env.step(action)
#     env.close()

# if __name__ == "__main__":
#     main()
