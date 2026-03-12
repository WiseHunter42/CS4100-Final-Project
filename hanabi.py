"""
Hanabi environment from PettingZoo:
https://pettingzoo.farama.org/environments/classic/hanabi/
"""

from pettingzoo.classic import hanabi_v5
from select_action import select_action
import torch
import variables

def learn():
    env = hanabi_v5.env(
        render_mode="human"
        # additional parameters here if we want to change them
        # e.g. num_players=4, colors=5, ranks=5, hand_size=4, etc.
    )
    env.reset(seed=42)

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    for i in range(num_episodes):
        # state, info = agent.env.reset() TODO FIGURE OUT THIS FIRST STATE
        state = torch.tensor(state, dtype=torch.float32, device=variables.device).unsqueeze(0)
        
        for agent in env.agent_iter():
            
            action = select_action(state)
            observation, reward, terminated, truncated, _ = agent.env.step(action.item())
            reward = torch.tensor([reward], device = variables.device)
            
            if terminated or truncated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=variables.device).unsqueeze(0)
            
            variables.memory.append(variables.Transition(state, action, next_state, reward))
            state = next_state
            # TODO finish this loop
    env.close()


def main():
    env = hanabi_v5.env(
        render_mode="human"
        # additional parameters here if we want to change them
        # e.g. num_players=4, colors=5, ranks=5, hand_size=4, etc.
    )
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # action_mask is a binary vector where each index of the vector represents whether the action is legal or not
            # The action_mask will be all zeros for any agent whose turn it is not
            # taking an illegal move will end the game; -1 reward for agent that moved illegaly (technically 0 for others, but we only care about -1)
            mask = observation["action_mask"]
            # insert policy here

            # for now, sample a random legal action
            action = env.action_space(agent).sample()

        # the action is applied to the environment and the next agent's turn begins
        env.step(action)
    env.close()

if __name__ == "__main__":
    main()
