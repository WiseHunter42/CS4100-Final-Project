"""
Hanabi environment from PettingZoo:
https://pettingzoo.farama.org/environments/classic/hanabi/
"""

from pettingzoo.classic import hanabi_v5


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
            # taking an illegal move will end the game for the illegally moving agent???
            mask = observation["action_mask"]

            # insert policy here

            # for now, sample a random legal action
            action = env.action_space(agent).sample()

        # the action is applied to the environment and the next agent's turn begins
        env.step(action)
    env.close()

if __name__ == "__main__":
    main()
