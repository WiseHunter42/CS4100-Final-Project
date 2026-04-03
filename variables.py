from collections import namedtuple
import random
import agent
import torch


class ReplayBuffer:
    """List-based circular replay buffer with O(1) random access for fast sampling."""
    def __init__(self, capacity):
        self._buf = []
        self._cap = capacity
        self._pos = 0

    def append(self, item):
        if len(self._buf) < self._cap:
            self._buf.append(item)
        else:
            self._buf[self._pos] = item
        self._pos = (self._pos + 1) % self._cap

    def sample(self, k):
        return random.sample(self._buf, k)

    def __len__(self):
        return len(self._buf)
    

CAPACITY = 200000

step = 0
episode = 0 # curr episode we're on
loss_history = []
episode_rewards = []
batch_size = 128
gamma = 0.99
eps_start = 1.0
eps_end = 0.01
eps_decay = None # determine at run time; should be 1/2 of total episodes
tau = 0.001
lr = 3e-4
train_frequency = 8    # how often to run an optimization step, in terms of env steps
update_frequency = 100 # how often to update the target network, in terms of number of steps
checkpoint_frequency = 10000 # how often to save a checkpoint, in terms of number of episodes

device = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")

memory = ReplayBuffer(CAPACITY)
policy_net = agent.Network().to(device=device)
target_net = agent.Network().to(device=device)
optimizer = torch.optim.AdamW(params=policy_net.parameters(), lr=lr, amsgrad=True)
criterion = torch.nn.SmoothL1Loss()

target_net.load_state_dict(policy_net.state_dict()) # load the default random weights/biases from policy into target, so they're equal at the start

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_action_mask'))
