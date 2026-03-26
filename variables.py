from collections import namedtuple, deque
import agent
import torch

CAPACITY = 100000

epoch = 0
episode = 0 # curr episode we're on 
loss_history = []
episode_rewards = []
batch_size = 128
gamma = 0.99
eps_start = 1.0
eps_end = 0.01
eps_decay = None # determine at run time; should be 1/2 of total episodes
tau = 0.005
lr = 3e-4
update_frequency = 200 # how often to update the target network, in terms of number of epochs

device = torch.device("cuda" if torch.cuda.is_available() 
                        else "mps" if torch.backends.mps.is_available() 
                        else "cpu")
memory = deque([], maxlen=CAPACITY)
policy_net = agent.Network().to(device=device) 
target_net = agent.Network().to(device=device) 
optimizer = torch.optim.AdamW(params=policy_net.parameters(), lr=lr, amsgrad=True)

target_net.load_state_dict(policy_net.state_dict()) # load the default random weights/biases from policy into target, so they're equal at the start

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

