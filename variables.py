from collections import namedtuple, deque
import agent
from torch import optim
CAPACITY = 10000

epoch = 1 
batch_size = 128
gamma = 0.99
eps_start = 0.9
eps_end = 0.01
eps_decay = 2500
tau = 0.005
lr = 3e-4

device = 0 # TODO: Determine what device
memory = deque([], maxlen=CAPACITY)
policy_net = agent.network().to(device=device) 
target_net = agent.network().to(device=device) 
optimizer = optim.AdamW(params=policy_net.parameters, lr=lr, amsgrad=True)

target_net.load_state_dict(policy_net.state_dict()) # load the default random weights/biases from policy into target, so they're equal at the start

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

