from collections import namedtuple, deque
import random

CAPACITY = 10000

epoch = 1
policy_net = 0 # TODO: Implement
target_net = 0 # TODO: Implement
device = 0 # TODO: Determine what device
memory = deque([], maxlen=CAPACITY)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

