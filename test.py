from collections import namedtuple
import numpy as np
from operator import itemgetter
import random
import torch
class a():
    def __init__(self):
        self.a = 5
        self.experience = (self.__dict__
                            .get('experience',
                                        namedtuple("Experience", 
                                                  field_names=["state", "action", "reward", "next_state", "done"])))
        self._buffer = []
        self._idx = 0
    def add(self, **kwargs):
        exp = self.experience(**kwargs)
        self._buffer.append(exp)
        self._idx += 1
        return
    def sample(self):
        batch_size = 3
        idxes = [random.randint(0, len(self._buffer) - 1) for _ in range(batch_size)]
        #res = np.array(itemgetter(*idxes)(self._buffer), dtype=object)
        res = np.array(itemgetter(*idxes)(self._buffer), dtype=object)
        print(*zip(res))
        ret_val = zip(*res)
        ret_list = []
        for vals in ret_val:
            d = torch.tensor(vals).float()
            if (d.dim() < 2):
                d = d.unsqueeze(-1)
            ret_list.append(d)
        return ret_list

class b(a):
    def __init__(self):
        self.experience = namedtuple("Experience", field_names=['state','action','reward','next_state','done','gamma'])
        super(b,self).__init__()
x = b()
y = a()
print('fields of b {}'.format(x.experience._fields))
print('fields of a {}'.format(y.experience._fields))
state = np.ones(129)
action = np.ones(20)
for i in range(20):
    x.add(state=state, action=action, reward=i+3, next_state=i+4, done=True, gamma=i+6)
states, actions, rewards, next_states, dones, gammas = x.sample()
print(states.shape)
print(rewards.shape)
print(dones)