from collections import namedtuple, deque
from operator import itemgetter
import torch
import numpy as np
import random
class a():
    
    

    def __init__(self, **kwargs):
        self.exp = namedtuple("exp", field_names=["a", "b", "c", "d"])
        self._buffer = []
        self.idx = 0
        self._sampled_exps = dict()
    def __len__(self):
        return len(self._buffer)

    def _add(self, **kwargs):
        data = self.exp(**kwargs)
        self.idx += len(data)
        self._buffer.append(data)
    
    def add(self, **kwargs):
        self._add(**kwargs)
    
    def sample(self, **kwargs):
        samples = kwargs.get('size', 10)
        idx = np.random.randint(len(self._buffer) - 1, size=samples)
        self._encode_samples(idx)
    
    def _encode_samples(self, idxes):
        res = list(itemgetter(*idxes)(self._buffer))
        b_size = len(res)

        ret_val = zip(*res)
        ret_list = deque()

        for vals in ret_val:
            d = torch.tensor(vals).float()
            if (d.dim() < 2):
                d = d.unsqueeze(-1)
            ret_list.append(d)
        for name in (self.exp._fields):
            self._sampled_exps[name]= ret_list.popleft() 
        return
    
    @property
    def a(self):
        return self._sampled_exps['a']
    
    @property
    def b(self):
        return self._sampled_exps['b']
    @property
    def c(self):
        return self._sampled_exps['c']
    @property
    def d(self):
        return self._sampled_exps['d']
buf = a()

for _ in range(100):
    buf.add(a=1, b=2, c=3, d=4)
buf.sample()
print(buf.a)

class b(a):
    def __init__(self,**kwargs):
        super(b,self).__init__(**kwargs)
        exp = namedtuple("exp", self.exp._fields +tuple("f"))
        self.exp = exp
    @property
    def f(self):
        return self._sampled_exps['f']

buf_2 = b()
for _ in range(100):
    buf_2.add(a=1, b=2, c=3, d=4, f=5)
buf_2.sample()
print(buf_2.f)