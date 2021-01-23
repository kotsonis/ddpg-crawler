"""Accumulator of experiences across agents for an episode. """

from collections import deque
import numpy as np
class Acumulator():
    """class to hold experiences of multiple agents, allowing to flush a specific agent
    to the experience replay in case of early termination. """
    def __init__(self, **kwargs):
        self.num_agents = kwargs.get('num_agents',0)
        self.gamma = kwargs.get('gamma',0.99)
        self.replay_buffer_ptr = kwargs['replay_buffer_obj']
        self._queues = [deque() for _ in range(self.num_agents)]

    def push(self, state, action, reward, next_state, done):
        if (self.num_agents):
            for i in range(len(done)):
                exp = ([state[i]],
                    [action[i]],
                    [reward[i]],
                    [next_state[i]],
                    [done[i]])
                self._queues[i].append(exp)
        else:
            self.replay_buffer_ptr.add(
                                      states=state,
                                      actions=action,
                                      rewards=reward,
                                      next_states=next_state,
                                      dones=done)

    def flush(self, agent=0, flush_all=False):
        if flush_all:   # no agent specified, so we flush all of them
            for i in range(self.num_agents):
                next_reward = 0
                while len(self._queues[i]) > 0:
                    state, action, reward, next_state, done = self._queues[i].pop()
                    next_reward = np.array(reward) + next_reward*self.gamma
                    self.replay_buffer_ptr.add(states=state,actions=action,rewards=next_reward,next_states=next_state,dones=done)
        else:
            next_reward = 0
            while len(self._queues[agent]) > 0:
                state, action, reward, next_state, done = self._queues[agent].pop()
                next_reward = np.array(reward) + next_reward*self.gamma
                self.replay_buffer_ptr.add(states=state,actions=action,rewards=next_reward,next_states=next_state,dones=done)
            return

