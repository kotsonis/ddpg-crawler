'''This module implements specialized Actor/Critic RL Agents for training on 
Unity Environments using different methods.

All classes provide an interface to Agent.play(episodes) and Agent.train(iterations)

* Agent         base Agent, uses a policy gradient with experience replay
* DDPG          Deep Deterministic Policy Gradient
                  Continuous action: tanh
                  Horizon : uses n-step returns
                  Memory: prioritized experience replay
* SDPG          Sample Based Deterministic Policy Gradient. [Rahul Singh et al, 2020]
                https://arxiv.org/abs/2001.02652
                  Continuous action: Multivariate Gaussian distribution
                  Horizon : uses n-step returns
                  Memory: prioritized experience replay
* MIQN          Munchausen Reinforcement Learning [Nino Vieillard et all, 2020]
                https://arxiv.org/abs/2007.14430


'''
