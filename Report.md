# Crawler Unity environment solution Report

This project implements a PPO to learn how to navigate within the environment and collect rewards.

## Environment particularities

Crawler is a four legged creature, with two joints per leg, which needs to learn how to keep it's balance and move as fast as possible in the forward direction. Unfortunately, as reward is also provided for looking in the right direction, so learning can have local maxima and the optimal solution not being found.

## Reinforcement Learning and Policy Gradient methods background

Reinforcement learning (RL) is one of the three basic machine learning paradigms, together with supervised learning and unsupervised learning. Whereas both supervised and unsupervised learning are concerned with finding an accurate system to predict an output given inputs, RL focuses on Markov Decision Processes (MDPs) and is concerned with the transition from states to states and the reward/cost associated with these transitions.  
This environment/agent interaction is depicted in below figure:
![Agent_environment_interagion](./images/Sutton_agent_environment.png)

Basically, the agent and environment interact in a sequence of discrete time steps t. At each time step t, the agent receives some representation of the environment's state, S<sub>t</sub>&#8712;S, and on that basis selects an action, A<sub>t</sub>&#8712;A(S). At the next timestep t+1, in part as a consequence of its action, the agent receives a scalar reward, R<sub>t+1</sub>&#8712;&#8477;, as well as an new state S<sub>t+1</sub>. The MDP and agent together create a trajectory from an initial time step t over n transitions of states,actions,rewards, and next states as follows:

<img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle S_t,A_t,R_t,S_{t%2b 1},A_{t%2b1},\cdots,S_{t%2bn},A_{t%2bn},R_{t%2bn}"/>


We represent the sum of rewards accumulated over a trajectory as <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle G_t = R_t %2b R_{t%2b1} %2b \cdots %2b R_{t%2bn}"/>. Clearly the limit of G<sub>t</sub> as the trajectory steps n increase is unbounded, so to make sure that we can have a bounded maximum total reward, we discount the rewards from the next transaction by a factor &gamma;&#8712;(0,1] with the case of γ=1 being useful only when a task is episodic, ie with a fixed number of transitions.
In the case that the sets S,R,and A are finite, then the MDP is finite and the following hold:
* Random variables <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle R_{t%2B1},S_{t%2b1}"/> have well defined discrete probability distributions **depending only on the preceding state <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle S_t"/> & action <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle A_t"/>**
* Given a random state s' ∈ S and reward r ∈ R the probability of s' and r occuring at time t given a preceding state s and action a is given by the four argument MDP <strong>dynamics function</strong> <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle p:S \times R \times S \times A \to [0,1]"/>
  <img alt="formula" src="https://render.githubusercontent.com/render/math?math=p(s',r|s,a) \doteq \Pr \{ {S_t} = {s'},{R_t} = {r'}|{S_{t - 1}} = s,{A_{t - 1}} = a\} \quad \forall (s',s) \in S,r \in R,a \in A(s)%0A"/>
* Namely, given a state s and an action a, a probability can be assigned to reaching a state s' and receiving reward r, however the sum of these probabilities over all the possible next states and rewards is 1. That is:
  * <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\displaystyle\sum_{s'\in S} {\sum_{r \in R} {p(s',r\space| s,a)}} = 1 \quad \forall s\in S,a \in A(s)%0A"/>

From the <em>dynamics</em> function <em>**p**</em> we can derive other useful functions:
* <em>state transition probabilities</em> : <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\begin{align}%0a\displaystyle p(s'|s,a) = \sum_{r \in R} {p(s',r|s,a)} \quad \quad p:S\times S\times A \to [0,1]\end{align}%0A"/>
* <em>expected rewards</em>: <img alt="formula" src="https://render.githubusercontent.com/render/math?math=\begin{align}%0a\displaystyle r(s,a) = \mathbb{E}[R_t|S_{t-1}=s,A_{t-1}=a] = \sum_{r \in R}\sum_{s'\in S} {p(s',r|s,a)} \quad \quad r:S\times A \to \mathbb{R}\end{align}%0A"/>

We define a policy, ![formula](https://render.githubusercontent.com/render/math?math=\pi(s)), as a function ![formula](https://render.githubusercontent.com/render/math?math=\pi:S\to%20A) that produces an action a given a state s. Thus the expected rewards at a state s can be expressed as ![formula](https://render.githubusercontent.com/render/math?math=r(s,a)=r(s,\pi(s))=\mathbb{E}[R_t|S_{t-1}=s,A_{t-1}=\pi(s)]). This allows us to define the action-value function ![formula](https://render.githubusercontent.com/render/math?math=q_\pi(s,a)), which defines the value of taking action a in state s under the policy π, and continuing the trajectory by taking following the policy π as follows:

[comment]: <> (%2B is plus sign)
[comment]: <> (%0A is line feed, %26 is &)

<img src="https://render.githubusercontent.com/render/math?math=\begin{align*}%0a\displaystyle q_\pi(s,a) %26= \mathbb{E}[G_t | S_t = s, A_t = a] \\%0A%26= \mathbb{E}_\pi\left[\displaystyle\sum_{k=0}^{\infty}{\gamma^kR_{t%2bk%2b1}|S_t=s,A_t=a }\right] \\%0A\end{align*}%0A">


A fundamental property of the action value function is that it satisfies recursive relationships. That is, for any policy ![formula](https://render.githubusercontent.com/render/math?math=\pi) and any state ![formula](https://render.githubusercontent.com/render/math?math=s), the following consistency holds between the action value of ![formula](https://render.githubusercontent.com/render/math?math=(s,a)) and the action value of (![formula](https://render.githubusercontent.com/render/math?math=s_{%2b1},a_{%2b1})):

<img src="https://render.githubusercontent.com/render/math?math=\begin{align*}%0a\displaystyle q_\pi(s,a) %26= \mathbb{E}[R_{t%2b 1} %2b \gamma G_{t%2b 1}|S_t=s,A_t=a] \\%0A%26= \displaystyle\sum_{s'}\sum_rp(s',r|s,a)r%2b \sum_{s'}\sum_rp(s',r|s,a)\gamma(\mathbb{E}[G_{t%2b 1}|S_{t%2b1}=s',A_{t%2b1}=a=\pi(s)]) \\%0A%26=\displaystyle\sum_{s'}\sum_rp(s',r|s,a)[r%2bq_\pi(s',a')]\end{align*}%0A">

The sum of probabilities over all possible next states s' and rewards r reflects the stochastisticy of the system. If we focus on a specific time-step t, then we can reformulate this with bootstrapping as follows:

![Image](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%20q_%5Cpi%28s%2Ca%29%20%26%3D%20r_t&plus;%5Cgamma%20q_%5Cpi%28s%27%2Ca%27%7Ca%27%3D%5Cpi%28s%29%29%5Cimplies%5C%5C%20%5Cdelta_t%20%26%3D%20%5Cdisplaystyle%20%5Cunderbrace%7B%5Coverbrace%7Br_t&plus;%5Cgamma%20q_%5Cpi%28s%27%2Ca%27%29%7D%5E%5Ctext%7Bestimated%20q%7D-q_%5Cpi%28s%2Ca%29%7D_%7B%5Ctext%7BTD%20Error%7D%7D%20%5Cend%7Baligned%7D)

When ![formula](https://render.githubusercontent.com/render/math?math=q_\pi) is approximated as a non-linear function with a neural network with parameters θ, then we denote the action-value function as ![formula](https://render.githubusercontent.com/render/math?math=Q^\pi%28%20s,a|\theta%20%29).
### Q - learning
In Q-learning, we consider that π is an optimal policy, that selects the next action based on the maximum q value of the future state. We learn the parameters of ![formula](https://render.githubusercontent.com/render/math?math=Q^\pi) through gradient descent, trying to fit the above function for a Bellmann error of 0. With a learning rate ![formula](https://render.githubusercontent.com/render/math?math=\alpha) the formula and gradient is thus:
<img src="https://render.githubusercontent.com/render/math?math=\begin{align*}%0a\Delta q_t %26= \text{TD Error}= (r_t%2b\gamma \max_a Q(s_{t%2b1},a , \theta) - Q(s_t,a_t , \theta) q_\pi(s,a) \\%0A \Delta \theta %26= \alpha\Delta q_t \nabla_\theta Q(s,a,\theta) \end{align*}%0A">

### Deep Q Learning Networks
Mnih et al., in their [DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) paper, used deep neural networks as a function approximator for the above Q-Learning.
Specifically, two Q-networks are trained by minimising a sequence of loss functions ![formula](https://render.githubusercontent.com/render/math?math=L_i(\theta_i)) that changes at each iteration i. One Q-network has parameters ![formula](https://render.githubusercontent.com/render/math?math=\theta) and is actively learning, while the second, target, Q-network has parameters ![formula](https://render.githubusercontent.com/render/math?math=\overline{\theta}) and it's parameters are gradually updated with the online's parameters. The loss function is thus:

<img alt="formula" src="https://render.githubusercontent.com/render/math?math=\begin{align*}%0a\displaystyle L_i(\theta_i)%26=\mathbb{E}_{s,a\sim \rho(\cdot)}[(y_i-Q(s,a|\theta_i))^2], \\%0Ay_i %26= \mathbb{E}_{s\prime\sim Env}[r%2b\gamma\displaystyle \max_{a\prime}Q(s',a'|\overline{\theta_i})|s,a]\quad =\text{estimated}\:q\:\text{at iteration}\:i,\\%0A\rho(a,s)%26=probability\: distribution\: over\: sequences\: s\: and\: actions\: a\end{align*}%0A">

### Policy Gradient methods

Finding the optimal policy with Q-learning is not trivial on continuous action spaces, since we would need to optimize the selected action at every timestep, requiring computation that will slow down our algorithm.
Instead, in [policy gradient methods](http://proceedings.mlr.press/v32/silver14.pdf?CFID=6293331&CFTOKEN=eaaee2b6cc8c9889-7610350E-DCAB-7633-E69F572DC210F301), the actor uses a learned policy function approximator <img src="https://render.githubusercontent.com/render/math?math=\mu(s|\theta^\mu):S\gets A "> to select the best action. Learning what action is best given a state is done by maximizing the Q value for the state. Assuming we are sampling actions from a behavior <img src="https://render.githubusercontent.com/render/math?math=\beta=\pi(s,a)"> the objective of the actor becomes:

<img alt="formula" src="https://render.githubusercontent.com/render/math?math=\begin{align*}%0a\displaystyle J_\beta (\mu_\theta) %26= \int_S\rho^\beta (s)Q^\mu(s,\mu_\theta (s))ds \\%0A%26=\mathbb{E}_{s\sim \rho^\beta}[Q^\mu(s,\mu_\theta (s))]\\%0A Loss%26=-\mathbb{E}_{s\sim \rho^\beta}[Q^\mu(s,\mu_\theta (s))]\end{align*}%0A">
And the gradient wrt θ becomes:

<img alt="formula" src="https://render.githubusercontent.com/render/math?math=\begin{align*}%0a\displaystyle \nabla_\theta J_\beta (\mu_\theta) %26\approx \int_S\rho^\beta (s)\nabla_\theta \mu_\theta(a|s)Q^\mu(s,a)ds \\%0A%26=\mathbb{E}_{s\sim \rho^\beta}[\nabla_\theta \mu_\theta (s) \nabla_aQ^\mu(s,a)|_{a=\mu_\theta(s)}]\\%0A\end{align*}%0A">

Instead of using just Q, we can actually learn to select the action that would provide the greatest advantage. In this case, advantage being the difference between the next action value (Q) minus the value of the current state (V).

This would lead to a loss function as follows:

<img alt="formula" src="https://render.githubusercontent.com/render/math?math=\begin{align*}%0a\displaystyle L(\theta) =  \mathbb{E}_{log \pi_\theta (a_t | s_t) A_t)end{align*}%0A">

With the above refresher and definitions, we can move to the presentation of the algorithm implemented.

## Proximal Policy Optimization

In [TRPO](https://arxiv.org/pdf/1502.05477.pdf) the concept of maximizing a surrogate is introduced and the objective becomes as follows:
![TRPO surrogate](images/TRPO.png)

We can remove the constraint by including it as a penalty and write this as the unconstrained optimization problem:

![TRPO optimization](images/TRPO_optimization_problem.png)


The action probability of our policy with the new parameters over the action probability with the previous parameters can be written as r(θ).

![r_theta](images/r_t_theta.png)

TRPO thus maximizes the surrogate objective:
![TRPO Loss](images/TRPO_Loss.png)


Proximal Policy Optimization Algorithm ([PPO](https://arxiv.org/abs/1707.06347)), by Schulman et. al., proposes to clip the loss as follows:

![PPO_Loss](images/PPO_Loss.png)

Since we try to learn both the value function (for estimating the advantage), as well as the policy, our loss function needs to take that into account. Furthermore, we want to make sure that our learned policy is as diverse as possible, so we give a bonus for the entry. As such, the overall loss on which we take the gradient is :

![PPO_overall_loss](images/PPO_overall_loss.png)

After sampling actions with the existing policy, the PPO algorithm uses this loss to optimize the θ parameters, with the below algorithm:

![PPO Algorithm](images/PPO_algorithm.png)

The above is implemented in our agent [ppo.py](agents/ppo.py) as follows:

```python

for policy_epochs in range(self.policy_optimization_epochs):
  idx = np.random.choice(n_samples, self.batch_size, replace=False)
  self.tot_epochs += 1
  # sample a batch
  with torch.no_grad():
      states = all_states[idx]
      actions = all_actions[idx]
      old_log_probs = all_old_log_probs[idx]
      returns = all_returns[idx]
      values = all_values[idx]
      gae = all_gae[idx]
  # get new predictions   
  values_pred, log_probs, entropy = self.policy.get_probs_and_value(states,actions)
  # compute value loss
  value_loss = F.smooth_l1_loss(values_pred, returns)
  # compute policy loss
  ratio = (log_probs - old_log_probs).exp()
  policy_objective = gae*ratio
  policy_objective_clamped = torch.where(gae > 0, (1+self.policy_clip_range) * gae, (1-self.policy_clip_range) * gae)
  policy_loss = -torch.min(policy_objective, policy_objective_clamped).mean()
  # compute entropy loss
  entropy_loss = -entropy.mean()
  # objective
  ppo_objective = policy_loss + self.vf_coeff*value_loss + self.entropy_coeff*entropy_loss
  # optimize PPO DNN
  self.policy_optimizer.zero_grad()
  ppo_objective.backward()
  nn.utils.clip_grad_norm_(self.policy.parameters(), self.policy_gradient_clip)
  self.policy_optimizer.step()
```

We would like to iterate over our samples as much as possible during a learning step, in order to get the most from these experiences, and reduce training time. However, during testing, we found that we can take too large steps in the wrong direction, so an 'early stop' mechanism was implemented according to the KL divergence. Specifically, the above `for` loop checks how far off our "old" policy is with regards to the learned one, and breaks as follows:


```python
with torch.no_grad():
  values, log_probs, _ = self.policy.get_probs_and_value(all_states,all_actions)
  kl = (all_old_log_probs - log_probs).mean()
  if kl > self.policy_stopping_kl: 
      break
```

## Plot of Rewards
With the above parameters, the agent was able to solve the game (average reward over 100 episodes >2000) in 1198 iterations.


Below is the 100 episode average reward per iteration, as well as the objective value per iteration.

![training_log](./images/return_and_objective_vs_step.png)
