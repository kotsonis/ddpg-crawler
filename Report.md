# Crawler Unity environment solution Report

This project implements a PPO to learn how to navigate within the environment and collect rewards.

## Environment particularities

Crawler is a four legged creature, with two joints per leg, which needs to learn how to keep it's balance and move as fast as possible in the forward direction. Unfortunately, as reward is also provided for looking in the right direction, so learning can have local maxima and the optimal solution not being found.

## Reinforcement Learning and Policy Gradient methods background

Reinforcement learning (RL) is one of the three basic machine learning paradigms, together with supervised learning and unsupervised learning. Whereas both supervised and unsupervised learning are concerned with finding an accurate system to predict an output given inputs, RL focuses on Markov Decision Processes (MDPs) and is concerned with the transition from states to states and the reward/cost associated with these transitions.  
This environment/agent interaction is depicted in below figure:
![Agent_environment_interagion](./images/Sutton_agent_environment.png)

Basically, the agent and environment interact in a sequence of discrete time steps t. At each time step t, the agent receives some representation of the environment's state, S<sub>t</sub>&#8712;S, and on that basis selects an action, A<sub>t</sub>&#8712;A(S). At the next timestep t+1, in part as a consequence of its action, the agent receives a scalar reward, R<sub>t+1</sub>&#8712;&#8477;, as well as an new state S<sub>t+1</sub>. The MDP and agent together create a trajectory from an initial time step t over n transitions of states,actions,rewards, and next states as follows:

S<sub>t</sub>,A<sub>t</sub>,R<sub>t</sub>,S<sub>t+1</sub>,…,A<sub>t+n-1</sub>,R<sub>t+n-1</sub>,S<sub>t+n</sub>,A<sub>t+n</sub>,R<sub>t+n</sub>

We represent the sum of rewards accumulated over a trajectory as G<sub>t</sub>=R<sub>t</sub>+R<sub>t+1</sub>+⋯+R<sub>t+n</sub>. Clearly the limit of G<sub>t</sub> as the trajectory steps n increase is unbounded, so to make sure that we can have a bounded maximum total reward, we discount the rewards from the next transaction by a factor &gamma;&#8712;(0,1] with the case of γ=1 being useful only when a task is episodic, ie with a fixed number of transitions.
In the case that the sets S,R,and A are finite, then the MDP is finite and the following hold:
* Random variables R<sub>t+1</sub> & S<sub>t+1</sub> have well defined discrete probability distributions depending only on the preceding state S<sub>t</sub> & action A<sub>t</sub>
* Given a random state s' ∈ S and reward r ∈ R the probability of s' and r occuring at time t given a preceding state s and action a is given by the four argument MDP <strong>dynamics function</strong> (<math xmlns='http://www.w3.org/1998/Math/MathML'> <semantics>  <mi>p</mi><mo>:</mo><mi>S</mi><mo>&#x00D7;</mo><mi>R</mi><mo>&#x00D7;</mo><mi>S</mi><mo>&#x00D7;</mo><mi>A</mi><mo>&#x2192;</mo><mo stretchy='false'>[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy='false'>]</mo> </semantics></math>)
&emsp;<math xmlns='http://www.w3.org/1998/Math/MathML'> <semantics>
  <mrow>
   <mi>p</mi><mo stretchy='false'>(</mo><mi>s</mi><mo>&#x0027;</mo><mo>,</mo><mi>r</mi><mo>&#x007C;</mo><mi>s</mi><mo>,</mo><mi>a</mi><mo stretchy='false'>)</mo><mo>&#x2250;</mo><mi>Pr</mi><mo>&#x007B;</mo><msub>
    <mi>S</mi><sub>
    <mi>t</mi></sub>
   </msub>
   <mo>=</mo><mi>s</mi><mo>&#x0027;</mo><mo>,</mo><msub>
    <mi>R</mi><sub>
    <mi>t</mi></sub>
   </msub>
   <mo>=</mo><mi>r</mi><mo>&#x007C;</mo><msub>
    <mi>S</mi><sub>
     <mi>t+1</mi></sub>
   </msub>
   <mo>=</mo><mi>s</mi><mo>,</mo><msub>
    <mi>A</mi>
    <mrow>
     <mi>t</mi><mo>&#x2212;</mo><mn>1</mn></mrow>
   </msub>
   <mo>=</mo><mi>a</mi><mo>&#x007D;</mo><mo>&#x2200;</mo><mi>s</mi><mo>&#x0027;</mo><mo>,</mo><mi>s</mi><mo>&#x2208;</mo><mi>S</mi><mo>,</mo><mi>r</mi><mo>&#x2208;</mo><mi>R</mi><mo>,</mo><mi>a</mi><mo>&#x2208;</mo><mi>A</mi><mo stretchy='false'>(</mo><mi>s</mi><mo stretchy='false'>)</mo></mrow>
 </semantics>
</math>
From the <em>dynamics</em> function <em>p</em> we can derive other useful functions:



## Plot of Rewards
With the above parameters, the agent was able to solve the game (average reward over 100 episodes >2000) in 1198 iterations.


Below is the 100 episode average reward per iteration, as well as the objective value per iteration.

![training_log](./images/return_and_objective_vs_step.png)
