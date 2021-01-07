

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# ddpg-reacher
A Deep Deterministic Policy Gradient Actor-Critic reinforcement learning solution to the Unity-ML(Udacity) [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. 

## Introduction

## The Environment

![Trained Agent][image1]

Reacher is an environment in which 20 agents control a double-jointed arm each to move to a target location. The target (goal location) is moving and each agent receives a reward of +0.1 for each step that the agent's hand is in the goal location. Thus, the goal of each agent is to maintain its position at the target location for as many time steps as possible.


**Set-up**: Double-jointed arm which can move to target locations.

**Goal**: The agents must move its hand to the goal location, and keep it there.

**Agents**: The environment contains 20 agent with same Behavior Parameters.

**Agent Reward Function** (_agent independent_):
+0.1 Each step agent's hand is in goal location.
##### Behavior Parameters:
- **_Vector Observation space_ (State Space)**: 26 variables corresponding to position, rotation, velocity, and angular velocities of the two arm rigid bodies.
- **_Actions_ (Action Space)**: 4 continuous actions, corresponding to torque applicable to two joints.

**Benchmark Mean Reward**: 30
**Turns**: An episode completes after **1000 frames**

## Getting Started
To set up your python environment and run the code in this repository, follow the instructions below.
### setup Conda Python environment

Create (and activate) a new environment with Python 3.6.

- __Linux__ or __Mac__: 
```shell
	conda create --name ddpg-rl python=3.6
	source activate ddpg-rl
```
- __Windows__: 
```bash
	conda create --name ddpg-rl python=3.6 
	activate ddpg-rl
```
### Download repository
 Clone the repository and install dependencies

```shell
	git clone https://github.com/kotsonis/ddpg-reacher.git
	cd ddpg-reacher
	pip install -r requirements.txt
```

### Install Reacher environment
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the `ddpg-reacher` folder, and unzip (or decompress) the file.

3. edit [hyperparams.py](hyperparams.py) to and set the `banana_location` entry to point to the right location. Example :
```python 
std_learn_params = {
        # Unity Environment parameters
        "banana_location": "./Banana_Windows_x86_64/Banana.exe",
```
## Instructions
### Training
To train an agent, [train.py](train.py) reads the hyperparameters from [hyperparams.py](hyperparams.py) and accepts command line options to modify parameters and/or set saving options.You can get the CLI options by running
```bash
python train.py -h
```
### Playing with a trained model
you can see the agent playing with the trained model as follows:
```bash
python play.py
```
You can also specify the number of episodes you want the agent to play, as well as the non-default trained model as follows:
```bash
python play.py --episodes 20 --model v2_model.pt
```

## Implementation and results
You can read about the implementation details and the results obtained in [Report.md](Report.md)
