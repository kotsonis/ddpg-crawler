

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

# ddpg-crawler
A PPO RL solution to the Unity-ML(Udacity) [Crawler](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler) environment. 

## Introduction

## The Environment

![Trained Agent][image1]

Crawler is a creature with 4 arms and 4 forearms, which needs to learn how to stand and walk forward without falling. The environment has 12 agents, and each one controls the target rotations for joints and heads of a cralwer through 20 continous actions.

The state consists of 129 float values representing position, rotation, velocity, and angular velocities of each limb.

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
	git clone https://github.com/kotsonis/ddpg-crawler.git
	cd ddpg-crawler
	pip install -r requirements.txt
	pip install tensorflow==1.15
```

### Install Crawler environment
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

	- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
	- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
	- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
	- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

2. Place the file in the `ddpg-crawler` folder, and unzip (or decompress) the file.

## Instructions
### Training
[crawler.py](crawler.py) reads the hyperparameters from the command line options to modify parameters and/or set saving options.You can get the CLI options by running
```bash
python train.py -h
```
A typical training invokation is provided below:
```bash
python crawler.py --train --trajectories 2001 --policy_optimization_epochs 160 \
		  --entropy_beta 0.002 --vf_coeff 0.05 --memory_batch_size 512 --actor_lr 8e-5 --gamma 0.95 \
		  --env [path to Crawler environment]
```
or, with default parameters:
```bash
pythom crawler.py --train
```

### Playing with a trained model
you can see the agent playing with the trained model as follows:
```bash
python crawler.py --play --notb
```
You can also specify the number of episodes you want the agent to play, as well as the non-default trained model as follows:
```bash
python crawler.py --play --notb --load ./model/model_saved.pt --episodes 20
```

## Implementation and results
You can read about the implementation details and the results obtained in [Report.md](Report.md)
