from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import os as os

import config as config
from agent import DPG
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = UnityEnvironment(file_name='../../deep-reinforcement-learning/p2_continuous-control/Crawler_Windows_x86_64/Crawler.exe')
#env = UnityEnvironment(file_name='../../deep-reinforcement-learning/p2_continuous-control/Reacher_Windows_x86_64/Reacher.exe')
hyper_params = config.Configuration()
hyper_params.process_env(env)
hyper_params.n_step = 75
hyper_params.PER_batch_size = 32
num_agents = hyper_params.num_agents
action_size = hyper_params.action_size
brain_name = hyper_params.brain_name
n_episodes = 5000
n_frames = 1000
hyper_params.update_every = 2
hyper_params.eps_start = 0.5
hyper_params.epsilon_min = 1e-2
solution = 30
solution_found = False

# create DPG Actor/Critic Agent
agent = DPG(hyper_params)
hyper_params.model_dir = "./model/"
if (hyper_params.model_dir) :
    if not os.path.exists(hyper_params.model_dir):
            os.makedirs(hyper_params.model_dir)
#agent.load_models(hyper_params.model_dir)
scores = []                                 # list containing scores from each episode
scores_window = deque(maxlen=100)           # last 100 scores

agent_scores = np.zeros(num_agents)                          # initialize the score (for each agent)
frames = 0

for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    states = env_info.vector_observations             # get the current state of each agent
    agent_scores = np.zeros(num_agents)
    frames = 0
    for _ in range(n_episodes):
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]          # send all actions to tne environment
        next_states = env_info.vector_observations        # get next state (for each agent)
        
        rewards = env_info.rewards                        # get reward (for each agent)
        
        dones = env_info.local_done                       # see if episode finished
        agent.step(states, actions, rewards, next_states, dones, frames
        )
        agent_scores += rewards                           # update the score (for each agent)
        states = next_states                              # roll over states to next time step
        frames = frames+1
        if frames % 10 == 0:
            print('\rEpisode {}\t Frame: {:4}/1000 \t Score: {:.2f}'.format(i_episode, frames, np.mean(agent_scores)), end="")
        if np.any(dones):                                 # exit loop if episode finished
            break
    scores.append(np.mean(agent_scores))              # store episodes mean reward over agents
    scores_window.append(np.mean(agent_scores))       # save most recent score
    
    if i_episode % 100 == 0:
        print('\rEpisodes: {}\tAverage Score: {:.2f}\t\t'.format(i_episode, np.mean(scores_window)))
        agent.save_models(hyper_params.model_dir)
    if np.mean(scores_window)>=solution:
        if ( not solution_found):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            solution_num_episodes = i_episode-100
            solution_found = True

# save model if requested
hyper_params.model_dir = "./model/"
if (hyper_params.model_dir) :
    if not os.path.exists(hyper_params.model_dir):
            os.makedirs(hyper_params.model_dir)
    agent.save_models(hyper_params.model_dir)

#plot results
plt.figure()
plt.title('Mean Score across Agents per Episode')
plt.xlabel("Episodes")
plt.ylabel("Mean Score")
episodes=range(len(scores))
plt.plot(episodes, scores)
plt.legend(loc='lower right')

hyper_params.plt_file = "results"
if (hyper_params.plt_file):
    plt.savefig(hyper_params.plt_file)
    plt.show()
else:
    plt.show()

env.close()
