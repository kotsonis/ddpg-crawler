from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import os as os
import config as config
from agent import DPG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hyper_params = config.Configuration()
#hyper_params.process_CLI(argv)

#env = UnityEnvironment(file_name=hyper_params.reacher_location)
env = UnityEnvironment(file_name='../../deep-reinforcement-learning/p2_continuous-control/Crawler_Windows_x86_64/Crawler.exe')

hyper_params.process_env(env)
num_agents = hyper_params.num_agents
action_size = hyper_params.action_size
brain_name = hyper_params.brain_name
n_episodes = hyper_params.n_episodes
solution = 31
solution_found = False

# create DPG Actor/Critic Agent
agent = DPG(hyper_params)

if(not hyper_params.model_dir): hyper_params.model_dir = './model/'
# load trained agent
agent.load_models(hyper_params.model_dir)

scores = []                                 # list containing scores from each episode
scores_window = deque(maxlen=100)           # last 100 scores

agent_scores = np.zeros(num_agents)                          # initialize the score (for each agent)
frames = 0

for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    states = env_info.vector_observations             # get the current state of each agent
    agent_scores = np.zeros(num_agents)
    frames = 0
    while True:
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]          # send all actions to tne environment
        next_states = env_info.vector_observations        # get next state (for each agent)
        rewards = env_info.rewards                        # get reward (for each agent)
        dones = env_info.local_done                       # see if episode finished
        agent_scores += rewards                           # update the score (for each agent)
        states = next_states                              # roll over states to next time step
        frames = frames+1
        if frames % 10 == 0:
            print('\rEpisode {}\t Frame: {:4}/1000 \t Score: {:.2f}'.format(i_episode, frames, np.mean(agent_scores)), end="")
        if np.any(dones):                                 # exit loop if episode finished
            break
    scores.append(np.mean(agent_scores))              # store episodes mean reward over agents
    scores_window.append(np.mean(agent_scores))       # save most recent score
    
    if i_episode % 10 == 0:
        print('\rEpisodes: {}\tAverage Score: {:.2f}\t\t'.format(i_episode, np.mean(scores_window)))
    if np.mean(scores_window)>=solution:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        solution_num_episodes = i_episode-100

env.close()
