from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import os as os

from .agents.base_agent import DPG
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = UnityEnvironment(file_name='../../deep-reinforcement-learning/p2_continuous-control/Crawler_Windows_x86_64/Crawler.exe')
#env = UnityEnvironment(file_name='./Reacher_Windows_x86_64/Reacher.exe', worker_id=1, no_graphics=True)
hyper_params = config.Configuration()
hyper_params.process_env(env)
hyper_params.n_step = 15
hyper_params.PER_batch_size = 64 #16
#hyper_params.PER_batch_size = 2 #16
num_agents = hyper_params.num_agents
action_size = hyper_params.action_size
brain_name = hyper_params.brain_name
n_episodes = 5000
n_frames = 1000
hyper_params.update_every = 4
hyper_params.eps_start = 1.0
hyper_params.epsilon_min = 0.01 # 1e-2
hyper_params.eps_decay_rate = 0.9999
#hyper_params.num_atoms = 5
hyper_params.num_atoms = 51
hyper_params.dense1_size = 400 #400
hyper_params.dense2_size = 300 #300
solution = 1200
solution_found = False
total_train_steps = 3e5
#hyper_params.PER_beta_start = 0.6
#hyper_params.PER_beta_decay = 0.025/4.0
#hyper_params.PER_beta_max = 0.9
#hyper_params.PER_alpha = 1



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
i_episode = 0
t_step = 0
step = 0
agent.noise.reset()
while t_step < total_train_steps:
    i_episode += 1
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    states = env_info.vector_observations             # get the current state of each agent
    agent_scores = np.zeros(num_agents)
    frames = 0
    agent.noise.reset()
    while True: # each frame
        actions = agent.act(states, add_noise=True)
        step += 1
        env_info = env.step(actions)[brain_name]          # send all actions to tne environment
        next_states = env_info.vector_observations        # get next state (for each agent)
        
        rewards = env_info.rewards                        # get reward (for each agent)
        if (np.any(np.isnan(rewards))): 
            print('got a NaN reward. Need to fix it.')
        rewards = np.nan_to_num(rewards,nan=-5.0)
        
        
        dones = env_info.local_done                       # see if episode finished
        #if frames < 999: rewards += np.array(dones)*-5.0
        t_step = agent.step(states, actions, rewards, next_states, dones)
        agent_scores += rewards                           # update the score (for each agent)
        states = next_states                              # roll over states to next time step
        frames = frames+1
        if frames % 20 == 0:
            print('\rEpisode {}\t Frame: {:4}/1000 \t Score: {:.2f} \t Training to go: {:.0f}, eps: {:.2f}'.format(i_episode, frames, np.mean(agent_scores), total_train_steps - t_step, agent.eps), end="")
        
        if np.any(dones):                                 # exit loop if episode finished
            break
    scores.append(np.mean(agent_scores))              # store episodes mean reward over agents
    scores_window.append(np.mean(agent_scores))       # save most recent score
    
    if i_episode % 100 == 0:
        print('\rEpisodes: {}\t 100 episode mean score: {:.2f}\t training to go: {:.0f}, eps: {:.2f}'.format(
            i_episode, np.mean(scores_window), total_train_steps - t_step, agent.eps ))
        agent.save_models(hyper_params.model_dir)
    if np.mean(scores_window)>=solution:
        if ( not solution_found):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            solution_num_episodes = i_episode-100
            solution_found = True
            break

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
