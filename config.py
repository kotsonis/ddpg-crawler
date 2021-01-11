import numpy as np
import torch
import torch.nn as nn
import torch.functional as F 
from unityagents import UnityEnvironment
import sys, getopt # for commandline argument parsing


class Configuration():
    def __init__(self):
        self.gamma = 0.99
        self.soft_update_tau = 1e-3
        self.eps_start = 0.99
        self.eps_decay_rate = 0.9995
        self.epsilon_min = 1e-4
        self.PER_batch_size = 64
        self.PER_alpha = 0.6
        self.n_step = 3
        self.PER_buffer = 1e6
        self.critic_state_layers = 1
        self.critic_action_layers = 3
        self.critic_hidden_units = 256
        self.PER_eps = 1e-05
        self.reacher_location = './Reacher_Windows_x86_64/Reacher.exe'
        self.PER_beta_start = 0.4
        self.PER_beta_decay = 0.000025/4.0
        self.PER_beta_max = 1.0
        self.update_every = 4
        self.num_atoms = 51
        self.dense1_size = 400
        self.dense2_size = 300
        self.log_dir = '.'
        self.model_dir = False
        self.plt_file = False
        
        return
    
    def process_env(self,env):
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        env_info = env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.action_size = self.brain.vector_action_space_size
        states = env_info.vector_observations
        self.state_size = states.shape[1]
        return
    
    def process_CLI(self,argv):
        try:
            opts, args = getopt.getopt(argv,
                                   ":h",
                                   [
                                       "eps-start=",
                                       "eps-decay=",
                                       "batch-size=",
                                       "episodes=",
                                       "reward-early-stop=",
                                       "save-model_dir=",
                                       "output-image=",
                                       "gamma=",
                                       "tau=",
                                       "beta-start=",
                                       "beta-decay=",
                                       "alpha=",
                                       "reacher_location=",
                                       "help"
                                    ])
        except getopt.GetoptError as err:
            # print help information and exit:
            print(err)  # will print something like "option -a not recognized"
            usage()
            sys.exit(2)
        hyperparams = std_learn_params

        for o, a in opts:
            if o == "--eps-start":
                self.eps_start = float(a)
            elif o == "--eps-decay":
                self.eps_decay_rate = float(a)
            elif o == "--batch-size":
                self.PER_batch_size = int(a)
            elif o == "--episodes":
                self.n_episodes = int(a)
            elif o == "--reward-early-stop":
                self.early_stop = float(a)
            elif o == "--tau":
                self.soft_update_tau = float(a)
            elif o == "--save-model_dir":
                self.model_dir = a
            elif o == "--output-image":
                self.plt_file = a
            elif o == "--gamma":
                self.gamma = float(a)
            elif o == "--beta-start":
                self.PER_beta_start =float(a)
            elif o == "--beta-decay":
                self.PER_beta_decay =float(a)
            elif o == "--alpha":
                self.PER_alpha = float(a)
            elif o in ("--reacher_location"):
                self.reacher_location =a
            elif o == "--help":
                usage()
                sys.exit(2)
            else:
                assert False, "unhandled option"
        # return the modified hyperparams
        return hyperparams



       

