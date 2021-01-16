import os
from absl import app
from absl import logging
from absl import flags
from unityagents import UnityEnvironment
from mpo import MPO
#import numpy as np
#import torch
#import torch.nn as nn
#from tensorboardX import SummaryWriter
#from tqdm import tqdm

config = flags.FLAGS
flags.DEFINE_string(name='device', default='cpu',
                        help="Device to use for torch")
flags.DEFINE_string(name='env', default='../../deep-reinforcement-learning/p2_continuous-control/Crawler_Windows_x86_64/Crawler.exe',
                        help='Unity Environment to load')
flags.DEFINE_boolean(name='render', default=True, help="execute Unity Enviroment with display")
flags.DEFINE_boolean(name='debug', default=None, help="run in debug mode")
flags.DEFINE_string(name='load',default=None,
                        help='model file to load with path')
flags.DEFINE_bool(name='play', default=None,
                        help='play environment with model')
flags.DEFINE_bool(name='train', default=None, 
                        help='train the agent')
flags.DEFINE_integer(name='episodes', default=20,
                        help='number of episodes to run')


def main(argv):
    del argv
    logging.get_absl_handler().use_absl_log_file()
    # modify some parameters of training
    env = UnityEnvironment(file_name=config.env)
    model = MPO(
        config.device,
        env,
        dual_constraint=config.dual_constraint,
        kl_mean_constraint=config.kl_mean_constraint,
        kl_var_constraint=config.kl_var_constraint,
        kl_constraint=config.kl_constraint,
        discount_factor=config.discount_factor,
        alpha=config.alpha,
        sample_process_num=config.sample_process_num,
        sample_episode_num=config.sample_episode_num,
        sample_episode_maxlen=config.sample_episode_maxlen,
        sample_action_num=config.sample_action_num,
        batch_size=config.batch_size,
        episode_rerun_num=config.episode_rerun_num,
        lagrange_iteration_num=config.lagrange_iteration_num)

    if config.load is not None:
        model.load_model(config.load)
    if config.play is not None:
        model.play(
            episodes= config.episodes,
            frames = 1000)
    if config.train is not None:
        model.train(
            iteration_num=config.iteration_num,
            log_dir=config.log_dir,
            render=config.render,
            debug=config.debug)
    env.close()

if __name__ == '__main__':
    app.run(main)