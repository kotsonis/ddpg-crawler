import os
from absl import app
from absl import logging
from absl import flags
import multiprocessing as mp

from unityagents import UnityEnvironment
from agents.spdg import SDPGAgent
from utils import replay

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
    if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
    logging.get_absl_handler().use_absl_log_file()
    logging.set_verbosity('debug')
    # modify some parameters of training
    env = UnityEnvironment(file_name=config.env, worker_id = 2)
    model = SDPGAgent(device=config.device,env=env, replay_buffer=replay.PriorityReplay)
    if config.load is not None:
        model.load_model(load_model = config.load)
    if config.play is not None:
        model.play(episodes= config.episodes,frames = 1000)
    if config.train is not None:
        model.train(
            training_iterations=config.training_iterations,
            log_dir=config.log_dir,
            render=config.render,
            debug=config.debug)
    env.close()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    app.run(main)