import os
from absl import app
from absl import logging
from absl import flags

from unityagents import UnityEnvironment
from agents.ppo import PPOAgent

config = flags.FLAGS

flags.DEFINE_string(name='env', default='../../deep-reinforcement-learning/p2_continuous-control/Crawler_Windows_x86_64/Crawler.exe',
                    help='Unity Environment to load')
flags.DEFINE_boolean(name='render', default=False, help="execute Unity Enviroment with display")
flags.DEFINE_string(name='load',default=None,
                    help='model file to load with path')
flags.DEFINE_bool(name='play', default=None,
                  help='play environment with model')
flags.DEFINE_bool(name='train', default=None, 
                  help='train the agent')
flags.DEFINE_integer(name='episodes', default=20,
                     help='number of episodes to run')
flags.DEFINE_float(name='gamma',default=0.995,
                   help='discount factor for future rewards (0,1]')
flags.DEFINE_integer(name='trajectories',default=2048,
                     help='number of trajectories to sample per iteration')
flags.DEFINE_integer(name='policy_optimization_epochs', default=160,
                     help='number of epochs to run (K in paper)')
flags.DEFINE_float(name='policy_stopping_kl', default=0.3,
                   help='log KL divergence to early stop PPO improvements')
flags.DEFINE_float(name='policy_clip_range', default=0.2,
                   help='clipping threshold for PPO policy optimization')
flags.DEFINE_float(name='gae_lambda', default=0.85,
                   help='lambda coefficient for generalized advantage estimate')
flags.DEFINE_float(name='entropy_beta', default=0.002,
                   help='coefficient to multiply beta loss in PPO step')
flags.DEFINE_float(name='vf_coeff', default=0.05,
                   help='coefficient to multiply value loss in PPO step')
flags.DEFINE_integer(name='memory_batch_size',default=512,
                     help='batch size of memory samples per epoch')
flags.DEFINE_bool(name='tb', default=True,
                  help='enable tensorboard logging')
flags.DEFINE_string(name='device', default='cpu',
                    help="Device to use for torch")

def main(argv):
    del argv
    if config.log_dir != '':
        if not os.path.exists(config.log_dir):
                os.makedirs(config.log_dir)
        logging.get_absl_handler().use_absl_log_file()
        logging.set_verbosity('info')
    env = UnityEnvironment(file_name=config.env, worker_id = 2, no_graphics=config.render)
    model = PPOAgent(device=config.device,env=env)
    if config.load is not None:
        model.load_model(load_model = config.load)
    if config.play is not None:
        model.play(episodes= config.episodes)
    if config.train is not None:
        model.train(
            training_iterations=config.training_iterations,
            log_dir=config.log_dir,
            render=config.render)
    env.close()

if __name__ == '__main__':
    app.run(main)