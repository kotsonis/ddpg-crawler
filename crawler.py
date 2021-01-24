import os
from absl import app
from absl import logging
from absl import flags

from unityagents import UnityEnvironment
from agents.spdg import SDPGAgent
from utils import replay

config = flags.FLAGS



def main(argv):
    del argv
    if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
    logging.get_absl_handler().use_absl_log_file()
    logging.set_verbosity('debug')
    # modify some parameters of training
    env = UnityEnvironment(file_name=config.env, worker_id = 1)
    model = SDPGAgent(device=config.device,env=env, replay_buffer_class=replay.PriorityReplay)
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
    app.run(main)