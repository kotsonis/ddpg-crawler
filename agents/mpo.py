#ref daisatojp

import os
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from ..networks import actor
from ..networks import critic
from utils import replaybuffer

from absl import app
from absl import logging
from absl import flags

config = flags.FLAGS
flags.DEFINE_float(name='dual_constraint', default=0.1, 
                        help='hard constraint of the E-step')
flags.DEFINE_float(name='kl_mean_constraint', default=0.01, 
                        help='hard constraint of the E-step')
flags.DEFINE_float(name='kl_var_constraint',default=1e-4,
                        help='hard constraint on variance parameter')
flags.DEFINE_float(name='kl_constraint', default=0.01,
                        help='hard constraint on variance parameter')
flags.DEFINE_float(name='discount_factor',default=0.99,
                        help='discount factor')
flags.DEFINE_float(name='alpha', default=10.0,
                        help='scaling factor of the lagrangian multiplier in the M-step')
flags.DEFINE_integer(name='sample_process_num',default=5,
                        help='number of training steps per sample')
flags.DEFINE_integer(name='sample_episode_num', default=30,
                        help='number of episodes to learn')
flags.DEFINE_integer(name='sample_episode_maxlen', default=200,
                        help='length of an episode (number of training steps)')
flags.DEFINE_integer(name='sample_action_num', default=64,
                        help='number of sampled actions')
flags.DEFINE_integer(name='batch_size', default=64,
                        help='batch size to sample for learning step')
flags.DEFINE_integer(name='iteration_num', default=1000,
                        help='number of iteration to learn')
flags.DEFINE_integer(name='lagrange_iteration_num', default=5,
                        help='number of optimization steps of the Lagrangian')
flags.DEFINE_integer(name='episode_rerun_num', default=5,
                        help='number of reruns of sampled episode')
flags.DEFINE_boolean(name='multiprocessing',default=True,
                        help='run with multiprocessing')
#flags.DEFINE_string(name='log_dir', default='.',help='log directory')


def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def gaussian_kl(μi, μ, Ai, A):
    """
    decoupled KL between two multivariate gaussian distribution
    C_μ = KL(f(x|μi,Σi)||f(x|μ,Σi))
    C_Σ = KL(f(x|μi,Σi)||f(x|μi,Σ))
    :param μi: (B, n)
    :param μ: (B, n)
    :param Ai: (B, n, n)
    :param A: (B, n, n)
    :return: C_μ, C_Σ: mean and covariance terms of the KL
    """
    n = A.size(-1)
    μi = μi.unsqueeze(-1)  # (B, n, 1)
    μ = μ.unsqueeze(-1)  # (B, n, 1)
    Σi = Ai @ bt(Ai)  # (B, n, n)
    Σ = A @ bt(A)  # (B, n, n)
    Σi_inv = Σi.inverse()  # (B, n, n)
    Σ_inv = Σ.inverse()  # (B, n, n)
    inner_μ = ((μ - μi).transpose(-2, -1) @ Σi_inv @ (μ - μi)).squeeze()  # (B,)
    inner_Σ = torch.log(Σ.det() / Σi.det()) - n + btr(Σ_inv @ Σi)  # (B,)
    C_μ = 0.5 * torch.mean(inner_μ)
    C_Σ = 0.5 * torch.mean(inner_Σ)
    return C_μ, C_Σ


class MPO(object):
    """
    Maximum A Posteriori Policy Optimization (MPO)
    :param device:
    :param env: Unity environment
    :param dual_constraint: (float) hard constraint of the dual formulation in the E-step
    :param kl_mean_constraint: (float) hard constraint of the mean in the M-step
    :param kl_var_constraint: (float) hard constraint of the covariance in the M-step
    :param kl_constraint:
    :param discount_factor: (float) learning rate in the Q-function
    :param alpha: (float) scaling factor of the lagrangian multiplier in the M-step
    :param sample_process_num:
    :param sample_episode_num:
    :param sample_episode_maxlen:
    :param sample_action_num:
    :param batch_size: (int) size of the sampled mini-batch
    :param episode_rerun_num:
    :param lagrange_iteration_num: (int) number of optimization steps of the Lagrangian
    :param sample_episodes: (int) number of sampling episodes
    :param add_act: (int) number of additional actions
    """
    def __init__(self,
                 device,
                 env,
                 dual_constraint=0.1,
                 kl_mean_constraint=0.01,
                 kl_var_constraint=0.01,
                 kl_constraint=0.01,
                 discount_factor=0.99,
                 alpha=10,
                 sample_process_num=5,
                 sample_episode_num=30,
                 sample_episode_maxlen=200,
                 sample_action_num=64,
                 batch_size=64,
                 episode_rerun_num=5,
                 lagrange_iteration_num=5,
                 multiprocessing=False):
        self.device = device
        self.env = env
        self.continuous_action_space = True
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[env.brain_names[0]]
        self.da = self.brain.vector_action_space_size
        self.ds = self.brain.vector_observation_space_size
        
        self.ε_dual = dual_constraint
        self.ε_kl_μ = kl_mean_constraint  # hard constraint for the KL
        self.ε_kl_Σ = kl_var_constraint  # hard constraint for the KL
        self.ε_kl = kl_constraint
        self.γ = discount_factor
        self.α = alpha  # scaling factor for the update step of η_μ
        self.sample_process_num = sample_process_num
        self.sample_episode_num = sample_episode_num
        self.sample_episode_maxlen = sample_episode_maxlen
        self.sample_action_num = sample_action_num
        self.batch_size = batch_size
        self.episode_rerun_num = episode_rerun_num
        self.lagrange_iteration_num = lagrange_iteration_num
        self.multiprocessing = multiprocessing

        self.actor = ActorContinuous(self.ds, self.da).to(self.device)
        self.critic = CriticContinuous(self.ds, self.da).to(self.device)
        self.target_actor = ActorContinuous(self.ds, self.da).to(self.device)
        self.target_critic = CriticContinuous(self.ds, self.da).to(self.device)
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.norm_loss_q = nn.SmoothL1Loss()

        self.η = np.random.rand()
        self.η_kl_μ = 0.0
        self.η_kl_Σ = 0.0
        self.η_kl = 0.0

        self.replaybuffer = ReplayBuffer()

        self.iteration = 0
        self.render = False

    def play(self,
             episodes= 10,
             frames = 1000):
        self.sample_episode_maxlen = frames
        self.replaybuffer.clear()
        buff = []
        #logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
        print('logs should be stored here: {}'.format(logging.find_log_dir()))
        
        logging.info('Starting a Play session')
        logging.info('Will run for %d episodes', episodes)
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        num_agents = len(env_info.agents)
        states = env_info.vector_observations
        scores = []                                 # list containing scores from each episode
        agent_scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        for it in range(0, episodes):
            agent_scores = np.zeros(num_agents)                          # initialize the score (for each agent)
            frames = 0
            for steps in range(self.sample_episode_maxlen):
                actions = self.target_actor.action(
                    torch.from_numpy(states).type(torch.float32).to(self.device)
                ).cpu().numpy()
            
                env_info = self.env.step(actions)[self.brain_name]          # send all actions to tne environment
                next_states = env_info.vector_observations        # get next state (for each agent)
                rewards = env_info.rewards                        # get reward (for each agent)
                rewards = np.nan_to_num(rewards,nan=-5.0)         # fix NaN rewards of crawler environment, by penalizing a NaN reward
                dones = env_info.local_done                       # see if episode finished
                frames += 1
                agent_scores += rewards                           # update the score (for each agent)
                if frames % 20 == 0:
                    print('\rEpisode {}\t Frame: {:4}/1000 \t Score: {:.2f}'
                        .format(it, frames, np.mean(agent_scores)), end="")
                if np.any(dones):
                    break
                else:
                    states = next_states
            scores.append(np.mean(agent_scores))              # store episodes mean reward over agents
    
            print('\rEpisodes: {}\tscore:{}\t running mean score: {:.2f}'
                .format(
                        it,
                        np.mean(agent_scores),
                        np.mean(scores)))
                
    def train(self,
              iteration_num=100,
              log_dir='log',
              model_save_period=10,
              render=False,
              debug=False):
        """
        :param iteration_num:
        :param log_dir:
        :param model_save_period:
        :param render:
        :param debug:
        """

        self.render = render
        print('log dir: {}'.format(log_dir))
        model_save_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        writer = SummaryWriter(os.path.join(log_dir, 'tb'))

        for it in range(self.iteration, iteration_num):
            self.__sample_trajectory(self.sample_episode_num)
            buff_sz = len(self.replaybuffer)

            mean_reward = self.replaybuffer.mean_reward()
            mean_return = self.replaybuffer.mean_return()
            mean_loss_q = []
            mean_loss_p = []
            mean_loss_l = []
            mean_est_q = []
            max_kl_μ = []
            max_kl_Σ = []
            max_kl = []

            # Find better policy by gradient descent
            for r in range(self.episode_rerun_num):
                for indices in tqdm(
                        BatchSampler(
                            SubsetRandomSampler(range(buff_sz)), self.batch_size, False),
                        desc='training {}/{}'.format(r+1, self.episode_rerun_num)):
                    B = len(indices)
                    M = self.sample_action_num
                    ds = self.ds
                    da = self.da

                    state_batch, action_batch, next_state_batch, reward_batch = zip(
                        *[self.replaybuffer[index] for index in indices])

                    state_batch = torch.from_numpy(np.stack(state_batch)).type(torch.float32).to(self.device)  # (B, ds)
                    action_batch = torch.from_numpy(np.stack(action_batch)).type(torch.float32).to(self.device)  # (B, da) or (B,)
                    next_state_batch = torch.from_numpy(np.stack(next_state_batch)).type(torch.float32).to(self.device)  # (B, ds)
                    reward_batch = torch.from_numpy(np.stack(reward_batch)).type(torch.float32).to(self.device)  # (B,)

                    # Policy Evaluation
                    # optimize critic by sampling next actions from state and minimizing loss between critic_target and critic
                    loss_q, q = self.__update_critic_td(
                        state_batch=state_batch,
                        action_batch=action_batch,
                        next_state_batch=next_state_batch,
                        reward_batch=reward_batch,
                        sample_num=self.sample_action_num
                    )
                    if loss_q is None:
                        raise RuntimeError('invalid policy evaluation')
                    mean_loss_q.append(loss_q.item())
                    mean_est_q.append(q.abs().mean().item())

                    # sample M additional action for each state
                    with torch.no_grad():
                        b_μ, b_A = self.target_actor.forward(state_batch)  # (B,)
                        b = MultivariateNormal(b_μ, scale_tril=b_A)  # (B,)
                        sampled_actions = b.sample((M,))  # (M, B, da)
                        expanded_states = state_batch[None, ...].expand(M, -1, -1)  # (M, B, ds)
                        target_q = self.target_critic.forward(
                            expanded_states.reshape(-1, ds),  # (M * B, ds)
                            sampled_actions.reshape(-1, da)  # (M * B, da)
                        ).reshape(M, B)  # (M, B)
                        target_q_np = target_q.cpu().numpy()  # (M, B)

                    # E-step (Expectation)
                    def dual(η):
                        """
                        dual function of the non-parametric variational
                        g(η) = η*ε + η \sum \log (\sum \exp(Q(a, s)/η))
                        """
                        max_q = np.max(target_q_np, 0)
                        return η * self.ε_dual + np.mean(max_q) \
                            + η * np.mean(np.log(np.mean(np.exp((target_q_np - max_q) / η), axis=0)))

                    #define the bounds for the scipy minimize  function with Sequential Least Squares Programming (SLSQP) method
                    bounds = [(1e-6, None)]
                    # TODO: remove dependency on scipy ...
                    res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
                    self.η = res.x[0]

                    qij = torch.softmax(target_q / self.η, dim=0)  # (M, B) or (da, B)

                    # M-step
                    # update policy based on lagrangian
                    for _ in range(self.lagrange_iteration_num):
                    
                        μ, A = self.actor.forward(state_batch)
                        π1 = MultivariateNormal(loc=μ, scale_tril=b_A)  # (B,)
                        π2 = MultivariateNormal(loc=b_μ, scale_tril=A)  # (B,)
                        loss_p = torch.mean(
                            qij * (
                                π1.expand((M, B)).log_prob(sampled_actions)  # (M, B)
                                + π2.expand((M, B)).log_prob(sampled_actions)  # (M, B)
                            )
                        )
                        mean_loss_p.append((-loss_p).item())

                        kl_μ, kl_Σ = gaussian_kl(
                            μi=b_μ, μ=μ,
                            Ai=b_A, A=A)
                        max_kl_μ.append(kl_μ.item())
                        max_kl_Σ.append(kl_Σ.item())

                        if debug and np.isnan(kl_μ.item()):
                            print('kl_μ is nan')
                            embed()
                        if debug and np.isnan(kl_Σ.item()):
                            print('kl_Σ is nan')
                            embed()

                        # Update lagrange multipliers by gradient descent
                        self.η_kl_μ -= self.α * (self.ε_kl_μ - kl_μ).detach().item()
                        self.η_kl_Σ -= self.α * (self.ε_kl_Σ - kl_Σ).detach().item()

                        if self.η_kl_μ < 0.0:
                            self.η_kl_μ = 0.0
                        if self.η_kl_Σ < 0.0:
                            self.η_kl_Σ = 0.0

                        self.actor_optimizer.zero_grad()
                        loss_l = -(
                                loss_p
                                + self.η_kl_μ * (self.ε_kl_μ - kl_μ)
                                + self.η_kl_Σ * (self.ε_kl_Σ - kl_Σ)
                        )
                        mean_loss_l.append(loss_l.item())
                        loss_l.backward()
                        clip_grad_norm_(self.actor.parameters(), 0.1)
                        self.actor_optimizer.step()
                        
            self.__update_param()

            self.η_kl_μ = 0.0
            self.η_kl_Σ = 0.0
            self.η_kl = 0.0

            it = it + 1
            mean_loss_q = np.mean(mean_loss_q)
            mean_loss_p = np.mean(mean_loss_p)
            mean_loss_l = np.mean(mean_loss_l)
            mean_est_q = np.mean(mean_est_q)
            if self.continuous_action_space:
                max_kl_μ = np.max(max_kl_μ)
                max_kl_Σ = np.max(max_kl_Σ)
            else:
                max_kl = np.max(max_kl)

            print('iteration :', it)
            print('  mean return :', mean_return)
            print('  mean reward :', mean_reward)
            print('  mean loss_q :', mean_loss_q)
            print('  mean loss_p :', mean_loss_p)
            print('  mean loss_l :', mean_loss_l)
            print('  mean est_q :', mean_est_q)
            print('  η :', self.η)
            if self.continuous_action_space:
                print('  max_kl_μ :', max_kl_μ)
                print('  max_kl_Σ :', max_kl_Σ)
            else:
                print('  max_kl :', max_kl)

            # saving and logging
            self.save_model(os.path.join(model_save_dir, 'model_latest.pt'))
            if it % model_save_period == 0:
                self.save_model(os.path.join(model_save_dir, 'model_{}.pt'.format(it)))
            writer.add_scalar('return', mean_return, it)
            writer.add_scalar('reward', mean_reward, it)
            writer.add_scalar('loss_q', mean_loss_q, it)
            writer.add_scalar('loss_p', mean_loss_p, it)
            writer.add_scalar('loss_l', mean_loss_l, it)
            writer.add_scalar('mean_q', mean_est_q, it)
            writer.add_scalar('η', self.η, it)
            writer.add_scalar('max_kl_μ', max_kl_μ, it)
            writer.add_scalar('max_kl_Σ', max_kl_Σ, it)

            writer.flush()

        # end training
        if writer is not None:
            writer.close()

    def load_model(self, path=None):
        """
        loads a model from a given path
        :param path: (str) file path (.pt file)
        """
        load_path = path if path is not None else self.save_path
        checkpoint = torch.load(load_path)
        self.iteration = checkpoint['iteration']
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic.train()
        self.target_critic.train()
        self.actor.train()
        self.target_actor.train()
        print('Loaded model: {}'.format(load_path))

    def save_model(self, path=None):
        """
        saves the model
        :param path: (str) file path (.pt file)
        """
        data = {
            'iteration': self.iteration,
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict()
        }
        torch.save(data, path)

    def sample_trajectory_worker(self, i):
        buff = []
        state = self.env.reset()
        env_info = self.env.reset(train_mode=True)[self.brain_name] 
        states = env_info.vector_observations

        for steps in range(self.sample_episode_maxlen):
            actions = self.target_actor.action(
                torch.from_numpy(states).type(torch.float32).to(self.device)
            ).cpu().numpy()
            
            env_info = self.env.step(actions)[self.brain_name]          # send all actions to tne environment
            next_states = env_info.vector_observations        # get next state (for each agent)
            rewards = env_info.rewards                        # get reward (for each agent)
            rewards = np.nan_to_num(rewards,nan=-5.0)         # fix NaN rewards of crawler environment, by penalizing a NaN reward
            dones = env_info.local_done                       # see if episode finished
            #penalize finishing early.. 
            if np.any(dones):
                rewards = rewards - 5.0
            # check if we have more than one observation, and accordingly store in buffer the rolled out experiences
            if len(rewards) > 1 :
                for state,action,next_state, reward in zip(states,actions,next_states, rewards):
                    buff.append((state, action, next_state, reward))
            else:
                buff.append((state,action,next_state,reward))
            if np.any(dones):
                break
            else:
                states = next_states
        return buff

    def __sample_trajectory(self, sample_episode_num):
        self.replaybuffer.clear()
        if self.multiprocessing:
            with Pool(self.sample_process_num) as p:
                episodes = p.map(self.sample_trajectory_worker, range(sample_episode_num))
        else:
            episodes = [self.sample_trajectory_worker(i)
                        for i in tqdm(range(sample_episode_num), desc='sample_trajectory')]
        self.replaybuffer.store_episodes(episodes)

    def __update_critic_td(self,
                           state_batch,
                           action_batch,
                           next_state_batch,
                           reward_batch,
                           sample_num=64):
        """
        :param state_batch: (B, ds)
        :param action_batch: (B, da) or (B,)
        :param next_state_batch: (B, ds)
        :param reward_batch: (B,)
        :param sample_num:
        :return:
        """
        B = state_batch.size(0)
        ds = self.ds
        da = self.da
        with torch.no_grad():
            r = reward_batch  # (B,)
            # get mean and covariance of action probability distributions for next state
            # ie next_action
            π_μ, π_A = self.target_actor.forward(next_state_batch)  # (B,)
            # create a normal (dimensions=action_size) distribution from actor output to sample from
            π = MultivariateNormal(π_μ, scale_tril=π_A)  # (B,)
            # sample a few actions from this π(next_state)
            sampled_next_actions = π.sample((sample_num,)).transpose(0, 1)  # (B, sample_num, da)
            # expand next_states batch to align with the actions samples. (B,ds)==>(B,sample_num,ds)
            expanded_next_states = next_state_batch[:, None, :].expand(-1, sample_num, -1)  # (B, sample_num, ds)
            # get q_target ... as the mean q for the actions that were sampled
            expected_next_q = self.target_critic.forward(
                expanded_next_states.reshape(-1, ds),  # (B * sample_num, ds)
                sampled_next_actions.reshape(-1, da)  # (B * sample_num, da)
            ).reshape(B, sample_num).mean(dim=1)  # (B,)
            # y = Q_target(s,a) = r(s,a) + gamma*Q_target(s+1,π(s+1))
            y = r + self.γ * expected_next_q
        self.critic_optimizer.zero_grad()
        # t = Q_theta(s,a)
        t = self.critic(
            state_batch,
            action_batch
        ).squeeze()
        loss = self.norm_loss_q(y, t)
        loss.backward()
        self.critic_optimizer.step()
        return loss, y

    def __update_param(self):
        """
        Sets target parameters to trained parameter
        """
        # Update policy parameters
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
    
        # Update critic parameters
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)


def main(argv):
    del argv
    logging.get_absl_handler().use_absl_log_file()
    # modify some parameters of training
    config.sample_episode_maxlen = 999
    config.sample_episode_num = 50
    config.batch_size = 128
    config.load = "./model/model_latest.pt"
    env = UnityEnvironment(file_name='../../deep-reinforcement-learning/p2_continuous-control/Crawler_Windows_x86_64/Crawler.exe')
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
    model.play(
        episodes= 100,
        frames = 1000)

    '''         
    model.train(
        iteration_num=args.iteration_num,
        log_dir=args.log_dir,
        render=args.render,
        debug=args.debug)
    '''
    env.close()


if __name__ == '__main__':
    app.run(main)
