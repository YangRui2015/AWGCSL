from awgcsl.envs import *
from awgcsl.envs.multi_world_wrapper import PointGoalWrapper, SawyerGoalWrapper, ReacherGoalWrapper
import awgcsl.envs
import awgcsl
import gym
from gym import wrappers
from gym.envs.registration import register
from argparse import ArgumentParser
import torch.nn.functional as F
import random
from copy import deepcopy
from collections import deque
from IPython.display import clear_output
import gym
import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
from collections import namedtuple
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from os.path import join as joindir
from os import makedirs as mkdir
import pandas as pd
import numpy as np
import argparse
import datetime
import math
from IPython import embed

def parse_args():
    parser = ArgumentParser(description='train args')
    parser.add_argument('-en','--env_name', type=str, default=None)
    parser.add_argument('-r','--repeat',type=int,default=None)
    parser.add_argument('-g','--gpu',type=int,default=None)
    return parser.parse_args()

argss = parse_args()
torch.cuda.set_device(argss.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def register_envs():
    register(
        id='SawyerReachXYZEnv-v1',
        entry_point='awgcsl.envs.sawyer_reach:SawyerReachXYZEnv',
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'murtaza'
        },
        kwargs={
            'hide_goal_markers': True,
            'norm_order': 2,
        },
    )
    register(
        id='Point2DLargeEnv-v1',
        entry_point='awgcsl.envs.point2d:Point2DEnv',
        tags={
            'git-commit-hash': '4efe2be',
            'author': 'Vitchyr'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 0.5,
            'boundary_dist':5,
            'render_onscreen': False,
            'show_goal': True,
            'render_size':512,
            'get_image_base_render_size': (48, 48),
            'bg_color': 'white',
        },
    )
    register(
        id='Point2D-FourRoom-v1',
        entry_point='awgcsl.envs.point2d:Point2DWallEnv',
        kwargs={
            'action_scale': 1,
            'wall_shape': 'four-room-v1', 
            'wall_thickness': 0.30,
            'target_radius':1,
            'ball_radius':0.5,
            'boundary_dist':5,
            'render_size': 512,
            'wall_color': 'darkgray',
            'bg_color': 'white',
            'images_are_rgb': True,
            'render_onscreen': False,
            'show_goal': True,
            'get_image_base_render_size': (48, 48),
        },
    )
    # register gcsl envs
    register(
        id='SawyerDoor-v0',
        entry_point='awgcsl.envs.sawyer_door:SawyerDoorGoalEnv',
    )
register_envs()
"""
ref: Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
ref: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
ref: https://github.com/openai/baselines/tree/master/baselines/ppo2
"""


for env_id in [argss.env_name]:
    env = gym.make(env_id)

    if env_id.startswith('Fetch'):
        env._max_episode_steps = 50
    elif env_id.startswith('Sawyer'):
        from awgcsl.envs.multi_world_wrapper import SawyerGoalWrapper
        env = SawyerGoalWrapper(env)
        if not hasattr(env, '_max_episode_steps'):
            env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
    elif env_id.startswith('Point2D'):
        from awgcsl.envs.multi_world_wrapper import PointGoalWrapper
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
        env = PointGoalWrapper(env)
    elif env_id.startswith('Reacher'):
        from awgcsl.envs.multi_world_wrapper import ReacherGoalWrapper
        env._max_episode_steps = 50
        env = ReacherGoalWrapper(env)
    else:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)

    class RewardWrapper(gym.RewardWrapper):
        def __init__(self, env):
            super().__init__(env)

        def reward(self, rew):
            if rew != 0 and rew!= -1:
                print(rew)
            assert rew == 0 or rew == -1, 'input reward should be -1/0'
            return rew + 1

    env = RewardWrapper(env)
    

    LR_PPO = 5e-4
    LR_HID = 0.0
    FUTURE_P = 0.0

    for repeat in range(argss.repeat, argss.repeat+2):
        class args(object):
            seed = 1234 + repeat
            num_episode = 400 if ('SawyerDoor' in env_id or 'Reacher' in env_id) else 50
            batch_size = 250
            max_step_per_round = 50
            gamma = 0.98
            lamda = 0.97
            log_num_episode = 1
            num_epoch = 10
            minibatch_size = 25
            clip = 0.2
            loss_coeff_value = 0.5
            loss_coeff_entropy = 0.01

            lr_ppo = LR_PPO
            lr_hid = LR_HID
            future_p = FUTURE_P # param of HER
            num_parallel_run = 1
            # tricks
            schedule_adam = 'linear'
            schedule_clip = 'linear'
            layer_norm = True
            state_norm = False
            advantage_norm = True
            lossvalue_norm = True
            replay_buffer_size_IER = 50000 

        rwds = []
        Succ_recorder = []
        global losses
        Horizon_list = [1]
        losses = [[]]

        Transition = namedtuple('Transition', ('state', 'value', 'action', 'logproba', 'mask', 'next_state', 'reward'))
        EPS = 1e-6
        RESULT_DIR = 'Result_PPO'
        mkdir(RESULT_DIR, exist_ok=True)

        
        class RunningStat(object):
            def __init__(self, shape):
                self._n = 0
                self._M = np.zeros(shape)
                self._S = np.zeros(shape)

            def push(self, x):
                x = np.asarray(x)
                assert x.shape == self._M.shape
                self._n += 1
                if self._n == 1:
                    self._M[...] = x
                else:
                    oldM = self._M.copy()
                    self._M[...] = oldM + (x - oldM) / self._n
                    self._S[...] = self._S + (x - oldM) * (x - self._M)

            @property
            def n(self):
                return self._n

            @property
            def mean(self):
                return self._M

            @property
            def var(self):
                return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

            @property
            def std(self):
                return np.sqrt(self.var)

            @property
            def shape(self):
                return self._M.shape


        class ZFilter:
            """
            y = (x-mean)/std
            using running estimates of mean,std
            """

            def __init__(self, shape, demean=True, destd=True, clip=10.0):
                self.demean = demean
                self.destd = destd
                self.clip = clip

                self.rs = RunningStat(shape)

            def __call__(self, x, update=True):
                if update: self.rs.push(x)
                if self.demean:
                    x = x - self.rs.mean
                if self.destd:
                    x = x / (self.rs.std + 1e-8)
                if self.clip:
                    x = np.clip(x, -self.clip, self.clip)
                return x

            def output_shape(self, input_space):
                return input_space.shape


        class ActorCritic(nn.Module):
            def __init__(self, num_inputs, num_outputs, layer_norm=True):
                super(ActorCritic, self).__init__()

                self.actor_fc1 = nn.Linear(num_inputs, 256)
                self.actor_fc2 = nn.Linear(256, 256)
                self.actor_fc3 = nn.Linear(256, num_outputs)
                self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

                self.critic_fc1 = nn.Linear(num_inputs, 256)
                self.critic_fc2 = nn.Linear(256, 256)
                self.critic_fc3 = nn.Linear(256, 1)

                if layer_norm:
                    self.layer_norm(self.actor_fc1, std=1.0)
                    self.layer_norm(self.actor_fc2, std=1.0)
                    self.layer_norm(self.actor_fc3, std=0.01)

                    self.layer_norm(self.critic_fc1, std=1.0)
                    self.layer_norm(self.critic_fc2, std=1.0)
                    self.layer_norm(self.critic_fc3, std=1.0)

            @staticmethod
            def layer_norm(layer, std=1.0, bias_const=0.0):
                torch.nn.init.orthogonal_(layer.weight, std)
                torch.nn.init.constant_(layer.bias, bias_const)

            def forward(self, states):
                action_mean, action_logstd = self._forward_actor(states)
                critic_value = self._forward_critic(states)
                return action_mean, action_logstd, critic_value

            def _forward_actor(self, states):
                x = torch.tanh(self.actor_fc1(states))
                x = torch.tanh(self.actor_fc2(x))
                action_mean = self.actor_fc3(x)
                action_logstd = self.actor_logstd.expand_as(action_mean)
                return action_mean, action_logstd

            def _forward_critic(self, states):
                x = torch.tanh(self.critic_fc1(states))
                x = torch.tanh(self.critic_fc2(x))
                critic_value = self.critic_fc3(x)
                return critic_value

            def select_action(self, action_mean, action_logstd, return_logproba=True):
                action_std = torch.exp(action_logstd)
                action = torch.normal(action_mean, action_std)
                if return_logproba:
                    logproba = self._normal_logproba(action, action_mean, action_logstd, action_std)
                return action, logproba

            @staticmethod
            def _normal_logproba(x, mean, logstd, std=None):
                if std is None:
                    std = torch.exp(logstd)

                std_sq = std.pow(2)
                logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
                return logproba.sum(1)

            def get_logproba(self, states, actions):
                action_mean, action_logstd = self._forward_actor(states)
                logproba = self._normal_logproba(actions, action_mean, action_logstd)
                return logproba


        class Memory(object):
            def __init__(self):
                self.memory = []

            def push(self, *args):
                self.memory.append(Transition(*args))

            def sample(self):
                return Transition(*zip(*self.memory))

            def __len__(self):
                return len(self.memory)
        class ReplayBuffer_imitation(object):
            def __init__(self, capacity):
                self.buffer = {'1step':deque(maxlen=capacity)}
                self.capacity = capacity
            def push(self, state, action, step_num):
                try:
                    self.buffer[step_num]
                except:
                    self.buffer[step_num] = deque(maxlen=self.capacity)
                self.buffer[step_num].append((state, action))


            def sample(self, batch_size,step_num):
                state, action= zip(*random.sample(self.buffer[step_num], batch_size))
                return np.stack(state), action

            def lenth(self,step_num):
                try:
                    self.buffer[step_num]
                except:
                    return 0
                return len(self.buffer[step_num])

            def __len__(self,step_num):
                try:
                    self.buffer[step_num]
                except:
                    return 0
                return len(self.buffer[step_num])


        num_inputs = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0] + env.observation_space.spaces['achieved_goal'].shape[0]# extended state
        num_actions = env.action_space.shape[0]
        network = ActorCritic(num_inputs, num_actions, layer_norm=args.layer_norm).to(device)
        running_state = ZFilter((num_inputs,), clip=5.0)
        Horizon_list = [1]
        def eval_policy(net = network, num_eval_epi=100, eval_std = True):
            net.eval()
            with torch.no_grad():
                eval_return_list = []
                eval_reward_list = []
                eval_success_list = []
                eval_Succ_num = 0
                eval_env = gym.make(env_id)

                if env_id.startswith('Fetch'):
                    eval_env._max_episode_steps = 50
                elif env_id.startswith('Sawyer'):
                    from awgcsl.envs.multi_world_wrapper import SawyerGoalWrapper
                    eval_env = SawyerGoalWrapper(eval_env)
                    if not hasattr(eval_env, '_max_episode_steps'):
                        eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=50)
                elif env_id.startswith('Point2D'):
                    from awgcsl.envs.multi_world_wrapper import PointGoalWrapper
                    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=50)
                    eval_env = PointGoalWrapper(eval_env)
                elif env_id.startswith('Reacher'):
                    from awgcsl.envs.multi_world_wrapper import ReacherGoalWrapper
                    eval_env._max_episode_steps = 50
                    eval_env = ReacherGoalWrapper(eval_env)
                else:
                    eval_env = gym.wrappers.TimeLimit(eval_env, max_episode_steps=50)

                class RewardWrapper(gym.RewardWrapper):
                    def __init__(self, eval_env):
                        super().__init__(eval_env)

                    def reward(self, rew):
                        if rew != 0 and rew!= -1:
                            print(rew)
                        assert rew == 0 or rew == -1, 'input reward should be -1/0'
                        return rew + 1

                eval_env = RewardWrapper(eval_env)
                for _ in range(num_eval_epi):
                    eval_state = eval_env.reset()
                    eval_state = np.concatenate((eval_state['observation'],eval_state['desired_goal'],eval_state['achieved_goal'])) # state_extended

                    if args.state_norm:
                        state = running_state(state)
                    eval_reward_sum = 0
                    eval_episode = []
                    eval_return_sum = 0
                    eval_Succ_in_env = 0
                    for t in range(50):
                        eval_action_mean, eval_action_logstd, eval_value = network(Tensor(eval_state).unsqueeze(0).to(device))
                        eval_action, eval_logproba = network.select_action(eval_action_mean, eval_action_logstd)
                        eval_action = eval_action.data.cpu().detach().numpy()[0]
                        #logproba = logproba.data.numpy()[0]
                        #print('action:',action)
                        if np.sum(np.isnan(eval_action))>0:
                            embed()
                        eval_next_state, eval_reward, eval_done, eval__ = env.step(eval_action)
                        if _['is_success'] !=0:
                            eval_Succ_in_env = 1
                            eval_Succ_num+=1
                        eval_next_state = np.concatenate((eval_next_state['observation'],eval_next_state['desired_goal'],eval_next_state['achieved_goal']))

                        eval_reward_sum += eval_reward
                        eval_return_sum += eval_reward * (args.gamma**(t))
                        if args.state_norm:
                            next_state = running_state(next_state)
                        eval_mask = 0 if done else 1

                        eval_state = eval_next_state
                    eval_success_list.append(eval_Succ_in_env)
                    eval_reward_list.append(eval_reward_sum)
                    eval_return_list.append(eval_return_sum)
                    eval_Winrate = 1.0*eval_Succ_num/num_eval_epi
                return np.mean(eval_reward_list), np.mean(eval_return_list), eval_Winrate, np.mean(eval_success_list)
        def ppo(args):
            num_inputs = env.observation_space.spaces['observation'].shape[0] + env.observation_space.spaces['desired_goal'].shape[0] + env.observation_space.spaces['achieved_goal'].shape[0]# extended state
            num_actions = env.action_space.shape[0]

            env.seed(args.seed)
            torch.manual_seed(args.seed)

            optimizer = opt.Adam(network.parameters(), lr=args.lr_ppo)


            


            reward_record = []
            eval_result = []
            global_steps = 0

            lr_now = args.lr_ppo
            clip_now = args.clip
            ier_buffer = ReplayBuffer_imitation(args.replay_buffer_size_IER)
            for i_episode in range(args.num_episode):
                memory = Memory()
                num_steps = 0
                reward_list = []
                return_list = []
                len_list = []
                Succ_num = 0

                game_num = 0
                succ_game = 0
                while num_steps < args.batch_size:
                    state = env.reset()
                    game_num +=1
                    state = np.concatenate((state['observation'],state['desired_goal'],state['achieved_goal'])) # state_extended

                    if args.state_norm:
                        state = running_state(state)
                    reward_sum = 0
                    episode = []
                    return_sum = 0
                    env_list = []
                    Succ_in_env = 0
                    for t in range(args.max_step_per_round):
                        action_mean, action_logstd, value = network(Tensor(state).unsqueeze(0).to(device))
                        action, logproba = network.select_action(action_mean, action_logstd)

                        action = action.cpu().detach().data.numpy()[0]
                        logproba = logproba.cpu().detach().data.numpy()[0]
                        #print('action:',action)
                        if np.sum(np.isnan(action))>0:
                            embed()
                        next_state, reward, done, _ = env.step(action)
                        if _['is_success'] !=0:
                            Succ_in_env = 1
                            Succ_num+=1
                        next_state = np.concatenate((next_state['observation'],next_state['desired_goal'],next_state['achieved_goal']))

                        reward_sum += reward
                        return_sum += reward * (args.gamma**(t//50))
                        if args.state_norm:
                            next_state = running_state(next_state)
                        mask = 0 if done else 1
                        episode.append((state, value, action, logproba, mask, next_state, reward))
                        memory.push(state, value, action, logproba, mask, next_state, reward)
                        #if done:
                        #    break

                        state = next_state
                    succ_game += Succ_in_env

                    #for ind,(state, value, action, logproba, mask, next_state, reward) in enumerate(episode):
                        



                    num_steps += (t + 1)
                    global_steps += (t + 1)
                    reward_list.append(reward_sum)
                    return_list.append(return_sum)
                    len_list.append(t + 1)
                    Winrate = 1.0*succ_game/game_num
                    Succ_recorder.append(Winrate)

                reward_record.append({
                    'episode': i_episode, 
                    'steps': global_steps, 
                    'meanepreward': np.mean(reward_list), 
                    'meanepreturn': np.mean(return_list),
                    'meaneplen': np.mean(len_list)})

                rwds.extend(reward_list)
                batch = memory.sample()
                batch_size = len(memory)

                SR = 1.0*Succ_num/num_steps

                # step2: extract variables from trajectories
                rewards = Tensor(batch.reward)
                values = Tensor(batch.value)
                masks = Tensor(batch.mask)
                actions = Tensor(batch.action)
                states = Tensor(batch.state)
                oldlogproba = Tensor(batch.logproba)

                returns = Tensor(batch_size)
                deltas = Tensor(batch_size)
                advantages = Tensor(batch_size)

                prev_return = 0
                prev_value = 0
                prev_advantage = 0
                for i in reversed(range(batch_size)):
                    returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
                    deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values[i]
                    # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
                    advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage * masks[i]

                    prev_return = returns[i]
                    prev_value = values[i]
                    prev_advantage = advantages[i]
                if args.advantage_norm:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

                for i_epoch in range(int(args.num_epoch * batch_size / args.minibatch_size)):
                    # sample from current batch
                    minibatch_ind = np.random.choice(batch_size, args.minibatch_size, replace=False)
                    minibatch_states = states[minibatch_ind].to(device)
                    minibatch_actions = actions[minibatch_ind].to(device)
                    minibatch_oldlogproba = oldlogproba[minibatch_ind].to(device)
                    minibatch_newlogproba = network.get_logproba(minibatch_states.to(device), minibatch_actions.to(device))
                    minibatch_advantages = advantages[minibatch_ind].to(device)
                    minibatch_returns = returns[minibatch_ind].to(device)
                    minibatch_newvalues = network._forward_critic(minibatch_states.to(device)).flatten()

                    ratio =  torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
                    surr1 = ratio * minibatch_advantages
                    surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
                    loss_surr = - torch.mean(torch.min(surr1, surr2))

                    if args.lossvalue_norm:
                        minibatch_return_6std = 6 * minibatch_returns.std() + EPS
                        loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
                    else:
                        loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

                    loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

                    total_loss = loss_surr + args.loss_coeff_value * loss_value + args.loss_coeff_entropy * loss_entropy
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()


                if args.schedule_clip == 'linear':
                    ep_ratio = 1 - (i_episode / args.num_episode)
                    clip_now = args.clip * ep_ratio

                if args.schedule_adam == 'linear':
                    ep_ratio = 1 - (i_episode / args.num_episode)
                    lr_now = args.lr_ppo * ep_ratio
                    for g in optimizer.param_groups:
                        g['lr'] = lr_now

                if i_episode % args.log_num_episode == 0:
                    eval_result.append(eval_policy(network, 100))
                    print('evaluation result:', eval_result[-1])
                    np.save('Result_PPO/evaluation_result_env{}_repeat{}.npy'.format(env_id, repeat), eval_result)
                    print('Finished episode: {} Reward: {:.4f} Return {:.4f} Stay Rate{:.4f} SuccessRate{:.4f}' \
                        .format(i_episode, reward_record[-1]['meanepreward'], reward_record[-1]['meanepreturn'],SR,Winrate))
                    print('-----------------')

            return reward_record

        def test(args):
            record_dfs = []
            for i in range(args.num_parallel_run):
                args.seed += 1
                reward_record = pd.DataFrame(ppo(args))
                reward_record['#parallel_run'] = i
                record_dfs.append(reward_record)
            record_dfs = pd.concat(record_dfs, axis=0)
            record_dfs.to_csv(joindir(RESULT_DIR, 'ppo-record-env{}_repeat{}.csv'.format(env_id, repeat)))


        test(args)
        rwds_HER_HID= deepcopy(rwds)
        Succ_recorder_HER_HID= deepcopy(Succ_recorder)
        np.save('Result_PPO/env{}_repeat{}'.format(env_id,repeat),(rwds_HER_HID,Succ_recorder_HER_HID))