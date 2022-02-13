import gym
from wgcsl.envs import register_envs
register_envs()
from rlkit.envs.wrappers_for_rlkit import ReacherGoalWrapper, SawyerGoalWrapper, \
    PointGoalWrapper, FlattenGoalWrapper, RewardWrapper
import pickle
import numpy as np
import os


def make_env(env_name):
    env = gym.make(env_name)
    if env_name.startswith('Fetch'):
        env._max_episode_steps = 50
    if env_name.startswith('Sawyer'):
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
        env = SawyerGoalWrapper(env)
    elif env_name.startswith('Point2D'):
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
        env = PointGoalWrapper(env)
    elif env_name.startswith('Reacher'):
        env._max_episode_steps = 50
        env = ReacherGoalWrapper(env)
    else:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
    return FlattenGoalWrapper(RewardWrapper(env))

'''
from type
{
	'o':[episode, episode_len, state_dim],
	'g':[episode, episode_len, goal_dim],
	'a':[episode, episode_len, action_dim],
	'ag':[episode, episode_len, goal_dim]
} to

state.npy: [transition, state_dim]
...
'''
def convert_buffer(path):
    print('converting path: ' + path)
    env_name = path.split('/')[-2]
    env = make_env(env_name)
    with open(path, 'rb') as f:
        buffer = pickle.load(f)
        N = buffer['u'].shape[0]
        L = buffer['u'].shape[1]
        size = N * L
        state_dim, goal_dim, action_dim = buffer['o'].shape[-1], buffer['g'].shape[-1], buffer['u'].shape[-1]
        states = np.zeros((size, state_dim))
        next_states = np.zeros((size, state_dim))
        achieved_goals = np.zeros((size, goal_dim))
        next_achieved_goals = np.zeros((size, goal_dim))

        for i in range(N):
            states[i * L: (i+1) * L] = buffer['o'][i][:-1]
            next_states[i * L:(i+1) * L] = buffer['o'][i][1:]
            achieved_goals[i * L:(i+1) * L] = buffer['ag'][i][:-1]
            next_achieved_goals[i * L:(i+1) * L] = buffer['ag'][i][1:]

        goals = buffer['g'].reshape(-1, goal_dim)
        actions = buffer['u'].reshape(-1, action_dim)
        state_goals = np.concatenate((states, goals, achieved_goals), axis=1)
        next_state_goals = np.concatenate((next_states, goals, next_achieved_goals), axis=1)
        rewards, dones = np.zeros((size, 1)), np.zeros((size, 1))
        for i in range(N):
            dones[(i+1) * L - 1] = 1
        for i in range(size):
            rewards[i] = env.env.compute_reward(achieved_goals[i], goals[i], None)
        if rewards.min() == -1:
            rewards += 1

    save_path = path.rstrip('buffer.pkl')
    np.save(os.path.join(save_path, 'state.npy'), state_goals)
    np.save(os.path.join(save_path, 'next_state.npy'), next_state_goals)
    np.save(os.path.join(save_path, 'action.npy'), actions)
    np.save(os.path.join(save_path, 'reward.npy'), rewards)
    np.save(os.path.join(save_path, 'not_done.npy'), 1 - dones)
    np.save(os.path.join(save_path, 'done.npy'), dones)
    import pdb;pdb.set_trace()
    
	


if __name__ == '__main__':
    import glob2
    directory = './random/'
    filename = 'buffer.pkl'
    files = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(directory, '**', filename))]
    for file in files:
        convert_buffer(os.path.join(file, filename))