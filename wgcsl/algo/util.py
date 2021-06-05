import os
import subprocess
import sys
import importlib
import inspect
import functools

import tensorflow as tf
import numpy as np

from awgcsl.common import tf_util as U


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def import_function(spec):
    """Import a function identified by a string like "pkg.module:fn_name".
    """
    mod_name, fn_name = spec.split(':')
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn


def flatten_grads(var_list, grads):
    """Flattens a variables and their gradients.
    """
    return tf.concat([tf.reshape(grad, [U.numel(v)])
                      for (v, grad) in zip(var_list, grads)], 0)


def nn(input, layers_sizes, reuse=None, flatten=False, name="", trainable='True', init='xavier', init_range=0.01):
    """Creates a simple neural network
    """
    if init == 'xavier':
        initializer = tf.contrib.layers.xavier_initializer()
    elif init == 'random':
        initializer = tf.random_uniform_initializer(minval=-init_range, maxval=init_range)
    else:
        raise NotImplementedError

    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=initializer,
                                reuse=reuse,
                                name=name + '_' + str(i),
                                trainable=trainable)
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


def install_mpi_excepthook():
    import sys
    from mpi4py import MPI
    old_hook = sys.excepthook

    def new_hook(a, b, c):
        old_hook(a, b, c)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort()
    sys.excepthook = new_hook


def mpi_fork(n, extra_mpi_args=[]):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        # "-bind-to core" is crucial for good performance
        args = ["mpirun", "-np", str(n)] + \
            extra_mpi_args + \
            [sys.executable]

        args += sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    else:
        install_mpi_excepthook()
        return "child"


def convert_episode_to_batch_major(episode):
    """Converts an episode to have the batch dimension in the major (first)
    dimension.
    """
    episode_batch = {}
    for key in episode.keys():
        val = np.array(episode[key]).copy()
        # make inputs batch-major instead of time-major
        episode_batch[key] = val.swapaxes(0, 1)

    return episode_batch


def transitions_in_episode_batch(episode_batch):
    """Number of transitions in a given episode batch.
    """
    shape = episode_batch['u'].shape
    return shape[0] * shape[1]


def reshape_for_broadcasting(source, target):
    """Reshapes a tensor (source) to have the correct shape and dtype of the target
    before broadcasting it with MPI.
    """
    dim = len(target.get_shape())
    shape = ([1] * (dim - 1)) + [-1]
    return tf.reshape(tf.cast(source, target.dtype), shape)

def obs_to_goal_fun(env):
    # only support Fetchenv and Handenv now
    from gym.envs.robotics import FetchEnv, hand_env
    from awgcsl.envs import point2d
    from awgcsl.envs import sawyer_reach
    from gym.envs.mujoco import reacher

    tmp_env = env
    while hasattr(tmp_env, 'env'):
        tmp_env = tmp_env.env

    if isinstance(tmp_env, FetchEnv):
        obs_dim = env.observation_space['observation'].shape[0]
        goal_dim = env.observation_space['desired_goal'].shape[0]
        temp_dim = env.sim.data.get_site_xpos('robot0:grip').shape[0]
        def obs_to_goal(observation):
            observation = observation.reshape(-1, obs_dim)
            if env.has_object:
                goal = observation[:, temp_dim:temp_dim + goal_dim]
            else:
                goal = observation[:, :goal_dim]
            return goal.copy()
    elif isinstance(tmp_env, hand_env.HandEnv):
        goal_dim = env.observation_space['desired_goal'].shape[0]
        def obs_to_goal(observation):
            goal = observation[:, -goal_dim:]
            return goal.copy()
    elif isinstance(tmp_env, point2d.Point2DEnv):
        def obs_to_goal(observation):
            return observation.copy()
    elif isinstance(tmp_env, sawyer_reach.SawyerReachXYZEnv):
        def obs_to_goal(observation):
            return observation
    elif isinstance(tmp_env, reacher.ReacherEnv):
        def obs_to_goal(observation):
            return observation[:, -3:-1]
    else:
        def obs_to_goal(observation):
            return observation
        # raise NotImplementedError('Do not support such type {}'.format(env))
        
    return obs_to_goal

def g_to_ag(o, env_id):
    if env_id == 'FetchReach':
        ag = o[:,0:3]
    elif env_id in ['FetchPush','FetchSlide', 'FetchPickAndPlace']:
        ag = o[:,3:6]
    else:
        raise NotImplementedError
    return ag


def dump_params(logger, params):
    import json
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f: # save params
        import copy
        dump_params = copy.deepcopy(params)
        for key, value in params.items():
            dump_params[key] = str(value)
        json.dump(dump_params, f)

def write_to_file(string, path='file.txt'):
    with open(path, 'a+') as file:
        file.writelines(string + '\n')
        file.flush()

def random_log(string):
    if np.random.random() < 0.02:
        print(string)

def discounted_return(rewards, gamma, reward_offset=True):
    L = len(rewards)
    if type(rewards[0]) == np.ndarray and len(rewards[0]):
        rewards = np.array(rewards).T
    else:
        rewards = np.array(rewards).reshape(1, L)

    if reward_offset:
        rewards += 1   # positive offset

    discount_weights = np.power(gamma, np.arange(L)).reshape(1, -1)
    dis_return = (rewards * discount_weights).sum(axis=1)
    undis_return = rewards.sum(axis=1)
    return dis_return.mean(), undis_return.mean()