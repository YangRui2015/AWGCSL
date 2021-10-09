"""
Helpers for scripts like run_atari.py.
"""

import os
import re
import sys
import tensorflow as tf
from wgcsl.envs.env_util import get_full_envname
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
import multiprocessing
from collections import defaultdict
from gym.wrappers import FlattenObservation
from wgcsl.common import logger
from wgcsl.common.monitor import Monitor
from wgcsl.common.util import set_global_seeds
from wgcsl.common.subproc_vec_env import SubprocVecEnv
from wgcsl.common.dummy_vec_env import DummyVecEnv
from wgcsl.common.wrappers import ClipActionsWrapper
from wgcsl.common.tf_util import get_session
from wgcsl.envs import register_envs
 
register_envs()

def get_game_envs():
    _game_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        # TODO: solve this with regexes
        try:
            env_type = env.entry_point.split(':')[0].split('.')[-1]
            _game_envs[env_type].add(env.id)
        except:
            pass
    return _game_envs

def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 env_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None,
                 initializer=None,
                 force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()
    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            env_kwargs=env_kwargs,
            logger_dir=logger_dir,
            initializer=initializer
        )

    set_global_seeds(seed)
    if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index, initializer=initializer) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(i + start_index, initializer=None) for i in range(num_env)])


def make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None, flatten_dict_observations=True, wrapper_kwargs=None, env_kwargs=None, logger_dir=None, initializer=None):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    if ':' in env_id:
        import re
        import importlib
        module_name = re.sub(':.*','',env_id)
        env_id = re.sub('.*:', '', env_id)
        importlib.import_module(module_name)
    env = gym.make(env_id, **env_kwargs)

    if env_id.startswith('Fetch'):
        from wgcsl.envs.multi_world_wrapper import FetchGoalWrapper
        env._max_episode_steps = 50
        env = FetchGoalWrapper(env)
    elif env_id.startswith('Hand'):
        env._max_episode_steps = 100
    elif env_id.startswith('Sawyer'):
        from wgcsl.envs.multi_world_wrapper import SawyerGoalWrapper
        env = SawyerGoalWrapper(env)
        if not hasattr(env, '_max_episode_steps'):
            env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
    elif env_id.startswith('Point'):
        from wgcsl.envs.multi_world_wrapper import PointGoalWrapper
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
        env = PointGoalWrapper(env)
    elif env_id.startswith('Reacher'):
        from wgcsl.envs.multi_world_wrapper import ReacherGoalWrapper
        env._max_episode_steps = 50
        env = ReacherGoalWrapper(env)
    else:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)
    return env




def get_env_type(args, _game_envs):
    env_id = get_full_envname(args.env)
    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        try:
            env_type = env.entry_point.split(':')[0].split('.')[-1]
            _game_envs[env_type].add(env.id)  # This is a set so add is idempotent
        except:
            pass

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def build_env(args, _game_envs):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    seed = args.seed

    env_type, env_id = get_env_type(args, _game_envs)

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)
    env = make_vec_env(env_id, env_type, args.num_env or 1, seed, flatten_dict_observations=False)
    return env
