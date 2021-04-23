import os
import numpy as np
import gym

from mher.common import logger
from mher.algo.ddpg import DDPG
from mher.algo.her_sampler import make_sample_her_transitions, make_random_sample
from mher.algo.util import obs_to_goal_fun
from mher.common.monitor import Monitor
from mher.envs.multi_world_wrapper import PointGoalWrapper, SawyerGoalWrapper, ReacherGoalWrapper

DEFAULT_ENV_PARAMS = {
    'Point2DLargeEnv-v1':{
         'n_cycles':5,
         'n_batches': 1,
    },
    'Point2D-FourRoom-v1':{
        'n_cycles':5,
        'n_batches': 1,
    },
    'SawyerReachXYZEnv-v1':{
        'n_cycles':5,
        'n_batches': 1,
    },
    'FetchReach-v1': {
        'n_cycles': 5,  
        'n_batches': 4,
    },
    'Reacher-v2': {
        'n_cycles': 20,  
        'n_batches': 4,
        'num_epoch': 100
    }
}


DEFAULT_PARAMS = {  
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'mher.algo.actor_critic:ActorCritic',
    'Q_lr': 5e-3,  # critic learning rate
    'pi_lr': 5e-3,  # actor learning rate
    'buffer_size': int(2E4),  # for experience replay
    'polyak': 0.9,  #polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,
    # training
    'num_epoch':50, 
    'n_cycles': 5,  # per epoch
    'rollout_batch_size': 1,  # per mpi thread
    'n_batches': 5,  # training batches per cycle
    'batch_size': 128,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 100,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 1e-4,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values

    # random init episode
    'random_init':20, 

    # n step hindsight experience
    'n_step':3,

    # dynamic n-step
    'use_dynamic_nstep':False,  
    'alpha':3,
    'mb_relabeling_ratio': 0.8,
    'no_mb_relabel':False,
    'no_mgsl':False,
    'dynamic_batchsize':512,  # warm up the dynamic model
    'dynamic_init':100,
    
    # use supervised
    'use_supervised': False,

    # if do not use her
    'no_her':False    # used for DDPG 
}


CACHED_ENVS = {}


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]

def prepare_mode(kwargs):
    if 'mode' in kwargs.keys():
        mode = kwargs['mode']
        if mode == 'dynamic':
            kwargs['use_dynamic_nstep'] = True
            kwargs['use_supervised'] = False
        elif mode == 'supervised':
            kwargs['use_supervised'] = True
            kwargs['use_dynamic_nstep'] = False
        else:
            logger.log('No such mode!')
            raise NotImplementedError()
    else:
        kwargs['use_dynamic_nstep'] = False
        kwargs['use_supervised'] = False
        kwargs['n_step'] = 1

    return kwargs


def prepare_params(kwargs):
    # default max episode steps
    kwargs = prepare_mode(kwargs)
    default_max_episode_steps = 50
    # DDPG params
    ddpg_params = dict()
    env_name = kwargs['env_name']
    def make_env(subrank=None):
        try:
            env = gym.make(env_name, rewrad_type='sparse') 
        except:
            logger.log('Can not make sparse reward environment')
            env = gym.make(env_name)
        # add wrapper for multiworld environment
        if env_name.startswith('FetchReach'):
            env._max_episode_steps = 50
        elif env_name.startswith('Point2D'):
            env = PointGoalWrapper(env)
            env.env._max_episode_steps = 50
        elif env_name.startswith('Sawyer'): 
            env = SawyerGoalWrapper(env)
        elif env_name.startswith('Reacher'):
            env = ReacherGoalWrapper(env)

        if (subrank is not None and logger.get_dir() is not None):
            try:
                from mpi4py import MPI
                mpi_rank = MPI.COMM_WORLD.Get_rank()
            except ImportError:
                MPI = None
                mpi_rank = 0
                logger.warn('Running with a single MPI process. This should work, but the results may differ from the ones publshed in Plappert et al.')

            if hasattr(env, '_max_episode_steps'):
                max_episode_steps = env._max_episode_steps
            else:
                max_episode_steps = default_max_episode_steps # otherwise use defaulit max episode steps
            env =  Monitor(env,
                           os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
                           allow_early_resets=True)
            # hack to re-expose _max_episode_steps (ideally should replace reliance on it downstream)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    if hasattr(tmp_env, '_max_episode_steps'):
        kwargs['T'] = tmp_env._max_episode_steps
    else:
        kwargs['T'] = default_max_episode_steps

    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers','network_class','polyak','batch_size', 
                 'Q_lr', 'pi_lr', 'norm_eps', 'norm_clip', 'max_u','action_l2', 'clip_obs', 
                 'scope', 'relative_goals', 'n_step', 'use_dynamic_nstep', 
                 'alpha', 'dynamic_init', 'dynamic_batchsize', 'mb_relabeling_ratio',
                 'no_mb_relabel', 'no_mgsl','use_supervised']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    
    kwargs['ddpg_params'] = ddpg_params
    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    obs_to_goal = obs_to_goal_fun(env)

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
        'obs_to_goal_fun':obs_to_goal,
        'no_her': params['no_her']
    }
    for name in ['replay_strategy', 'replay_k']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]
        del params[name]

    sample_her, sample_nstep_dynamic_her, sample_nstep_supervised_her = make_sample_her_transitions(**her_params)
    random_sampler = make_random_sample(her_params['reward_fun'])
    samplers = {
        'her': sample_her,
        'random': random_sampler,
        'dynamic':sample_nstep_dynamic_her,
        'supervised':sample_nstep_supervised_her
    }
    return samplers, reward_fun

def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b

def configure_ddpg(dims, params, reuse=False, use_mpi=True, clip_return=True):
    samplers, reward_fun = configure_her(params)
    # Extract relevant parameters.
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params['ddpg_params']

    input_dims = dims.copy()
    # DDPG agent
    env = cached_make_env(params['make_env'])
    env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - params['gamma'])) if clip_return else np.inf,  # max abs of return 
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': samplers['her'],
                        'random_sampler':samplers['random'],
                        'nstep_dynamic_sampler':samplers['dynamic'],
                        'nstep_supervised_sampler':samplers['supervised'],
                        'gamma': params['gamma'],
                        'su_method': params['su_method']
                        })
    ddpg_params['info'] = {
        'env_name': params['env_name'],
        'reward_fun':reward_fun
    } 
    policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=use_mpi)  
    return policy


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())

    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
    }
    return dims
