import sys
import re
import os
import os.path as osp
import gym
import tensorflow as tf
import numpy as np

from awgcsl.common.vec_env import VecEnv
from awgcsl.common.env_util import get_env_type, build_env, get_game_envs
from awgcsl.common.parse_args import common_arg_parser, parse_unknown_args
from awgcsl.common import logger
from awgcsl.common.parse_args import get_learn_function_defaults, parse_cmdline_kwargs, parse_unknown_args
from awgcsl.algo.her import learn
from awgcsl.util import init_logger


_game_envs = get_game_envs()


def train(args, extra_args):
    env_type, env_id = get_env_type(args, _game_envs)
    print('env_type: {}'.format(env_type))
    seed = args.seed
    alg_kwargs = get_learn_function_defaults('her', env_type)
    alg_kwargs.update(extra_args)
    env = build_env(args, _game_envs)
    print('Training {} on {}:{} with arguments \n{}'.format(args.mode, env_type, env_id, alg_kwargs))

    ## make save dir
    if args.save_path:
        os.makedirs(os.path.expanduser(args.save_path), exist_ok=True)

    model = learn(
        env=env,
        seed=seed,
        num_epoch=args.num_epoch,
        save_path=args.save_path,
        load_buffer=args.load_buffer,
        load_path=args.load_path,
        play_no_training=args.play_no_training,
        offline_train=args.offline_train,
        mode=args.mode,
        su_method=args.su_method,
        **alg_kwargs
    )
    return model, env


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    rank = init_logger(args)

    model, env = train(args, extra_args)
    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        last_policy_path = os.path.join(save_path, 'policy_last.pkl')
        model.save(last_policy_path)
        if args.save_buffer:
            buffer_path = os.path.join(save_path, 'buffer.pkl')
            model.buffer.save(buffer_path)

    if args.play: #  or args.play_no_training
        logger.log("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0

    env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)
