from importlib import import_module

def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    # parser.add_argument('--alg', help='Algorithm', type=str, default='her')
    parser.add_argument('--network', help='network type mlp', default='mlp')
    parser.add_argument('--num_epoch', help='number of epochs to train', type=int, default=50)
    parser.add_argument('--num_env', help='Number of environment copies being run', default=1, type=int)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--load_path', help='Path to load trained model to', default=None, type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--save_buffer', help='If save the buffer or not', action='store_true')
    parser.add_argument('--load_buffer', help='If to load the offline buffer', action='store_true')
    parser.add_argument('--load_model', help='If to load the saved model', action='store_true')
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--play_no_training', default=False, action='store_true')
    parser.add_argument('--mode', help='mode of algorithms "dynamic", "supervised"', default=None, type=str)
    parser.add_argument('--su_method', help='method for supervised learning', default='', type=str)
    parser.add_argument('--offline_train', help='If training offline or not', default=False, action='store_true')
    return parser


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval
