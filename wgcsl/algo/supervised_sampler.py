from wgcsl.common import logger
import numpy as np
from wgcsl.algo.util import random_log
from wgcsl.algo.adv_que import advque

global global_threshold 
global_threshold = 0

def make_random_sample(reward_fun):
    def _random_sample(episode_batch, batch_size_in_transitions): 
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                            for key in episode_batch.keys()}
        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # # Re-compute reward since we may have substituted the u and o_2 ag_2
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                        for k in transitions.keys()}
        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions
    return _random_sample


def make_sample_transitions(replay_strategy, replay_k, reward_fun, no_relabel=False):
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  
        future_p = 0

    if no_relabel:
        print( '*' * 10 + 'Do not use relabeling in this method' + '*' * 10)
    
    def _preprocess(episode_batch, batch_size_in_transitions):
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}
        return transitions, episode_idxs, t_samples, batch_size, T

    def _get_reward(ag_2, g):
        info = {}
        reward_params = {'ag_2':ag_2, 'g':g}
        reward_params['info'] = info
        # make rewards -1/0 --->0/1
        return reward_fun(**reward_params) + 1  

    def _get_future_ags(episode_batch, episode_idxs, t_samples, batch_size, T, future_p=future_p, return_t=False):
        her_indexes = (np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T-t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        if not return_t:
            return future_ag.copy(), her_indexes.copy()
        else:
            return future_ag.copy(), her_indexes.copy(), future_offset
    
        
    def _reshape_transitions(transitions, batch_size, batch_size_in_transitions):
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions
    
    def _sample_her_transitions(episode_batch, batch_size_in_transitions, info=None):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        if not no_relabel:
            future_ag, her_indexes = _get_future_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            if len(transitions['g'].shape) == 1:
                transitions['g'][her_indexes] = future_ag.reshape(-1)
            else:
                transitions['g'][her_indexes] = future_ag

        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    def _sample_supervised_transitions(episode_batch, batch_size_in_transitions, info):
        train_policy, gamma, get_Q_pi, method = info['train_policy'], info['gamma'], info['get_Q_pi'], info['method']
        baw_delta, baw_max = info['baw_delta'], info['baw_max']
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)

        random_log('using supervide policy learning with method {} and no relabel {}'.format(method, no_relabel))
        original_g = transitions['g'].copy() # save to train the value function
        if not no_relabel:
            future_ag, her_indexes, offset = _get_future_ags(episode_batch, episode_idxs, t_samples, batch_size, T, future_p=1, return_t=True)
            transitions['g'][her_indexes] = future_ag
        else:
            offset = np.zeros(batch_size)

        if method == '': # do not use weights for GCSL
            loss = train_policy(o=transitions['o'], g=transitions['g'], u=transitions['u'])   
        else:
            method_lis = method.split('_')
            if 'gamma' in method_lis:
                weights = pow(gamma, offset)  
            else:
                weights = np.ones(batch_size)

            if 'adv' in method_lis:
                value = get_Q_pi(o=transitions['o'], g=transitions['g']).reshape(-1)
                next_value = get_Q_pi(o=transitions['o_2'], g=transitions['g']).reshape(-1)
                adv = _get_reward(transitions['ag_2'], transitions['g']) + gamma * next_value  - value
              
                if 'baw' in method_lis:
                    advque.update(adv)
                    global global_threshold
                    global_threshold = min(global_threshold + baw_delta, baw_max)
                    threshold = advque.get(global_threshold)

                if 'exp' in method_lis:  # exp weights
                    if 'clip10' in method_lis:
                        weights *= np.clip(np.exp(adv), 0, 10)
                    elif 'clip5' in method_lis:
                        weights *= np.clip(np.exp(adv), 0, 5)
                    elif 'clip1' in method_lis:
                        weights *= np.clip(np.exp(adv), 0, 1)
                    else:
                        weights *= np.exp(adv) 
                
                if 'baw' in method_lis:
                    positive = adv.copy()
                    positive[adv >= threshold] = 1
                    positive[adv < threshold] = 0.05
                    weights *= positive
                
            loss = train_policy(o=transitions['o'], g=transitions['g'], u=transitions['u'], weights=weights)  

        keep_origin_rate = 0.2
        origin_index = (np.random.uniform(size=batch_size) < keep_origin_rate)
        transitions['g'][origin_index] = original_g[origin_index]
        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g']) 
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    return _sample_supervised_transitions, _sample_her_transitions

