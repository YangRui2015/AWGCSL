from tensorflow.python.ops.gen_batch_ops import batch
from awgcsl.common import logger
import numpy as np
from awgcsl.algo.util import obs_to_goal_fun, random_log

def dynamic_interaction(o, g, action_fun, dynamic_model, steps):
    last_state = o.copy()
    next_states_list = []
    for _ in range(0, steps):
        action_array = action_fun(o=last_state, g=g)
        next_state_array = dynamic_model.predict_next_state(last_state, action_array) 
        next_states_list.append(next_state_array.copy())
        last_state = next_state_array
    return next_states_list

def make_random_sample(reward_fun):
    def _random_sample(episode_batch, batch_size_in_transitions): 
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout
        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                            for key in episode_batch.keys()}

        # Reconstruct info dictionary for reward  computation.
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


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, obs_to_goal_fun=None, no_her=False):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  
        future_p = 0

    if no_her:
        print( '*' * 10 + 'Will not use HER in this method' + '*' * 10)
    
    def _preprocess(episode_batch, batch_size_in_transitions, ags_std=None, use_ag_std=False):
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout

        # np.random.randint doesn't contain the last one, so comes from 0 to roolout_batch_size-1
        if use_ag_std:
            episode_idxs = np.random.choice(np.arange(rollout_batch_size), batch_size, p=ags_std[:rollout_batch_size]/ags_std[:rollout_batch_size].sum())
        else:
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}
        return transitions, episode_idxs, t_samples, batch_size, T

    def _get_reward(ag_2, g):
        info = {}
        reward_params = {'ag_2':ag_2, 'g':g}
        reward_params['info'] = info
        return reward_fun(**reward_params) + 1  # make rewards positive
    
    def _get_return(episode_batch, episode_idxs, t_samples, goal_idxs, batch_size, gamma):
        new_return = np.zeros(batch_size)
        for i in range(batch_size):
            ags = episode_batch['ag_2'][episode_idxs[i], t_samples[i]:]
            g = episode_batch['ag'][episode_idxs[i], goal_idxs[i]]
            gs = np.array([g for _ in range(len(ags))])
            new_return[i] = (_get_reward(ags, gs) * np.power(gamma, np.arange(len(ags)))).sum()
        return new_return

    def _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T, future_p=future_p, return_t=False):
        her_indexes = (np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T-t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        if not return_t:
            return future_ag.copy(), her_indexes.copy()
        else:
            return future_ag.copy(), her_indexes.copy(), future_offset
    
    def _get_ags_from_states(batch_size, states, ratio=0.3, indexs=None):
        if indexs is None:
            indexs = (np.random.uniform(size=batch_size) < ratio)
        next_goals = obs_to_goal_fun(states[indexs])
        return next_goals.copy(), indexs.copy()
        
    def _reshape_transitions(transitions, batch_size, batch_size_in_transitions):
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions

    def _sample_her_transitions(episode_batch, batch_size_in_transitions, info=None):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        if not no_her:
            future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            transitions['g'][her_indexes] = future_ag

        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)


    def _sample_nstep_dynamic_her_transitions(episode_batch, batch_size_in_transitions, info):
        steps, gamma, Q_fun, alpha = info['nstep'], info['gamma'], info['get_Q_pi'], info['alpha']
        dynamic_model, action_fun = info['dynamic_model'], info['action_fun']
        dynamic_ag_ratio = info['mb_relabeling_ratio'] 
        transitions, _, _, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        train_policy, no_mb_relabel, no_mgsl = info['train_policy'], info['no_mb_relabel'], info['no_mgsl']
        dynamic_ag_ratio_cur = dynamic_ag_ratio

        random_log('using nstep dynamic sampler with step:{}, alpha:{}, and dynamic relabeling rate:{}'.format(steps, alpha, dynamic_ag_ratio_cur))
        # update dynamic model
        loss = dynamic_model.update(transitions['o'], transitions['u'], transitions['o_2'], times=2)  
        if np.random.random() < 0.01:
            print(loss)

        relabel_indexes = (np.random.uniform(size=batch_size) < dynamic_ag_ratio_cur)
        # # Re-compute reward since we may have substituted the goal.
        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])

        # model-based augmentations
        last_state = transitions['o_2'].copy()  
        if dynamic_ag_ratio_cur > 0:
            next_states_list = dynamic_interaction(last_state, transitions['g'], action_fun, dynamic_model, steps)
            next_states_list.insert(0, last_state.copy())
            next_states_array = np.concatenate(next_states_list,axis=1).reshape(batch_size, steps+1, -1) 
            step_idx = np.random.randint(next_states_array.shape[1], size=(batch_size))
            last_state = next_states_array[np.arange(batch_size).reshape(-1), step_idx]
            # add dynamic achieve goals
            new_ags, _= _get_ags_from_states(batch_size, last_state, 1)
            
            if not no_mb_relabel:
                transitions['g'][relabel_indexes] = new_ags[relabel_indexes]  

            transitions['idxs'] = relabel_indexes.copy()
            if not no_mgsl and no_mb_relabel:
                # Auxilary task for no MBR (set alpha=0)
                train_policy(o=transitions['o'][relabel_indexes], g=new_ags[relabel_indexes], u=transitions['u'][relabel_indexes])  

            # recompute rewards
            transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])

        target_step1 = transitions['r'] + gamma * Q_fun(o=transitions['o_2'], g=transitions['g']).reshape(-1)
        transitions['r'] = target_step1.copy()
        
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    def _sample_nstep_supervised_her_transitions(episode_batch, batch_size_in_transitions, info):
        train_policy, gamma, get_Q_pi, method, get_ags_std  = info['train_policy'], info['gamma'], info['get_Q_pi'], info['method'], info['get_ags_std']
        use_adv_norm, adv_norm = info['use_adv_norm'], info['adv_norm']
        ags_std = get_ags_std()
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions, ags_std, use_ag_std=False)

        random_log('using nstep supervide policy learning with method {}'.format(method))
        future_ag, her_indexes, offset = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T, future_p=1, return_t=True)
        original_g = transitions['g'].copy() # save to train the value function
        transitions['g'][her_indexes] = future_ag

        if method == '':
            loss = train_policy(o=transitions['o'], g=transitions['g'], u=transitions['u'])   # do not use weights
        else:
            method_lis = method.split('_')
            if 'gamma' in method_lis:
                weights = pow(gamma, offset)  
            else:
                weights = np.ones(batch_size)

            if 'adv' in method_lis:
                value = get_Q_pi(o=transitions['o'], g=transitions['g']).reshape(-1)
                next_value = get_Q_pi(o=transitions['o_2'], g=transitions['g']).reshape(-1)
                adv = _get_reward(transitions['ag_2'], transitions['g']) + gamma * next_value - value
                # print(adv.min(), adv.max(), adv.mean())
                # if use_adv_norm:
                #     # print(adv_norm.average_absolute)
                #     print(adv_norm.max_absolute)
                #     adv_norm.update(adv)
                #     adv = adv_norm.normalize(adv)

                if 'exp' in method_lis:
                    if 'clip10' in method_lis:
                        weights *= np.clip(np.exp(adv), 0, 10)
                    elif 'clip5' in method_lis:
                        weights *= np.clip(np.exp(adv), 0, 5)
                    elif 'clip1' in method_lis:
                        weights *= np.clip(np.exp(adv), 0, 1)
                    else:
                        weights *= np.exp(adv) # exp weights
                    
                elif 'tanh' in method_lis:
                    # weights *= (np.tanh(adv) * 0.5 + 0.5)
                    weights *= np.tanh(adv) + 1
                elif '01' in method_lis:
                    adv[adv < 0] = 0
                    adv[adv >= 0] = 1
                    weights *= adv
                else:
                    weights *= adv

            loss = train_policy(o=transitions['o'], g=transitions['g'], u=transitions['u'], weights=weights)  

        # train value function
        keep_origin_rate = 0.2
        origin_index = (np.random.uniform(size=batch_size) < keep_origin_rate)
        transitions['g'][origin_index] = original_g[origin_index]
        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g']) 

        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    return _sample_her_transitions, _sample_nstep_dynamic_her_transitions, _sample_nstep_supervised_her_transitions

