import mher.common.tf_util as U
import tensorflow as tf
import numpy as np
from mher.common import logger
from mher.common.mpi_adam import MpiAdam
from mher.algo.util import store_args
from mher.algo.normalizer import NormalizerNumpy


def nn(input, layers_sizes, reuse=None, flatten=False, use_layer_norm=False, name=""):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        norm = tf.contrib.layers.layer_norm if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),

                                reuse=reuse,
                                name=name + '_' + str(i))
        if use_layer_norm and norm:
            input = norm(input, reuse=reuse, scope=name + '_layer_norm_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


def _vars(scope):
    res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    assert len(res) > 0
    return res



# numpy forward dynamics
class ForwardDynamicsNumpy:
    @store_args
    def __init__(self, dimo, dimu, clip_norm=5, norm_eps=1e-4, hidden=256, layers=8, learning_rate=1e-3):
        self.obs_normalizer = NormalizerNumpy(size=dimo, eps=norm_eps)
        self.action_normalizer = NormalizerNumpy(size=dimu, eps=norm_eps)
        self.sess = U.get_session()

        with tf.variable_scope('forward_dynamics_numpy'):
            self.obs0_norm = tf.placeholder(tf.float32, shape=(None,self.dimo) , name='obs0')
            self.obs1_norm = tf.placeholder(tf.float32, shape=(None,self.dimo) , name='obs1')
            self.actions_norm = tf.placeholder(tf.float32, shape=(None,self.dimu) , name='actions')

            self.dynamics_scope = tf.get_variable_scope().name
            input = tf.concat(values=[self.obs0_norm, self.actions_norm], axis=-1)
            self.next_state_diff_tf = nn(input, [hidden] * layers + [self.dimo])
            self.next_state_norm_tf = self.next_state_diff_tf + self.obs0_norm

        # loss functions
        self.per_sample_loss_tf = tf.reduce_mean(tf.abs(self.next_state_diff_tf - self.obs1_norm + self.obs0_norm), axis=1)
        self.mean_loss_tf = tf.reduce_mean(self.per_sample_loss_tf)
        self.dynamics_grads = U.flatgrad(self.mean_loss_tf, _vars(self.dynamics_scope), clip_norm=clip_norm)

        # optimizers
        self.dynamics_adam = MpiAdam(_vars(self.dynamics_scope), scale_grad_by_procs=False)
        # initial
        tf.variables_initializer(_vars(self.dynamics_scope)).run()
        self.dynamics_adam.sync()
    
    def predict_next_state(self, obs0, actions):
        obs0_norm = self.obs_normalizer.normalize(obs0)
        action_norm = self.action_normalizer.normalize(actions)
        obs1 = self.sess.run(self.next_state_norm_tf, feed_dict={
            self.obs0_norm: obs0_norm,
            self.actions_norm:action_norm
        })
        obs1_norm = self.obs_normalizer.denormalize(obs1)
        return obs1_norm
    
    def clip_gauss_noise(self, size):
        clip_range = 0.002
        std = 0.001
        return np.clip(np.random.normal(0, std, size), -clip_range, clip_range)
    
    def update(self, obs0, actions, obs1, times=1):
        self.obs_normalizer.update(obs0)
        self.obs_normalizer.update(obs1)
        self.action_normalizer.update(actions)

        for _ in range(times):
             # use small noise for smooth
            obs0_norm = self.obs_normalizer.normalize(obs0) + self.clip_gauss_noise(size=self.dimo) 
            action_norm = self.action_normalizer.normalize(actions) + self.clip_gauss_noise(size=self.dimu)
            obs1_norm = self.obs_normalizer.normalize(obs1) 
            
            dynamics_grads, dynamics_loss, dynamics_per_sample_loss = self.sess.run(
                    [self.dynamics_grads, self.mean_loss_tf, self.per_sample_loss_tf],
                    feed_dict={
                        self.obs0_norm: obs0_norm,
                        self.actions_norm: action_norm,
                        self.obs1_norm: obs1_norm
                    })
            self.dynamics_adam.update(dynamics_grads, stepsize=self.learning_rate)
        return dynamics_loss

