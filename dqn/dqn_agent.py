import random
import numpy as np
from utils import ReplayBuffer, gather
from models.qnet import QNet
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import pickle
import os

class DqnAgent:
    """DQN agent to train for Q(s,a) using QNet
    learning_rate <float> : optimizer learning rate
    learn_period  <int>   : how often to learn
    update_period <int>   : How often to update the target network
    replay_buffer_size <int>: Replay buffer size
    batch_size <int> : training batch size
    gamma <float>    : discount factor for reward
    degp_epsilon <float>: exploration vs exploitation control
    degp_decay_rate <float>: Decay rate for the degp_epsilon
    degp_min_epsilon <float>: Minimum exploration rate
    frames <int>  : How many frames to consider to estimate motion
    tau <float> : learning rate for target network
    """
    
    def __init__(self, state_size, action_size,
                 learning_rate = 1e-3,
                 learn_period = 1,
                 update_period = 16,
                 replay_buffer_size = 100_000,
                 batch_size = 1024,
                 gamma = 0.9,
                 degp_epsilon = 1,
                 degp_decay_rate = 0.99,
                 degp_min_epsilon = 0.1,
                 frames = 2,
                 tau = 0.1
                 ):

        # Q-Network
        self.q_net_target = QNet(state_size, action_size)
        self.q_net_main = QNet(state_size, action_size)

        self.q_net_target.build(input_shape = (1, state_size * frames))
        self.q_net_main.build(input_shape = (1, state_size * frames))
        
        self.optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError()

        # Replay buffer
        self.memory = ReplayBuffer(replay_buffer_size, batch_size)

        # Parameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.learn_period = learn_period
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.degp_epsilon = degp_epsilon
        self.degp_decay_rate = degp_decay_rate
        self.degp_min_epsilon = degp_min_epsilon
        self.tau = tau
        self.update_period = update_period

        # stats
        self.step_count = 0
        self.best_average = 0

    def learn(self):
        """ Learning method to minimize temporal difference in Q(s,a) """
        samples = self.memory.batch()
        s, a, r, s_n, dones = samples

        variables = self.q_net_main.variables
        v_s_next = tf.reduce_max(self.q_net_target(s_n), axis = -1, keepdims=True) ;# max(Q(n_s,a))

        with tf.GradientTape() as tape:
            q_sa = gather(self.q_net_main(s), a)
            # TD = r + gamma * V(s') - Q(s,a)
            td = r + (self.gamma * v_s_next * (1 - dones)) - q_sa
            loss = self.loss(td, tf.zeros(td.shape))
        
        gradient = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradient, variables))

        # update target network
        if self.step_count % self.update_period == 0:
            self.soft_update()

    def soft_update(self):
        t_params = self.q_net_target.variables
        n_params = self.q_net_main.variables
        for i, (t_param, n_param) in enumerate(zip(t_params, n_params)):
            blended_params = self.tau * n_param.numpy() + (1.0 - self.tau) * t_param.numpy()
            t_params[i] = tf.Variable(blended_params)
        self.q_net_target.set_weights(t_params)

    def save(self, path):
        self.q_net_main.save_weights(path + ".h5")
        with open(path + "_mdata.pickle", 'wb') as fout:
            pickle.dump((self.memory, self.degp_epsilon, self.best_average), fout, pickle.HIGHEST_PROTOCOL)

    def load(self, path, compile=True):
        print(f"Loading model from {path}")
        self.q_net_main.load_weights(path + ".h5")
        self.q_net_target.load_weights(path + ".h5")
        fname = path + "_mdata.pickle"
        if os.path.isfile(fname):
            print (fname)
            with open(fname, 'rb') as fin:
                (self.memory, self.degp_epsilon, self.best_average) = pickle.load(fin)


    def before_episode(self):
        """ Adjusting decayed epsilon greedy policy parameters before new episode """
        self.degp_epsilon *= self.degp_decay_rate
        self.degp_epsilon = max(self.degp_epsilon, self.degp_min_epsilon)
        
    def step(self, state, action, reward, next_state, done):
        # save experience
        self.memory.add(state, action, reward, next_state, done)
        self.step_count += 1
        if self.step_count % self.learn_period == 0 and \
           len(self.memory) > self.batch_size:
            self.learn()

    def act(self, state, mode = 'train'):
        r = random.random()
        random_action = mode == 'train' and r < self.degp_epsilon
        if random_action:
            action = random.choice(np.arange(self.action_size))
        else:
            state = state.reshape(1, len(state))
            action = np.argmax(self.q_net_main(state))
        return action
