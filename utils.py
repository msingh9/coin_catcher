import tensorflow as tf
from collections import deque
import numpy as np
import random

def gather(params, idx):
    idx = tf.stack([tf.range(tf.shape(idx)[0]), idx[:, 0]], axis = -1)
    out = tf.gather_nd(params, idx)
    out = tf.expand_dims(out, axis = 1)
    return out

class ScreenMotion:
    """ To capture frames and store to estimate motion """
    frame_number = 2

    def __init__(self) -> None:
        self.frames = deque(maxlen = ScreenMotion.frame_number + 1)

    def add(self, state):
        self.frames.append(state)

    def get_frames(self):
        F = ScreenMotion.frame_number
        return np.stack([self.frames[i] for i in range(1, F+1)])

    def get_prev_frames(self):
        F = ScreenMotion.frame_number
        return np.stack([self.frames[i] for i in range(0, F)])

    def is_full(self):
        return len(self.frames) == ScreenMotion.frame_number + 1

class ReplayBuffer:
    """ Replay buffer """

    def __init__(self, buffer_size, batch_size):
        """
        buffer_size: maximum buffer length
        batch_size: size of training batch
        """

        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def batch(self, batch_size = None):
        """ Randomly sample a batch of experiences from memory. """
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        experiences = random.sample(self.memory, k = batch_size)
        states = np.vstack([e[0] for e in experiences]).astype(np.uint8)
        actions = np.vstack([e[1] for e in experiences]).astype(np.uint8)
        rewards = np.vstack([e[2] for e in experiences]).astype(np.uint8)
        next_states = np.vstack([e[3] for e in experiences]).astype(np.uint8)
        dones = np.vstack([e[4] for e in experiences]).astype(np.uint8)
        return states, actions, rewards, next_states, dones

if __name__ == '__main__':
    params = tf.constant([[1, 2, 3], [4, 5, 6]])
    idx = tf.constant([[0], [2]])
    out = gather(params, idx)
    print (out.numpy())
