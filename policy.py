from random import random, choice
import gym
import numpy as np
from collections import defaultdict
import dill as pickle
from env.catch_coins_env import CatchCoinsEnv
import pygame
import os

class Policy():

    def __init__(self, env, policy="random", epsilon=0.1):
        """ policy = random|human|monte_carlo"""
        
        self.G = None
        self.N = None
        self.Q = None
        self.policy = policy
        self.env = env
        self.epsilon = epsilon

    def get_action(self, state=None):
        """ Get action based on policy """
        
        if self.policy == "random":
            return choice(self.env.action_space)

        if self.policy == "human":
            events = pygame.event.get()
            action = 0
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = -1
                    elif event.key == pygame.K_RIGHT:
                        action += 1
                    else:
                        action = 0

            return action

        if self.policy == "monte_carlo" or self.policy == "q_learning":
            state = self._transform(state)
            r = random()
            if r <= self.epsilon or state not in self.Q:
                action = choice(self.env.action_space)
            else:
                action = self.env.action_space[np.argmax(self.Q[state])]
            return action

    def read_Q(self, qfile):
        with open(qfile, "rb") as fin:
            self.G, self.N, self.Q = pickle.load(fin)

    def _transform(self, state):
        """ state space is too big, manual transformation to experiment """
        display, position = state
        #sum_rows = [str(np.sum(display[i])) for i in range(self.env.display_width)]
        sum_columns = [np.sum([display[i][j] for j in range(self.env.display_height)]) for i in range(self.env.display_width)]
        col_is_empty = [sum_columns[i] != 0 for i in range(self.env.display_width)]
        col_is_m8 = [sum_columns[i] > 8 for i in range(self.env.display_width)]
        return (position, str(col_is_empty), str(col_is_m8))
    
    def _train_monte_carlo(self, train_episodes, gamma):
        for e in range(1, train_episodes + 1):
            state_history = []
            state = self.env.reset()
            playtime = 0
            while True:
                action = choice(self.env.action_space)
                next_state, reward, done, __ = self.env.step(action)
                if done: 
                    break
                playtime += 1
                state_history.append([state, action, reward, next_state])
                state = next_state

            if len(state_history) == 0:
                continue
            states, actions, rewards, next_state = zip(*state_history)
            discounts = np.array([gamma**i for i in range(playtime+1)])

            for i, st in enumerate(states):
                g = sum(rewards[i:] * discounts[:-(1 + i)])
                # transform the state
                st = self._transform(st)
                self.G[st][actions[i]] += g
                self.N[st][actions[i]] += 1.0
                self.Q[st][actions[i]] = self.G[st][actions[i]] / self.N[st][actions[i]]

            if e % 10_000 == 0:
                print (f"Episodes: {e}/{train_episodes}")

    def _train_q_learing(self, train_episodes, alpha, gamma):
        for e in range(1, train_episodes + 1):
            state = self.env.reset()
            while True:
                r = random()
                action = self.get_action(state)
                next_state, reward, done, __ = self.env.step(action)
                if done: 
                    break
                st = self._transform(state)
                td = (reward + self.gamma * np.max(self.Q[self._transform(next_state)])) - \
                    self.Q[st][action]
                self.Q[st][action] += alpha * round(td, 2)
                state = next_state

    def train(self, train_episodes, lt="monte_carlo", alpha=0.3, gamma=1):
        """"
        lt = "monte_carlo|q_learning"
        """

        if not self.Q:
            self.G = defaultdict(lambda: np.zeros(3))
            self.N = defaultdict(lambda: np.zeros(3))
            self.Q = defaultdict(lambda: np.zeros(3))

        if lt == "monte_carlo":
            self._train_monte_carlo(train_episodes, gamma)
            return
        if lt == "q_learning":
            self._train_q_learning(train_episodes, alpha, gamma)
            return
        exit(f"{lt} traning method is not implemented")


    def write_Q(self, qfile):
        with open(qfile, "wb") as fout:
            pickle.dump((self.G, self.N, self.Q), fout, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    env = CatchCoinsEnv()

    if 0:
        q_file = "trained_models/catchcoin_m.pickle"
        my_policy = Policy(env, policy="monte_carlo")
        my_policy.read_Q(q_file)
        my_policy.train(100000)
        my_policy.write_Q(q_file)

        my_policy.read_Q(q_file)
        for i in range(50):
            state = env.reset()
            print(my_policy.get_action(state))


    if 1:
        q_file = "trained_models/catchcoin_q.pickle"
        my_policy = Policy(env, policy="q_learning")
        my_policy.read_Q(q_file)
        my_policy.train(100000)
        my_policy.write_Q(q_file)

        my_policy.read_Q(q_file)
        for i in range(50):
            state = env.reset()
            print(my_policy.get_action(state))
    

    env.close()

    
