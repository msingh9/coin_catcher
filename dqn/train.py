import os
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import ScreenMotion
import pygame
from dqn.dqn_agent import DqnAgent
from env.catch_coins_env import CatchCoinsEnv
import gc

env = CatchCoinsEnv()
action_size = env.action_size
state_size = env.state_size
save_path = '../coin_catcher/trained_models/dqn_model'


# Agent
agent = DqnAgent(state_size, action_size,
                 batch_size = 128,
                 replay_buffer_size = 10000,
                 update_period = 8,
                 learn_period = 1
                 )
agent.load(save_path)

training_episodes = 100
scores = []
avg_scores = []

for e in range(1, training_episodes + 1):

    state = env.reset()
    score = 0
    motion = ScreenMotion()

    if e % 1 == 0:
        agent.before_episode()

    # New Episode
    while True:
        #env.render('human')
        if not motion.is_full():
            action_id = 0 # None
        else:
            action_id = agent.act(np.ndarray.flatten(motion.get_frames()))
        action = env.action_space[action_id]

        next_state, reward, done, _ = env.step(action)

        # Add state to motion frames
        motion.add(env.state_to_list(next_state))

        # If motion is full then perform agent.step
        if motion.is_full():
            agent.step(
                motion.get_prev_frames().flatten(),
                action_id,
                reward,
                motion.get_frames().flatten(),
                done
                )

        state = next_state
        score += reward
        if done:
            gc.collect()
            break

    # Add statistics
    scores.append(score)
    avg_score = np.mean(scores[-min(10, len(scores)):])
    avg_scores.append(avg_score)
    print(f"Episode: {e}, Score: {score}. Avg Score: {round(avg_scores[-1])}, replay buffer = {sys.getsizeof(agent.memory.memory)}")

    # save the best model
    if avg_score > agent.best_average:
        agent.save(save_path)


# Training Results
plt.plot(scores)
plt.plot(avg_scores, label = 'Average Score')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.plot()
plt.show()
