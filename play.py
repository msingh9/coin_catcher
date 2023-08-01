import random
from time import sleep
from env.catch_coins_env import CatchCoinsEnv
import pygame
from policy import Policy
from dqn.dqn_agent import DqnAgent
from utils import ScreenMotion
import numpy as np

policy = "dqn" ; # "q_learning"; "monte_carlo" ; # human|random|monte_carlo|q_learning|dqn
fc_units = [128, 64, 32, 16, 8]
model_path = '../coin_catcher/trained_models'

env = CatchCoinsEnv()
state = env.reset()
my_policy = Policy(env, policy)

state = env.reset()
if policy == "monte_carlo":
    my_policy.read_Q(model_path + "/catchcoin_m.pickle")

if policy == "q_learning":
    my_policy.read_Q(model_path + "/catchcoin_q.pickle")

if policy == 'dqn':
    agent = DqnAgent(env.state_size, env.action_size, fc_units)
    agent.load(model_path + "/dqn_model")
    motion = ScreenMotion()

while True:
    env.render('human')
    if policy == 'dqn':
        if not motion.is_full():
            action_id = 0
        else:
            action_id = agent.act(np.ndarray.flatten(motion.get_frames()), mode="play")
        action = env.action_space[action_id]
    else:
        action = my_policy.get_action(state)
    state, reward, done, debug = env.step(action)
    if policy == "dqn":
        motion.add(env.state_to_list(state))

    if done:
        break
    sleep(.1)

score = env.total_score
print (f"Total score = {score}")
env.close()
