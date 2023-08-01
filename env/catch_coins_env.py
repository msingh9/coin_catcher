## Notice of copyright
""" The original code is from book "Practical Deep Reinforcement
Learning with Python" by Ivan Gridin. I modified a bit.

"""



from math import ceil
from random import random, randint
from time import sleep
import gym
import collections


class CatchCoinsEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, display_width = 10, display_height = 10, density = .8):
        self.display_width = display_width
        self.display_height = display_height
        self.density = density
        self.display = collections.deque(maxlen = display_height)
        self.last_action = None
        self.last_reward = None
        self.total_score = 0
        self.v_position = 0
        self.game_scr = None
        self.action_space = [-1, 0, 1]
        self.MAXTIME = 200
        self.current_time = None
        self.state_size = display_height * display_width + 1
        self.action_size = len(self.action_space)

    def step(self, action):
        self.last_action = action
        self.v_position = min(max(self.v_position + action, 0), self.display_width - 1)
        if self.display[0][self.v_position] == 'B':
            reward = 0;
        else:
            reward = self.display[0][self.v_position]
        self.last_reward = reward
        self.total_score += reward
        self.display.append(self.line_generator())
        state = self.display, self.v_position
        done = self.display[0][self.v_position] == 9 or self.current_time == 0
        self.current_time -= 1
        info = {}
        #print (f"{self.current_time}")
        return state, reward, done, info

    def reset(self):
        for _ in range(self.display_height):
            self.display.append(self.line_generator())
        self.v_position = ceil(self.display_width / 2)
        state = self.display, self.v_position
        self.current_time = self.MAXTIME
        self.total_score = 0
        return state

    def line_generator(self):
        line = [0] * self.display_width
        if random() > (1 - self.density):
            r = random()
            if r < .6:
                v = 1
            elif r < .9:
                v = 2
            else:
                if r < 0.92:
                    v = 9
                else:
                    v = 3

            line[randint(0, self.display_width - 1)] = v
        return line

    def render(self, mode = "human"):
        if mode == "human":
            self._render_human()
        else:
            raise Exception('Not Implemented')

    def _render_human(self):
        from env.catch_coins_screen import CatchCoinsScreen
        if not self.game_scr:
            self.game_scr = CatchCoinsScreen(
                h = self.display_height,
                w = self.display_width
            )

        if self.last_reward:
            self.game_scr.plus()
            sleep(.05)

        self.game_scr.update(
            self.display,
            self.v_position,
            self.total_score,
            self.current_time
        )

    def state_to_list(self, state):
        display, position = state
        ans = [i for x in display for i in x]
        ans.append(position)
        return ans
