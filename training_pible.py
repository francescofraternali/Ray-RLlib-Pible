"""Example of a custom gym environment and model. Run this for a demo.
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym.spaces import Discrete, Box
from gym import spaces, logger

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search

from Pible_parameters import *
import Pible_func

tf = try_import_tf()

t_gran = 60
steps = 1

n_input_steps = 1
n_input = 24

class SimplePible(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config={"corridor_length": 5}):
        self.end_pos = config["corridor_length"]
        self.time = 0
        #self.cur_pos = [0, -1, -2, -3, -4, -5, -6]
        #self.cur_pos = [x * t_gran for x in self.cur_pos]
        #self.cur_pos = []
        #for i in range(0, -(n_input), -n_input_steps):
        #    self.cur_pos.append(i)
        #self.cur_pos = np.array([self.cur_pos])

        self.action_space = Discrete(2)
        start = -24.0 #* t_gran
        self.end = 23.0 #* t_gran

        #self.observation_space = Box(
        #    start, self.end, shape=(n_input, ), dtype=np.float32)
        self.observation_space = spaces.Tuple((
            spaces.Box(start, self.end, shape=(n_input, ), dtype=np.float32),
            spaces.Box(2.3, 5.5, shape=(1, ), dtype=np.float32)
            #spaces.Discrete(24)
        ))

    def reset(self):
        #self.cur_pos = [0, -1, -2, -3, -4]
        self.cur_pos = []
        self.time = 0
        self.SC_volt = np.array([SC_begin])
        #self.cur_pos = np.array([])

        for i in range(0, -(n_input), -n_input_steps):
            self.cur_pos.append(i)

        self.cur_pos = np.array(self.cur_pos)

        return (self.cur_pos, self.SC_volt)

    def step(self, action):
        assert action in [0, 1], action

        self.time += 1
        hour = int(self.time / t_gran)
        #print(self.time, hour)

        if (self.time % t_gran) == 0:
            self.cur_pos = [x+(1*steps) for x in self.cur_pos]
#            print(self.cur_pos)
        else:
            self.cur_pos = [x for x in self.cur_pos]

        light = int(Pible_func.light_env(hour))

        self.SC_volt, time_passed, event = Pible_func.Energy(self.SC_volt, light, action)

        reward = Pible_func.reward_func(action, self.cur_pos, t_gran)

        done = self.cur_pos[0] >= self.end
        #print((self.cur_pos, self.SC_volt))
        #self.SC_volt = np.array([SC_begin])
        return (self.cur_pos, self.SC_volt), reward, done, {}

if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init()
    #ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 200000,
        },
        config={
            "observation_filter": 'MeanStdFilter',
            "env": SimplePible,  # or "corridor" if registered above
            "vf_share_layers": True,
            "lr": grid_search([1e-2, 1e-3, 1e-4, 1e-5]),  # try different lrs
            "num_workers": 1,  # parallelism
            "env_config": {
                "corridor_length": 5,
            },
        },
    )
