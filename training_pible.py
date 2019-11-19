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

t_gran = 60; steps = 1

n_input_steps = 1; n_input = 24

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
        self.SC_Volt = []; self.SC_Norm = []; self.Reward = []; self.PIR_hist = []; self.Perf = []; self.Time = []; self.Light = []; self.Action = []
        self.cur_pos = []
        self.time = 0
        self.events = Pible_func.events()
        self.SC_volt = np.array([SC_begin])
        #self.cur_pos = np.array([])

        for i in range(0, -(n_input), -n_input_steps):
            self.cur_pos.append(i)

        self.cur_pos = np.array(self.cur_pos)

        return (self.cur_pos, self.SC_volt)

    def step(self, action):
        assert action in [0, 1], action

        next_wake_up_time = 1 # in min
        self.time += 1 # in min
        hour = int(self.time / t_gran)

        if (self.time % t_gran) == 0:
            self.cur_pos = [x+(1*steps) for x in self.cur_pos]
        else:
            self.cur_pos = [x for x in self.cur_pos]

        light = int(Pible_func.light_env(hour))

        event, self.events = Pible_func.event_func(self.time, next_wake_up_time, self.events)

        self.SC_volt = Pible_func.Energy(self.SC_volt, light, action, next_wake_up_time, event)

        reward = Pible_func.reward_func(action, event, self.SC_volt)

        done = self.cur_pos[0] >= self.end

        self.SC_Volt.append(self.SC_volt); self.Reward.append(reward); self.PIR_hist.append(event); self.Time.append(self.time); self.Light.append(light); self.Action.append(action);
        #print((self.cur_pos, self.SC_volt))
        #self.SC_volt = np.array([SC_begin])
        return (self.cur_pos, self.SC_volt), reward, done, {}

    def render(self, episode, tot_rew):
        Pible_func.plot_hist(self.Time, self.Light, self.Action, self.Reward, self.Perf, self.SC_Volt, self.SC_Norm, self.PIR_hist, episode, tot_rew)

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
        checkpoint_freq=10,
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
