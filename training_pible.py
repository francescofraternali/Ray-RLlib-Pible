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
import datetime

tf = try_import_tf()

steps = 1

n_input_steps = 1; n_input = 24

class SimplePible(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config={"corridor_length": 5}):
        self.end_pos = config["corridor_length"]
        #self.time = 0
        self.time = datetime.datetime.strptime('01/01/17 00:00:00', '%m/%d/%y %H:%M:%S')
        #self.end = self.time + datetime.timedelta(0, 24*60*60)
        #self.cur_pos = [0, -1, -2, -3, -4, -5, -6]
        #self.cur_pos = [x * t_gran for x in self.cur_pos]
        #self.cur_pos = []
        #for i in range(0, -(n_input), -n_input_steps):
        #    self.cur_pos.append(i)
        #self.cur_pos = np.array([self.cur_pos])

        #self.action_space = Discrete(2)
        #self.observation_space = Box(
        #    start, self.end, shape=(n_input, ), dtype=np.float32)
        self.action_space = spaces.Tuple((
            spaces.Discrete(2),
            spaces.Discrete(4),
            #spaces.Box(1, 2, shape=(1, ), dtype=np.float32)
        ))
        #start = -24.0 #* t_gran
        self.end = 23.0 #* t_gran
        self.observation_space = spaces.Tuple((
            spaces.Box(-24, self.end, shape=(n_input, ), dtype=np.float32),
            spaces.Box(SC_volt_min, SC_volt_max, shape=(1, ), dtype=np.float32)
            #spaces.Discrete(24)
        ))

    def reset(self):
        #self.cur_pos = [0, -1, -2, -3, -4]
        self.cur_pos = []; self.SC_Volt = []; self.SC_Norm = []; self.Reward = []; self.PIR_hist = []; self.Perf = []; self.Time = []; self.Light = []; self.PIR_OnOff = []; self.State_Trans = []
        #self.time = 0
        #self.time = datetime.datetime.strptime(Splitted[0], '%m/%d/%y %H:%M:%S')
        self.time = datetime.datetime.strptime('01/01/17 00:00:00', '%m/%d/%y %H:%M:%S')
        #self.end = self.time + datetime.timedelta(0, 24*60*60)

        self.events = Pible_func.events() # Build events array
        self.SC_volt = np.array([SC_begin])

        for i in range(0, -(n_input), -n_input_steps):
            self.cur_pos.append(i)

        self.cur_pos = np.array(self.cur_pos)

        return (self.cur_pos, self.SC_volt)

    def step(self, action):
        PIR_on_off = action[0]
        #next_wake_up_time = action[1] # It is the state_trans in min
        if action[1] == 0:
            next_wake_up_time = 1
        else:
            next_wake_up_time = action[1]*5

        assert action[0] in [0, 1], action

        #next_wake_up_time = 1 # in min
        #self.time += 1 # in min
        self.time = self.time + datetime.timedelta(0, 60*int(next_wake_up_time)) #next_wake_up_time # in min
        self.time_next = self.time + datetime.timedelta(0, 60*int(next_wake_up_time)) #next_wake_up_time # in min
        #hour = int(self.time / 60)
        hour = self.time.hour

        for i in range(0,len(self.cur_pos)):
            self.cur_pos[i] = self.time.hour - i

        #print(self.time, hour, self.end, self.cur_pos)
        #if int(self.time % 60) == 0:
        #    self.cur_pos = [x+(1*steps) for x in self.cur_pos]
        #else:
        #    self.cur_pos = [x for x in self.cur_pos]
        #print(self.cur_pos)
        light = int(Pible_func.light_env(hour))

        event, self.events = Pible_func.event_func(self.time, self.time_next, self.events)

        self.SC_volt = Pible_func.Energy(self.SC_volt, light, PIR_on_off, next_wake_up_time, event)

        reward = Pible_func.reward_func(PIR_on_off, event, self.SC_volt)

        done = self.cur_pos[0] >= self.end

        self.SC_Volt.append(self.SC_volt); self.Reward.append(reward); self.PIR_hist.append(event); self.Time.append(self.time); self.Light.append(light); self.PIR_OnOff.append(action[0]); self.State_Trans.append(action[1]);
        #print((self.cur_pos, self.SC_volt))
        #self.SC_volt = np.array([SC_begin])
        return (self.cur_pos, self.SC_volt), reward, done, {}

    def render(self, episode, tot_rew):
        Pible_func.plot_hist(self.Time, self.Light, self.PIR_OnOff, self.State_Trans, self.Reward, self.Perf, self.SC_Volt, self.SC_Norm, self.PIR_hist, episode, tot_rew)

if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init()
    #ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 1000000,
        },
        checkpoint_freq=10,
        config={
            "observation_filter": 'MeanStdFilter',
            "env": SimplePible,  # or "corridor" if registered above
            #"vf_share_layers": True,
            "lr": grid_search([1e-3]),  # try different lrs
            "num_workers": 1,  # parallelism
            "env_config": {
                "corridor_length": 5,
            },
        },
    )
