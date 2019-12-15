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
#import Gen_Light_Events
import datetime
from time import sleep
import os
import random
from scipy.ndimage.interpolation import shift

tf = try_import_tf()
curr_path = os.getcwd()
path_light_data = curr_path + '/FF66_2150_Middle_Event_RL_Adapted.txt'  # light /1.7
#path_light_data = curr_path + '/FF21_2146_Corridor_Event_RL_Adapted.txt'

num_hours_input = 1; num_minutes_input = 1; num_volt_input = 2; num_light_input = 1

s_t_min_act = 0; s_t_max_act = 1
s_t_min_new = 1; s_t_max_new = 15


class SimplePible(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    #def __init__(self, config={"corridor_length": 5}):
    def __init__(self, config):

        st = config["start"]
        end = config["end"]
        self.start_sc = config["sc_volt_start"]

        start_data_date = datetime.datetime.strptime(st, '%m/%d/%y %H:%M:%S')
        end_data_date = datetime.datetime.strptime(end, '%m/%d/%y %H:%M:%S')

        self.file_data = []
        self.path_light_data = path_light_data
        #if self.traintest == "train":
        if 1:
            with open(self.path_light_data, 'r') as f:
                for line in f:
                    line_split = line.split("|")
                    checker = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')
                    #print(start_data_date, checker, end_data_date)
                    if start_data_date <= checker and checker <= end_data_date:
                        self.file_data.append(line)

        self.next_wake_up_time = 0
        self.events_count = 0

        #with open(self.path_light_data, 'r') as f:
        #    self.file_data = f.readlines()
        self.file_data_orig = self.file_data
        self.light_len = len(self.file_data)
        line = self.file_data[0].split('|')
        self.light = int(int(line[8])/1.7)
        self.light_count = self.light_len
        #self.time = datetime.datetime.strptime('01/01/17 00:00:00', '%m/%d/%y %H:%M:%S')
        self.time = datetime.datetime.strptime(line[0], '%m/%d/%y %H:%M:%S')
        self.time_begin = self.time
        self.end = self.time + datetime.timedelta(hours=24)

        self.action_space = spaces.Tuple((
            spaces.Discrete(2),
            spaces.Box(s_t_min_act, s_t_max_act, shape=(1, ), dtype=np.float32)
        ))
        self.observation_space = spaces.Tuple((
            #spaces.Box(-24, self.end, shape=(n_input, ), dtype=np.float32),
            spaces.Box(0, 23, shape=(num_hours_input, ), dtype=np.float32),       # hours
            spaces.Box(0, 59, shape=(num_minutes_input, ), dtype=np.float32),       # minutes
            spaces.Box(0, 2000, shape=(num_light_input, ), dtype=np.float32),       # light
            spaces.Discrete(2),      #week/weekends
            spaces.Box(SC_volt_min, SC_volt_max, shape=(num_volt_input, ), dtype=np.float32)
            #spaces.Discrete(24)
        ))

    def reset(self):
        #self.time = datetime.datetime.strptime(Splitted[0], '%m/%d/%y %H:%M:%S')
        #os.system('python Gen_Light_Events.py')
        #print("reset")
        if self.events_count % 200 == 0:
            if  self.start_sc == "rand":
                self.SC_rand = random.uniform(SC_volt_die, SC_volt_max)
            else:
                self.SC_rand = float(self.start_sc)
            #print("reset SC", self.events_count)
            input_volt = []
            for i in range(0, num_volt_input):
                input_volt.append(self.SC_rand)
            self.SC_volt = np.array(input_volt)
        self.events_count += 1

        if self.light_len == self.light_count:
            self.SC_Volt = []; self.SC_Norm = []; self.Reward = []; self.PIR_hist = []; self.Perf = []; self.Time = []; self.Light = []; self.PIR_OnOff = []; self.State_Trans = []
            self.time = self.time_begin
            weekday = self.time.weekday()
            if weekday > 4:
                self.week_end = 1
            else:
                self.week_end = 0
            self.end = self.time + datetime.timedelta(hours=24)
            #self.file_data = Pible_func.randomize_light_time(self.file_data_orig)
            self.light_count = 0
            self.tot_events = 0
            self.events_detect = 0

            #print("reset counter")
            #sleep(10)
        #self.events = Pible_func.events() # Build events array

            input_hour = []
            time_temp = self.time
            for i in range(0, num_hours_input):
                input_hour.append(time_temp.hour)
                time_temp = time_temp - datetime.timedelta(hours=1)
            self.hour = np.array(input_hour)

            input_minute = []
            time_temp = self.time
            for i in range(0, num_minutes_input):
                input_minute.append(time_temp.minute)
                time_temp = time_temp - datetime.timedelta(minutes=1)
            self.minute = np.array(input_minute)

        #self.hour = np.array([self.time.hour])
        #self.minute = np.array([self.time.minute])
        self.light_ar = np.array([self.light])


        return (self.hour, self.minute, self.light_ar, self.week_end, self.SC_volt)

    def step(self, action):
        assert action[0] in [0, 1], action
        #self.time = self.time + datetime.timedelta(0, 60*int(self.next_wake_up_time)) #next_wake_up_time # in min
        PIR_on_off = action[0]
        self.next_wake_up_time  = int((s_t_max_new-s_t_min_new)/(s_t_max_act-s_t_min_act)*(action[1]-s_t_max_act)+s_t_max_new)

        self.time_next = self.time + datetime.timedelta(0, 60*int(self.next_wake_up_time)) #next_wake_up_time # in min

        time_temp = self.time
        input_hour = []
        for i in range(0, num_hours_input):
            input_hour.append(time_temp.hour)
            time_temp = time_temp - datetime.timedelta(hours=1)
        self.hour = np.array(input_hour)

        input_minute = []
        time_temp = self.time
        for i in range(0, num_minutes_input):
            input_minute.append(time_temp.minute)
            time_temp = time_temp - datetime.timedelta(minutes=1)
        self.minute = np.array(input_minute)

        #self.hour = np.array([self.time.hour])
        #self.minute = np.array([self.time.minute])
        #light, self.light_count = Pible_func.light_env(self.time, self.light_count, self.light_len)
        #print(self.SC_volt)
        self.SC_volt = np.roll(self.SC_volt, 1)

        self.light, self.light_count, event = Pible_func.light_event_func(self.time, self.time_next, self.light_count, self.light_len, self.light, self.file_data)
        self.SC_volt[0] = Pible_func.Energy(self.SC_volt[1], self.light, PIR_on_off, self.next_wake_up_time, event)
        reward = Pible_func.reward_func(PIR_on_off, event, self.SC_volt[0])
        #print(self.SC_volt)
        #sleep(10)
        if reward > 0:
            self.events_detect += event
        self.tot_events += event

        done = self.time >= self.end

        if done:
            self.end = self.time + datetime.timedelta(hours=24)
        #    print("Done")
            #sleep(1)
        #print(event, reward)
        #sleep(1)
        self.SC_Volt.append(self.SC_volt[0]); self.Reward.append(reward); self.PIR_hist.append(event); self.Time.append(self.time); self.Light.append(self.light); self.PIR_OnOff.append(action[0]); self.State_Trans.append(self.next_wake_up_time);
        #if self.light_count == 0:
        #    self.time = self.time_begin
        weekday = self.time.weekday()
        if weekday > 4:
            self.week_end = 1
        else:
            self.week_end = 0
        self.time = self.time + datetime.timedelta(0, 60*int(self.next_wake_up_time)) #next_wake_up_time # in min

        self.light_ar = np.array([self.light])
        return (self.hour, self.minute, self.light_ar, self.week_end, self.SC_volt), reward, done, {}

    def render(self, episode, tot_rew):
        Pible_func.plot_hist(self.Time, self.Light, self.PIR_OnOff, self.State_Trans, self.Reward, self.Perf, self.SC_Volt, self.SC_Norm, self.PIR_hist, episode, tot_rew, self.events_detect, self.tot_events)

if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init()
    #ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 100e6,
        },
        checkpoint_freq=10,
        config={
            #"vf_clip_param": 10.0,
            "observation_filter": 'MeanStdFilter',
            #'train_batch_size' = 4000,
            #'train_batch_size': 65536,  # try with 3 millions
            #'sgd_minibatch_size': 4096,
            "batch_mode": "complete_episodes",
            "env": SimplePible,  # or "corridor" if registered above
            #"vf_share_layers": True,
            "lr": grid_search([1e-4]),  # try different lrs
            "num_workers": 3,  # parallelism
            "env_config": {
                "start": "11/24/19 00:00:00",
                "end": "12/02/19 00:00:00",
                "sc_volt_start": '3.5',
            },
        },
    )
