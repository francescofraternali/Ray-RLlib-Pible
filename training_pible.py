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
from ray.tune import grid_search

from Pible_parameters import *
import Pible_func
#import Gen_Light_Events
import datetime
from time import sleep
import os
import random

curr_path = os.getcwd()
#path_light_data = curr_path + '/FF66_2150_Middle_Event_RL_Adapted.txt'; light_divider = 2  # light /1.5 and 1.7, SC_start = 4.0
#path_light_data = curr_path + '/FF21_2146_Corridor_Event_RL_Adapted.txt'
#path_light_data = curr_path + '/FF59_2104_Door_Event_RL_Adapted.txt'; light_divider = 1.77
path_light_data = curr_path + '/B3_2140_Stairs1F_Event_Battery_Adapted.txt'; light_divider = 2.35
#path_light_data = curr_path + '/FF5_2106_Door_Event_Battery_Adapted.txt'; light_divider = 0.7

num_hours_input = 20; num_minutes_input = 20;
num_volt_input = 20; num_light_input = 20

s_t_min_act = 0; s_t_max_act = 1; s_t_min_new = 1; s_t_max_new = 60

class SimplePible(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    #def __init__(self, config={"corridor_length": 5}):
    def __init__(self, config):
        self.train = config["train/test"]
        st = config["start"]
        end = config["end"]
        self.start_sc = config["sc_volt_start"]
        self.light_div = light_divider

        start_data_date = datetime.datetime.strptime(st, '%m/%d/%y %H:%M:%S')
        end_data_date = datetime.datetime.strptime(end, '%m/%d/%y %H:%M:%S')
        diff_date = end_data_date - start_data_date
        self.diff_days = diff_date.days

        self.file_data = []
        self.path_light_data = path_light_data
        with open(self.path_light_data, 'r') as f:
            for line in f:
                line_split = line.split("|")
                checker = datetime.datetime.strptime(line_split[0], '%m/%d/%y %H:%M:%S')
                if start_data_date <= checker and checker <= end_data_date:
                    self.file_data.append(line)

        self.next_wake_up_time = 0
        self.episode_count = 0
        self.file_data_orig = self.file_data
        self.light_len = len(self.file_data)
        line = self.file_data[0].split('|')
        self.light = int(int(line[8])/light_divider)
        self.light_count = 0
        self.days_repeat = 0
        #self.time = datetime.datetime.strptime('01/01/17 00:00:00', '%m/%d/%y %H:%M:%S')
        self.time = datetime.datetime.strptime(line[0], '%m/%d/%y %H:%M:%S')
        self.time_begin = self.time
        self.end = self.time + datetime.timedelta(hours=24)

        self.action_space = spaces.Tuple((
            spaces.Discrete(2),
            spaces.Box(s_t_min_act, s_t_max_act, shape=(1, ), dtype=np.float32)
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 23, shape=(num_hours_input, ), dtype=np.float32),       # hours
            #spaces.Box(0, 59, shape=(num_minutes_input, ), dtype=np.float32),       # minutes
            #spaces.Discrete(2),      #week/weekends
            spaces.Box(0, 1000, shape=(num_light_input, ), dtype=np.float32),       # light
            spaces.Box(SC_volt_min, SC_volt_max, shape=(num_volt_input, ), dtype=np.float32)
        ))
        self.SC_Volt = []; self.Reward = []; self.PIR_hist = []; self.Perf = []; self.Time = []; self.Light = []; self.PIR_OnOff = []; self.State_Trans = []
        self.tot_events = 0
        self.events_detect = 0

        self.hour, self.minute, self.light_ar = Pible_func.build_inputs(self.time, num_hours_input, num_minutes_input, num_light_input, self.light)

    def reset(self):
        #print("reset")
        if self.episode_count % 200 == 0:
            if  self.start_sc == "rand":
                self.SC_rand = random.uniform(SC_volt_die, SC_volt_max)
            else:
                self.SC_rand = float(self.start_sc)
            #print("reset SC", self.events_count)
            input_volt = []
            for i in range(0, num_volt_input):
                input_volt.append(self.SC_rand)
            self.SC_volt = np.array(input_volt)
        self.episode_count += 1


        if self.light_len == self.light_count: # all the light available is over, reset light and time
           self.light_count = 0
           self.days_repeat += 1
           #print("here", self.light_count)
           self.time = self.time.replace(hour=self.time_begin.hour, minute=self.time_begin.minute, second=self.time_begin.second)
           self.end = self.time + datetime.timedelta(hours=24)
           #sleep(10)

            #self.file_data = Pible_func.randomize_light_time(self.file_data_orig)
            #self.light_count = 0
            #self.tot_events = 0
            #self.events_detect = 0

            #self.events = Pible_func.events() # Build events array

        #self.hour = np.array([self.time.hour])
        #self.minute = np.array([self.time.minute])
        #self.light_ar = np.array([self.light])

        return (self.hour, self.light_ar, self.SC_volt)

    def step(self, action):
        assert action[0] in [0, 1], action
        #self.time = self.time + datetime.timedelta(0, 60*int(self.next_wake_up_time)) #next_wake_up_time # in min
        PIR_on_off = action[0]
        self.next_wake_up_time  = int((s_t_max_new-s_t_min_new)/(s_t_max_act-s_t_min_act)*(action[1]-s_t_max_act)+s_t_max_new)

        self.time_next = self.time + datetime.timedelta(minutes=self.next_wake_up_time) #next_wake_up_time # in min

        self.hour = np.roll(self.hour, 1)
        self.hour[0] = self.time.hour

        self.minute = np.roll(self.minute, 1)
        self.minute[0] = self.time.minute

        self.light = self.light_ar[0]
        self.light_ar = np.roll(self.light_ar, 1)

        SC_temp = self.SC_volt[0]
        self.SC_volt = np.roll(self.SC_volt, 1)

        #self.hour = np.array([self.time.hour])
        #self.minute = np.array([self.time.minute])

        #print(self.light_count, "light count")
        #sleep(1)
        self.light, self.light_count, event = Pible_func.light_event_func(self.time, self.time_next, self.light_count, self.light_len, self.light, self.file_data, self.days_repeat, self.diff_days, self.light_div)
        SC_temp, en_prod, en_used = Pible_func.Energy(SC_temp, self.light, PIR_on_off, self.next_wake_up_time, event)
        reward = Pible_func.reward_func(PIR_on_off, event, SC_temp)
        self.light_ar[0] = self.light
        self.SC_volt[0] = SC_temp

        if reward > 0:
            self.events_detect += event
        self.tot_events += event

        done = self.time >= self.end

        if done: # one day is over
            self.end = self.time + datetime.timedelta(hours=24)
            #print(self.light_count, self.light_len)
            #sleep(0.5)
        else:
            self.time = self.time_next #next_wake_up_time # in min

        if self.train != "train":
            self.SC_Volt.append(SC_temp); self.Reward.append(reward); self.PIR_hist.append(event); self.Time.append(self.time); self.Light.append(self.light); self.PIR_OnOff.append(action[0]); self.State_Trans.append(self.next_wake_up_time);

        self.week_end = Pible_func.calc_week(self.time)

        #self.light_ar = np.array([self.light])
        #return (self.hour, self.minute, self.light_ar, self.SC_volt), reward, done, {}, en_prod, en_used
        info = {}
        info["Energy_used"] = en_used
        info["Energy_prod"] = en_prod
        info["Tot_events"] = self.tot_events
        return (self.hour, self.light_ar, self.SC_volt), reward, done, info


    def render(self, episode, tot_rew, start, end):
        if start != "":
            start_pic_detail = datetime.datetime.strptime(start, '%m/%d/%y %H:%M:%S')
            end_pic_detail = datetime.datetime.strptime(end, '%m/%d/%y %H:%M:%S')
            Time = []; Light = []; PIR_OnOff = []; State_Trans = []; Reward = []; Perf = []; SC_Volt = []; PIR_hist = [];
            for i in range(0, len(self.Time)):
                line = self.Time[i]
                if start_pic_detail <= line and line <= end_pic_detail:
                    Time.append(self.Time[i]); Light.append(self.Light[i]); PIR_OnOff.append(self.PIR_OnOff[i]); State_Trans.append(self.State_Trans[i]); Reward.append(self.Reward[i]); SC_Volt.append(self.SC_Volt[i]); PIR_hist.append(self.PIR_hist[i])
            Pible_func.plot_hist(Time, Light, PIR_OnOff, State_Trans, Reward, self.Perf, SC_Volt, PIR_hist, episode, tot_rew, self.events_detect, self.tot_events)
        else:
            Pible_func.plot_hist(self.Time, self.Light, self.PIR_OnOff, self.State_Trans, self.Reward, self.Perf, self.SC_Volt, self.PIR_hist, episode, tot_rew, self.events_detect, self.tot_events)

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
            "lr": grid_search([1e-4, 1e-3]),  # try different lrs
            "num_workers": 1,  # parallelism
            "env_config": {
                "train/test": "train",
                #"start": "11/24/19 00:00:00",
                #"end": "12/02/19 00:00:00",
                "start": "06/23/19 00:00:00",
                "end": "06/30/19 00:00:00",
                "sc_volt_start": 'rand',
            },
        },
    )
