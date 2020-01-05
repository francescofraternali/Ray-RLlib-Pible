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
from ray.tune.logger import pretty_print

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
import json
import subprocess
import glob
from ray.rllib.agents import ppo


with open('settings.json', 'r') as f:
    data = json.load(f)

path_light = data[0]["path_light_data"]
light_divider = float(data[0]["light_divider"])
title = data[0]["title"]

curr_path = os.getcwd()
resume = ""
#path_light_data = curr_path + '/FF66_2150_Middle_Event_RL_Adapted.txt'; light_divider = 2  # light /1.5 and 1.7, SC_start = 4.0
#path_light_data = curr_path + '/FF21_2146_Corridor_Event_RL_Adapted.txt'
#path_light_data = curr_path + '/FF59_2104_Door_Event_RL_Adapted.txt'; light_divider = 1.77
path_light_data = curr_path + path_light; #light_divider = 2.35
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
        if self.train == "train":
            st = config["start_train"]
            end = config["end_train"]
            self.start_sc = config["sc_volt_start_train"]
        else:
            st = config["start_test"]
            end = config["end_test"]
            self.start_sc = config["sc_volt_start_test"]

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
        self.death = 0
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
            spaces.Box(0, 1000, shape=(num_light_input, ), dtype=np.float32),       # light
            spaces.Box(SC_volt_min, SC_volt_max, shape=(num_volt_input, ), dtype=np.float32),
            spaces.Discrete(2)      #week/weekends
        ))
        self.SC_Volt = []; self.Reward = []; self.PIR_hist = []; self.Perf = []; self.Time = []; self.Light = []; self.PIR_OnOff = []; self.State_Trans = []
        self.tot_events = 0
        self.events_detect = 0
        self.event_det_hist = []
        self.event_miss_hist = []

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
           self.death = 0
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
        self.week_end = Pible_func.calc_week(self.time)
        #self.hour = np.array([self.time.hour])
        #self.minute = np.array([self.time.minute])
        #self.light_ar = np.array([self.light])

        return (self.hour, self.light_ar, self.SC_volt, self.week_end)

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
        reward, event_det, event_miss, self.death = Pible_func.reward_func(PIR_on_off, event, SC_temp, self.death)
        self.light_ar[0] = self.light
        self.SC_volt[0] = SC_temp

        if reward > 0:
            self.events_detect += event
        self.tot_events += event

        done = self.time >= self.end

        if self.train != "train":
            self.SC_Volt.append(SC_temp); self.Reward.append(reward); self.PIR_hist.append(event); self.Time.append(self.time); self.Light.append(self.light); self.PIR_OnOff.append(action[0]); self.State_Trans.append(self.next_wake_up_time); self.event_det_hist.append(event_det); self.event_miss_hist.append(event_miss)

        if done: # one day is over
            self.end = self.time + datetime.timedelta(hours=24)
            #print(self.light_count, self.light_len)
            #sleep(0.5)
        else:
            self.time = self.time_next #next_wake_up_time # in min

        self.week_end = Pible_func.calc_week(self.time)

        #self.light_ar = np.array([self.light])
        #return (self.hour, self.minute, self.light_ar, self.SC_volt), reward, done, {}, en_prod, en_used
        info = {}
        info["Energy_used"] = en_used
        info["Energy_prod"] = en_prod
        info["Tot_events"] = self.tot_events
        info["events_detect"] = self.events_detect
        info["Death"] = self.death
        return (self.hour, self.light_ar, self.SC_volt, self.week_end), reward, done, info

    def render(self, episode, tot_rew, start, end, title):
        if start != "":
            start_pic_detail = datetime.datetime.strptime(start, '%m/%d/%y %H:%M:%S')
            end_pic_detail = datetime.datetime.strptime(end, '%m/%d/%y %H:%M:%S')
            Time = []; Light = []; PIR_OnOff = []; State_Trans = []; Reward = []; Perf = []; SC_Volt = []; PIR_hist = []; PIR_det = []; PIR_miss = []
            for i in range(0, len(self.Time)):
                line = self.Time[i]
                if start_pic_detail <= line and line <= end_pic_detail:
                    Time.append(self.Time[i]); Light.append(self.Light[i]); PIR_OnOff.append(self.PIR_OnOff[i]); State_Trans.append(self.State_Trans[i]); Reward.append(self.Reward[i]); SC_Volt.append(self.SC_Volt[i]); PIR_hist.append(self.PIR_hist[i]); PIR_det.append(self.event_det_hist[i]); PIR_miss.append(self.event_miss_hist[i])
            Pible_func.plot_hist(Time, Light, PIR_OnOff, State_Trans, Reward, self.Perf, SC_Volt, PIR_hist, PIR_det, PIR_miss, episode, tot_rew, self.events_detect, self.tot_events, title)
            self.SC_Volt = []; self.Reward = []; self.PIR_hist = []; self.Perf = []; self.Time = []; self.Light = []; self.PIR_OnOff = []; self.State_Trans = []

        else:
            Pible_func.plot_hist(self.Time, self.Light, self.PIR_OnOff, self.State_Trans, self.Reward, self.Perf, self.SC_Volt, self.PIR_hist, self.event_det_hist, self.event_miss_hist, episode, tot_rew, self.events_detect, self.tot_events, title)

def find_agent_saved():
    Agnt = 'PPO'
    # Detect latest folder for trainer to resume
    latest = 0
    path = subprocess.getoutput('eval echo "~$USER"')
    proc = subprocess.Popen("ls " + path + "/ray_results/", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = out.decode()
    spl = out.strip().split('\n')
    for i in spl:
        test = i.split('.')
        #print(test)
        if "json" not in test and len(test[0].split('_')) > 1:
            d = i.split('_')
            #print(d)
            date = d[2].split('-')
            hour = d[3].split('-')
            x = datetime.datetime(int(date[0]), int(date[1]), int(date[2]), int(hour[0]), int(hour[1]))
            if latest == 0:
                folder = i; time = x; latest = 1
                folder_found = i
            else:
                if x >= time:
                    folder = i; time = x
                    # Checking for a better folder
                    #if d[3] == "lr=0.0001":
                    #if d[3] == "lr=1e-05":
                    if 1:
                        folder_found = i

    #print("folder: ", folder_found)
    folder = folder_found

    # detect checkpoint to resume
    proc = subprocess.Popen("ls " + path + "/ray_results/" + folder + '/', stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    #print(out)
    out = out.decode()
    spl = out.strip().split('\n')
    max = 0
    for i in spl:
        tester = i.split('_')
        #print(tester, len(tester), tester[1].isdigit())
        if "checkpoint" in tester and len(tester)==2 and tester[1].isdigit():
            if int(tester[1]) > max:
                max = int(tester[1])
                iteration = i
    iteration = max
    print("\nFound folder: ", folder, "Last checkpoint found: ", iteration)

    max_mean = - 10000
    count = 0
    for line in open(path + "/ray_results/" + folder + "/result.json", 'r'):
        count += 1
        dict = json.loads(line)
        if dict['episode_reward_mean'] >= max_mean and count > 9:
            max_mean = dict['episode_reward_mean']
            #iteration = count
            iteration = dict['training_iteration']
        #data = json.loads(text)

        #for p in data["episode_reward_mean"]:
        #    print(p)
    #print(iteration)
    iter_str = str(iteration)
    iteration = (int(iter_str[:-1])* 10)
    print("Best checkpoint found:", iteration, ". Mean Reward Episode: ", max_mean)

    return folder, iteration

def check_perf_results(folder, iteration, start, end):
    Agnt = "PPO"
    path = glob.glob(subprocess.getoutput('eval echo "~$USER"') + '/ray_results/' + folder  +
                         '/checkpoint_' + str(iteration) + '/checkpoint-' + str(iteration), recursive=True)
    print(path)
    assert len(path) == 1, path
    agent = ppo.PPOAgent(config={
        "observation_filter": 'MeanStdFilter',
        "batch_mode": "complete_episodes",
        "env_config": {
            "train/test": "test",
            "start_test": start,
            "end_test": end,
            "sc_volt_start_test": '3.1',
            "num_workers": 0,  # parallelism
         },
    }, env=SimplePible)
    print(path[0])
    agent.restore(path[0])
    config = {
            "train/test": "test",
            "start_test": start,
            "end_test": end,
            "sc_volt_start_test": '3.1',
            "num_workers": 0,  # parallelism
        }
    diff_days = datetime.datetime.strptime(end, '%m/%d/%y %H:%M:%S') - datetime.datetime.strptime(start, '%m/%d/%y %H:%M:%S')
    print("days", diff_days.days)
    #exit()
    Env = SimplePible(config)
    obs = Env.reset()
    pre_action = [0, 0]
    pre_reward = 0
    tot_rew = 0
    repeat = 1
    stop = 0
    energy_used_tot = 0; energy_prod_tot = 0
    while True:
        action_0_list = []
        action_1_list = []
        for i in range(0,200):
            learned_action = agent.compute_action(
                observation = obs,
                prev_action = pre_action,
                prev_reward = pre_reward,
                #full_fetch=True
            )
            #w = agent.get_weights()
            #print("weight", w)
            #learned_action[0][0] = 1
            #learned_action[1][0] = 1
            #print("action", learned_action)
            action_0_list.append(learned_action[0][0])
            action_1_list.append(learned_action[1][0])

        action_0_avg = sum(action_0_list)/ len(action_0_list)
        action_1_avg = sum(action_1_list)/ len(action_1_list)
        #print(int(round(action_0_avg)), action_1_avg)
        #obs, reward, done, none, energy_prod, energy_used = Env.step(learned_action)
        pre_action = [int(round(action_0_avg)), action_1_avg]
        obs, reward, done, res = Env.step(pre_action)
        energy_used_tot += float(res["Energy_used"])
        energy_prod_tot += float(res["Energy_prod"])
        tot_rew += reward
        pre_reward = reward
        #pre_action = [learned_action[0][0], learned_action[1][0]]
        if done:
            obs = Env.reset()
            stop +=1
        if stop >= repeat*(diff_days.days):
            #print("observation:", obs, "action: ", learned_action, "rew: ", reward)
            break
    print("Energy Prod: ", energy_prod_tot/(repeat*diff_days.days), "Energy Used: ", energy_used_tot/(repeat*diff_days.days) )

    print("Tot reward: ", tot_rew)
    print_start = start
    print_end = end
    print("Tot events averaged per day: ", int(res["Tot_events"])/(repeat*diff_days.days))
    print("System Death Tot times: ", int(res["Death"]))
    try:
        percent = round((((res["events_detect"]) /  int(res["Tot_events"])) * 100), 2)
    except:
        percent = 0
    if (res["events_detect"]) == 0  and  int(res["Tot_events"]) == 0:
        percent = 100

    title_final = title + ('{0}%').format(percent)
    Env.render(1, tot_rew, print_start, print_end, title_final)
    Pible_func.write_results(tot_rew, start, end, percent, diff_days.days,
        energy_prod_tot/(repeat*diff_days.days), energy_used_tot/(repeat*diff_days.days), res["events_detect"],
        int(res["Tot_events"]), int(res["Death"]))
    print("Percent detected: ", percent)

    return path
    #proc = subprocess.Popen("cp -r ", stdout=subprocess.PIPE, shell=True)
'''
def training(train_test, start_train, end_train, num_workers):
    #ModelCatalog.register_custom_model("my_model", CustomModel)
    trainer = tune.run(
        "PPO",
        stop={
            "timesteps_total": data[0]["timesteps_total"],
        },
        checkpoint_freq=10,
        config={
            #"vf_clip_param": 10.0,
            "observation_filter": 'MeanStdFilter',
            "batch_mode": "complete_episodes",
            "env": SimplePible,  # or "corridor" if registered above
            #"vf_share_layers": True,
            "lr": grid_search([1e-4]),  # try different lrs
            "num_workers": num_workers,  # parallelism
            "env_config": {
                "train/test": train_test,
                "start_train": start_train,
                "end_train": end_train,
                #"start_test": start_test,
                #"end_test": end_test,
                "sc_volt_start_train": 'rand',
                "sc_volt_start_test": '3.5',
            },
        },
    )

'''
def PPO(config):
    max = -1000
    resume = config["env_config"]["resume"]
    #sleep(2)
    trainer = ppo.PPOTrainer(config=config, env=SimplePible)
    if resume != "":
        print("Restoring checkpoint: ", resume[0])
        sleep(2)
        trainer.restore(resume[0])

    # Can optionally call trainer.restore(path) to load a checkpoint.

    #for i in range(0, int(data[0]["timesteps_total"])):
    for i in range(0, 200):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))
        #if result["timesteps_total"] >= int(data[0]["timesteps_total"]):
        #    break

        if int(result["training_iteration"]) % 10 == 0:
        #if result["episode_reward_mean"] > max:
            #max = result["episode_reward_mean"]
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
            sleep(1)

if __name__ == "__main__":
    ray.init()

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    proc = subprocess.Popen("nproc", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    #print(out)
    out = out.decode()
    spl = out.strip().split('\n')
    print("Number of cores used: ", int(spl[0])-2)

    start_train = data[0]["start_train"]
    time_train = data[0]["time_train"]
    end_train = data[0]["end_train"]
    time_test = data[0]["time_test"]

    start_train_date = datetime.datetime.strptime(start_train, '%m/%d/%y %H:%M:%S')
    end_train_date = datetime.datetime.strptime(end_train, '%m/%d/%y %H:%M:%S')

    end_train_date_temp = start_train_date + datetime.timedelta(hours=time_train)
    end_train_temp = end_train_date_temp.strftime("%m/%d/%y %H:%M:%S")

    while True:
        print("train", start_train, end_train_temp)
        #sleep(10)
        #training("train", start_train, end_train_temp, int(spl[0])-2)
        config={
            "observation_filter": 'MeanStdFilter',
            "batch_mode": "complete_episodes",
            "lr" : grid_search([1e-4]),
            "num_workers": int(spl[0])-2,  # parallelism
            "env_config": {
                "train/test": "train",
                "start_train": start_train,
                "end_train": end_train,
                "sc_volt_start_train": 'rand',
                "resume": resume,
             },
        }
        tune.run(PPO, config=config)

        start_test = end_train_temp
        start_test_date = datetime.datetime.strptime(start_test, '%m/%d/%y %H:%M:%S')
        end_test_date = start_test_date + datetime.timedelta(hours=time_test)
        end_test = end_test_date.strftime("%m/%d/%y %H:%M:%S")

        # Resume folder and best checkpoint
        folder, iteration = find_agent_saved()
        agent_save = curr_path + "/Agents_Saved"
        proc = subprocess.Popen("cp -r /home/francesco/ray_results/" + folder + " " + agent_save, stdout=subprocess.PIPE, shell=True)
        print("test", start_test, end_test)
        sleep(10)
        resume = check_perf_results(folder, iteration, start_test, end_test)

        end_train_temp = end_test

        if end_test_date >= end_train_date:
            break

    print("Done")
