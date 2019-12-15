import glob
import subprocess
import ray
from ray.rllib.agents import ddpg
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents import dqn
#from gym_envs.envs import Pible_env
from time import sleep
import os
import datetime
import numpy as np
import json
#import arrow

#path_data = subprocess.getoutput('eval echo "~$USER"') + "/Desktop/Ray-RLlib-Pible/gym_envs/envs"
path_data = os.getcwd()

# Read data
#observations, actions, consumptions, time_range, scalers = read_data(start_time, end_time, '2146', normalize=True)

# Init exp
from training_pible import SimplePible
#from gym_envs.envs.Pible_env import pible_env_creator
#register_env('Pible-v2', pible_env_creator)
Agnt = 'PPO'

#assert len(observations.shape) == 2

ray.init()

# Detect latest folder for trainer to resume
latest = 0
path = subprocess.getoutput('eval echo "~$USER"')
proc = subprocess.Popen("ls " + path + "/ray_results/" + Agnt + "/", stdout=subprocess.PIPE, shell=True)
(out, err) = proc.communicate()
out = out.decode()
spl = out.strip().split('\n')
for i in spl:
    test = i.split('.')
    if "json" not in test:
        d = i.split('_')
        date = d[4].split('-')
        hour = d[5].split('-')
        x = datetime.datetime(int(date[0]), int(date[1]), int(date[2]), int(hour[0]), int(hour[1]))
        if latest == 0:
            folder = i; time = x; latest = 1
            folder_found = i
        else:
            if x >= time:
                folder = i; time = x
                # Checking for a better folder
                if d[3] == "lr=0.0001":
                #if d[3] == "lr=1e-05":
                    folder_found = i

#print("folder: ", folder_found)
folder = folder_found

# detect checkpoint to resume
proc = subprocess.Popen("ls " + path + "/ray_results/" + Agnt + "/" + folder + '/', stdout=subprocess.PIPE, shell=True)
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
print("\nFound folder: ", folder, "Last checkpoint found: ", iteration, '\n')


max_mean = - 10000
count = 0
for line in open(path + "/ray_results/" + Agnt + "/" + folder + "/result.json", 'r'):
    count += 1
    dict = json.loads(line)
    if dict['episode_reward_mean'] >= max_mean:
        max_mean = dict['episode_reward_mean']
        iteration = count
    #data = json.loads(text)

    #for p in data["episode_reward_mean"]:
    #    print(p)

iter_str = str(iteration)
iteration = (int(iter_str[:-1])* 10)
print("Best checkpoint found:", iteration, ". Mean Reward Episode: ", max_mean)


sleep(1)
if True:
    path = glob.glob(subprocess.getoutput('eval echo "~$USER"') + '/ray_results/' + Agnt +'/' + folder  +
    #path = glob.glob(subprocess.getoutput('eval echo "~$USER"') + '/ray_results/DDPG/DDPG_VAV-v0_0_2019-05-10_20-02-38zocesjrb' +
                     '/checkpoint_' + str(iteration) + '/checkpoint-' + str(iteration), recursive=True)
    assert len(path) == 1, path
    start = "10/15/19 00:00:00"
    end = "10/21/19 00:00:00"
    #start = "06/23/19 00:00:00"
    #end = "06/30/19 00:00:00"
    agent = ppo.PPOAgent(config={
    #agent = dqn.DQNAgent(config={
    #agent = ddpg.DDPGAgent(config={
        #"vf_share_layers": True,
        "observation_filter": 'MeanStdFilter',
        "batch_mode": "complete_episodes",
        #"path": path_data,
        "env_config": {
            "train/test": "test",
            "start": start,
            "end": end,
            "sc_volt_start": '4',
         },
    #}, env='Pible-v2')
    }, env=SimplePible)
    print(path[0])
    agent.restore(path[0])
    config = {
            "train/test": "test",
            "start": start,
            "end": end,
            "sc_volt_start": '4',
        }
    diff_days = datetime.datetime.strptime(end, '%m/%d/%y %H:%M:%S') - datetime.datetime.strptime(start, '%m/%d/%y %H:%M:%S')
    print(diff_days.days)
    #exit()
    Env = SimplePible(config)
    obs = Env.reset()
    pre_action = [0, 0]
    pre_reward = 0
    tot_rew = 0
    stop = 0
    repeat = 7
    energy_used_tot = 0; energy_prod_tot = 0
    while True:
        learned_action = agent.compute_action(
            observation = obs,
            prev_action = pre_action,
            prev_reward = pre_reward
        )
        #learned_action[0][0] = 0
        #learned_action[1][0] = 1
        #print(learned_action)
        #obs, reward, done, none, energy_prod, energy_used = Env.step(learned_action)
        obs, reward, done, none = Env.step(learned_action)
        #energy_used_tot += energy_used
        #energy_prod_tot += energy_prod
        tot_rew += reward
        pre_reward = reward
        pre_action = [learned_action[0][0], learned_action[1][0]]
        if done:
            obs = Env.reset()
            stop +=1
        if stop >= repeat*(diff_days.days):
            print("observation:", obs, "action: ", learned_action, "rew: ", reward)
            break
    #print("Diff En: ", round(energy_prod - energy_used, 5) , "Diff Volt: ", np.sqrt((2*abs(energy_prod - energy_used))/1.5))

print_start = "11/09/19 06:00:00"
#print_start = ""
print_end = "11/10/19 18:00:00"
Env.render(1, tot_rew, print_start, print_end)
