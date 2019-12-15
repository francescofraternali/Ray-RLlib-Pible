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
print(path_data)

# Read data
#observations, actions, consumptions, time_range, scalers = read_data(start_time, end_time, '2146', normalize=True)

# Init exp
from training_pible import SimplePible
#from gym_envs.envs.Pible_env import pible_env_creator
#register_env('Pible-v2', pible_env_creator)
Agnt = 'PPO'

#assert len(observations.shape) == 2

ray.init()

folder = "PPO_SimplePible_0_lr=0.0001_2019-12-13_12-47-46358jvutp"

print("\nFound folder: ", folder)

max_mean = - 10000
count = 0
for line in open(path_data + "/" + folder + "/result.json", 'r'):
    count += 1
    dict = json.loads(line)
    if dict['episode_reward_mean'] >= max_mean:
        max_mean = dict['episode_reward_mean']
        iteration = count
    #data = json.loads(text)

    #for p in data["episode_reward_mean"]:
    #    print(p)
iter_str = str(iteration)
iteration = (int(iter_str[:-1])* 10) + 10
print("Best checkpoint found:", iteration, ". Mean Reward Episode: ", max_mean)

sleep(1)
if True:
    path = glob.glob(folder  +
                     '/checkpoint_' + str(iteration) + '/checkpoint-' + str(iteration), recursive=True)
    assert len(path) == 1, path
    start = "12/02/19 00:00:00"
    end = "12/09/19 00:00:00"
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
    repeat = 6
    energy_used_tot = 0; energy_prod_tot = 0
    while True:
        learned_action = agent.compute_action(
            observation = obs,
            prev_action = pre_action,
            prev_reward = pre_reward
        )
        #learned_action[0][0] = 1
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

print_start = "12/29/19 04:00:00"
print_end = "12/29/19 22:00:00"
Env.render(1, tot_rew, print_start, print_end)
