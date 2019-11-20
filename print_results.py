import glob
import subprocess
import ray
from ray.rllib.agents import ddpg
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents import dqn
from gym_envs.envs import Pible_env
from time import sleep
import os
import datetime
import numpy as np
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

# Initialize action/obs
pre_action = 0
pre_reward = 0

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
        else:
            if x > time:
                folder = i; time = x

print("folder", folder)

# detect checkpoint to resume
proc = subprocess.Popen("ls " + path + "/ray_results/" + Agnt + "/" + folder + '/', stdout=subprocess.PIPE, shell=True)
(out, err) = proc.communicate()
print(out)
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

print("\nFound folder: ", folder, "Last checkpoint found: ", iteration, '\n')
sleep(1)

if True:
    path = glob.glob(subprocess.getoutput('eval echo "~$USER"') + '/ray_results/' + Agnt +'/' + folder  +
    #path = glob.glob(subprocess.getoutput('eval echo "~$USER"') + '/ray_results/DDPG/DDPG_VAV-v0_0_2019-05-10_20-02-38zocesjrb' +
                     '/checkpoint_' + str(max) + '/checkpoint-' + str(max), recursive=True)
    assert len(path) == 1, path
    agent = ppo.PPOAgent(config={
    #agent = dqn.DQNAgent(config={
    #agent = ddpg.DDPGAgent(config={
        "env_config": {
            "vf_share_layers": True,
            "observation_filter": 'MeanStdFilter',
            "path": path_data,
            "corridor_length": 5,
         },
    #}, env='Pible-v2')
    }, env=SimplePible)
    print(path[0])
    agent.restore(path[0])
    '''
    config = {
                "path": path_data,
            }
    Env = Pible_env.PibleEnv(config)
    SC_volt = Env.reset()
    '''
    config = {
            "path": path_data,
            "corridor_length": 5,
        }
    Env = SimplePible(config)
    obs = Env.reset()
    pre_action = [np.array(0), np.array(0)]
    #print(SC_volt)
    tot_rew = 0
    while True:
        learned_action = agent.compute_action(
	    #observation = [SC_volt[0], SC_volt[1]],
            observation = obs,
            prev_action = pre_action,
            #prev_reward = pre_reward
        )
        #learned_action = 0
        #print(learned_action)
        obs, reward, done, none = Env.step(learned_action)
        print(reward)
        tot_rew += reward
        pre_reward = reward
        pre_action = learned_action
        print("action learned", learned_action, "reward", reward)
        if done:
            break

    Env.render(1, tot_rew)
