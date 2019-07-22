import glob
import subprocess
import ray
from ray.rllib.agents import ddpg
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from gym_envs.envs import Pible_env
from time import sleep
import os
#import arrow

#path_data = subprocess.getoutput('eval echo "~$USER"') + "/Desktop/Ray-RLlib-Pible/gym_envs/envs"
path_data = os.getcwd()

#start_time = arrow.get(2018, 4, 1, tzinfo=PT)
#end_time = arrow.get(2018, 4, 5, tzinfo=PT)

# Read data
#observations, actions, consumptions, time_range, scalers = read_data(start_time, end_time, '2146', normalize=True)

# Init exp
from gym_envs.envs.Pible_env import pible_env_creator

register_env('Pible-v2', pible_env_creator)

# Initialize action/obs
pre_action = 0
pre_reward = 0

#assert len(observations.shape) == 2

ray.init()

# Detect folder for trainer to resume
path = subprocess.getoutput('eval echo "~$USER"')
proc = subprocess.Popen("ls " + path + "/ray_results/PPO/", stdout=subprocess.PIPE, shell=True)
(out, err) = proc.communicate()
out = out.decode()
spl = out.strip().split('\n')
for i in spl:
    folder = i.split('.')
    if "json" not in folder:
        folder = i
        break
print("folder", folder)

# detect checkpoint to resume
proc = subprocess.Popen("ls " + path + "/ray_results/PPO/" + folder + '/', stdout=subprocess.PIPE, shell=True)
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
        
print("Found folder: ", folder, "Last checkpoint found: ", iteration)
sleep(3)

if True:
    path = glob.glob(subprocess.getoutput('eval echo "~$USER"') + '/ray_results/PPO/' + folder  +
    #path = glob.glob(subprocess.getoutput('eval echo "~$USER"') + '/ray_results/DDPG/DDPG_VAV-v0_0_2019-05-10_20-02-38zocesjrb' +
                     '/checkpoint_' + str(max) + '/checkpoint-' + str(max),
                     recursive=True)
    assert len(path) == 1, path
    agent = ppo.PPOAgent(config={
        "env_config": {
         "path": path_data,
         },
    }, env='Pible-v2')
    agent.restore(path[0])
    config = {
                "path": path_data,
            }
    Env = Pible_env.PibleEnv(config)
    SC_volt = Env.reset()
    
    while True:
        
        learned_action = agent.compute_action( 
	    observation = [SC_volt[0]],
            prev_action = pre_action,
            prev_reward = pre_reward
        )
        #learned_action = 0
        SC_volt, reward, done, none = Env.step(learned_action)
        
        pre_reward = reward
        pre_action = learned_action
        print("action learned", learned_action, "reward", reward)
        if done:
            break

    Env.render(0, 0)
