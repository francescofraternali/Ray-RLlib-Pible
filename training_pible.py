"""Example of a custom gym environment and model. Run this for a demo.
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
from gym.spaces import Discrete, Box

import ray
from ray import tune
from ray.tune import grid_search
from ray.tune.registry import register_env
import subprocess
import os

user = subprocess.getoutput('eval echo "~$USER"')
#path_data = user + "/Desktop/Ray-RLlib-Pible/gym_envs/envs" 
path_data = os.getcwd()

if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    
    from gym_envs.envs.Pible_env import pible_env_creator
    
    # Delete previous training
    if False:
        subprocess.run("rm -r " + user + "/ray_results/PPO/", shell=True)
   
    register_env("Pible-v2", pible_env_creator)
    ray.init()
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 500000,
        },
	checkpoint_freq=10,
        config={
            "env": "Pible-v2",  # or "corridor" if registered above
            "lr": grid_search([0.0001e-1]),  # try different lrs
            "num_workers": 0,  # parallelism
            #"sample_batch_size": 32,
            "env_config": {
                "path": path_data,
                #"corridor_length": 5,
            },
        },
    )
