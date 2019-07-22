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

if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    
    from gym_envs.envs.Pible_env import pible_env_creator
    
    # Delete previous training
    if True:
        subprocess.run("rm -r /home/francesco/ray_results/PPO/", shell=True)
   
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
            "lr": grid_search([1e-2]),  # try different lrs
            "num_workers": 4,  # parallelism
            #"env_config": {
            #    "corridor_length": 5,
            #},
        },
    )
