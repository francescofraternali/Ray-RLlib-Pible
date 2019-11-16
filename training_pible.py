"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
#from ray.rllib.models import ModelCatalog
#from ray.rllib.models.tf.tf_modelv2 import TFModelV2
#from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from gym.spaces import Discrete, Box

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search

tf = try_import_tf()

t_gran = 60
steps = 1

n_input_steps = 1
n_input = 24

class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config={"corridor_length": 5}):
        self.end_pos = config["corridor_length"]
        self.time = 0 
        #self.cur_pos = [0, -1, -2, -3, -4, -5, -6]
        #self.cur_pos = [x * t_gran for x in self.cur_pos]
        self.cur_pos = []
        for i in range(0, -(n_input), -n_input_steps):
            self.cur_pos.append(i)

        self.action_space = Discrete(2)
        start = -60.0 #* t_gran
        self.end = 23.0 #* t_gran
        self.observation_space = Box(
            start, self.end, shape=(n_input, ), dtype=np.float32)

    def reset(self):
        #self.cur_pos = [0, -1, -2, -3, -4]
        self.cur_pos = []
        self.time = 0
        for i in range(0, -(n_input), -n_input_steps):
            self.cur_pos.append(i)

        #self.cur_pos = [x * t_gran for x in self.cur_pos]
        return self.cur_pos

    def step(self, action):
        assert action in [0, 1], action
        #if action == 0 and self.cur_pos > 0:
        #    self.cur_pos -= 1
        #elif action == 1:
        #    self.cur_pos += 1
        #done = self.cur_pos >= self.end_pos
        #         self.cur_pos += 1.0

        self.time += 1
        if (self.time % t_gran) == 0:
            self.cur_pos = [x+(1*steps) for x in self.cur_pos]
#            print(self.cur_pos)
        else:
            self.cur_pos = [x for x in self.cur_pos]

        reward = 0

        if action == 0 and self.cur_pos[0] <= 5*t_gran:
            reward = 1
        elif action == 1 and self.cur_pos[0] >= 6*t_gran and self.cur_pos[0] <= 10*t_gran:
            reward = 1
        elif action == 0 and self.cur_pos[0] >= 11*t_gran and self.cur_pos[0] <= 12*t_gran:
            reward = 1
        elif action == 1 and self.cur_pos[0] >= 13*t_gran and self.cur_pos[0] <= 14*t_gran:
            reward = 1
        elif action == 0 and self.cur_pos[0] >= 15*t_gran and self.cur_pos[0] <= 16*t_gran:
            reward = 1
        elif action == 1 and self.cur_pos[0] >= 17*t_gran and self.cur_pos[0] <= 19*t_gran:
            reward  = 1
        elif action == 0 and self.cur_pos[0] >= 20*t_gran and self.cur_pos[0] <= 21*t_gran:
            reward = 1
        elif action == 1 and self.cur_pos[0] >= 22*t_gran and self.cur_pos[0] <= 23*t_gran:
            reward = 1
        elif action == 0 and self.cur_pos[0] >= 24*t_gran:
            reward = 1

        done = self.cur_pos[0] >= self.end
#        if done:    
#            print("tutto finitooooooooooooooooo")
        return self.cur_pos, reward, done, {}

'''
class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(
            0.0, self.end_pos, shape=(1, ), dtype=np.float32)

    def reset(self):
        self.cur_pos = 0
        return [self.cur_pos]

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        return [self.cur_pos], 1 if done else 0, done, {}


class CustomModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()
'''

if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init()
    #ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 200000,
        },
        config={
            "env": SimpleCorridor,  # or "corridor" if registered above
            #"model": {
            #    "custom_model": "my_model",
            #},
            "observation_filter": 'MeanStdFilter',
            "vf_share_layers": True,
            "lr": grid_search([1e-3, 1e-4, 1e-5, 1e-6]),  # try different lrs
            "num_workers": 1,  # parallelism
            "env_config": {
                "corridor_length": 5,
            },
        },
    )
