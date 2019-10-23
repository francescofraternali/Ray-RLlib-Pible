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
x = 1
input_min = -5.0*x
input_max = 23.0*x

class SimplePible(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config={"corridor_length": 5}):
        #self.end_pos = config["corridor_length"]
        
        self.cur_pos = [norm(0), norm(-1*x), norm(-2*x), norm(-3*x), norm(-4*x)]
        self.action_space = Discrete(2)
        self.observation_space = Box(norm(input_min), norm(input_max), shape=(5, ), dtype=np.float32)

    def reset(self):
        self.cur_pos = [norm(0), norm(-1*x), norm(-2*x), norm(-3*x), norm(-4*x)]
        return self.cur_pos

    def step(self, action):
        assert action in [0, 1], action
        steps = 1.0/28.0
        #print(steps)
        #print(self.cur_pos)
        self.cur_pos = [y + steps for y in self.cur_pos]

        reward = 0
        
        if action == 0 and self.cur_pos[0] <= norm(5*x):
            reward = 1
        elif action == 1 and self.cur_pos[0] >= norm(6*x) and self.cur_pos[0] <= norm(10*x):
            reward = 1
        elif action == 0 and self.cur_pos[0] >= norm(11*x) and self.cur_pos[0] <= norm(12*x):
            reward = 1
        elif action == 1 and self.cur_pos[0] >= norm(13*x) and self.cur_pos[0] <= norm(14*x):
            reward = 1
        elif action == 0 and self.cur_pos[0] >= norm(15*x) and self.cur_pos[0] <= norm(16*x):
            reward = 1
        elif action == 1 and self.cur_pos[0] >= norm(17*x) and self.cur_pos[0] <= norm(19*x):
            reward  = 1
        elif action == 0 and self.cur_pos[0] >= norm(20*x) and self.cur_pos[0] <= norm(21*x):
            reward = 1
        elif action == 1 and self.cur_pos[0] >= norm(22*x) and self.cur_pos[0] <= norm(23*x):
            reward = 1
        #elif action == 0 and self.cur_pos[0] >= norm(24*x):
        #    reward = 1

        done = self.cur_pos[0] >= norm(input_max)

        return self.cur_pos, reward, done, {}
    
def norm(val):
    norm = (val - input_min)/(input_max-input_min)
    #norm = val
    return norm

'''
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
            "env": SimplePible,  # or "corridor" if registered above
            #"model": {
            #    "custom_model": "my_model",
            #},
            "vf_share_layers": True,
            "lr": grid_search([1e-2, 1e-3, 1e-4, 1e-5]),  # try different lrs
            "num_workers": 1,  # parallelism
            "env_config": {
                "corridor_length": 5,
            },
        },
    )
