import numpy as np
from typing import *

import gym
from gym import spaces

from ray.rllib.env.vector_env import VectorEnv

from pyrlprob.tests.cpp_tests.landing1d_cpp import Landing1DEnv_cpp, Landing1DVectorEnv_cpp


#Default config for Landing1DEnv
DEFAULT_CONFIG_LANDING1D = {
    "H": 40,
    "h0_min": 0.8,
    "h0_max": 1.2,
    "v0_min": -0.85,
    "v0_max": -0.75,
    "m0": 1.0,
    "tf": 1.397,
    "hf": 0.0,
    "vf": 0.0,
    "Tmax": 1.227,
    "c": 2.349,
    "g": 1.0
}


class cppLanding1DEnv(Landing1DEnv_cpp, gym.Env):
    """
    One-Dimensional Landing Problem.
    Reference: https://doi.org/10.2514/6.2008-6615
    """

    def __init__(self, 
                 config: Optional[Dict[str, Union[int, float]]] = None) -> None:
        """ 
        Class constructor 
        
        Args:
            config (dict): environment configs
        """

        if (not config) or (config is None):
            config = DEFAULT_CONFIG_LANDING1D

        Landing1DEnv_cpp.__init__(self, config)


class cppLanding1DVectorEnv(Landing1DVectorEnv_cpp, VectorEnv):
    """Vectorized version of the Landing1DEnv on multi-CPU processors.

    Contains `num_envs` Landing1DEnv instances.
    """

    def __init__(self, 
                 config: Optional[Dict[str, Union[int, float]]] = None) -> None:
        """ 
        Class constructor 
        
        Args:
            config (dict): environment configs
        """


        #Initialize parent classes
        Landing1DVectorEnv_cpp.__init__(
            self, 
            config["num_envs"], 
            config["num_threads"], 
            config)
        VectorEnv.__init__(
            self,
            observation_space=self.envs[0].observation_space,
            action_space=self.envs[0].action_space,
            num_envs=config["num_envs"])
        


