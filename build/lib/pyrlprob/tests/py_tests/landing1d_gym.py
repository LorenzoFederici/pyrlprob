import numpy as np
from numpy.random import uniform
from typing import *

import gym
from gym import spaces

from pyrlprob.tests.py_tests.landing1d_dyn import *


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


class pyLanding1DEnvGym(gym.Env):
    """
    One-Dimensional Landing Problem.
    Reference: https://doi.org/10.2514/6.2008-6615
    """

    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None) -> None:
        """ 
        Class constructor 
        
        Args:
            config (dict): environment configs
        """

        super().__init__()

        if (not config) or (config is None):
            config = DEFAULT_CONFIG_LANDING1D

        #Create class attributes
        for key, item in config.items():
            setattr(self, key, item)

        #Class attributes
        self.dt = self.tf/(float(self.H))

        #Observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        #Action space
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

        #Reward range
        self.reward_range = (-float('inf'), 0.)

        #Maximum episode steps
        self.max_episode_steps = self.H
    

    def get_observation(self,
                        state: Optional[Any]=None,
                        control: Optional[Any]=None) -> np.ndarray:
        """
        Get current observation: height and velocity
        """

        observation = np.array([state["h"], state["v"], state["m"], state["t"]], dtype=np.float32)

        return observation
    

    def get_control(self,
                    action: Any,
                    state: Optional[Any]=None) -> float:
        """
        Get current control: thrust value
        """

        control = 0.5*(action[0] + 1.)*self.Tmax

        return control


    def next_state(self,
                   state: Optional[Any], 
                   control: Any) -> Dict[str, float]:
        """
        Propagate state: integration of system dynamics
        """

        #State at current time-step
        s = np.array([state["h"], state["v"], state["m"]])

        #Integration of equations of motion
        data = np.array([self.g, control, self.c])

        # Integration
        t_eval = np.array([state['t'], state['t'] + self.dt])
        sol = rk4(dynamics, s, t_eval, data)

        #State at next time-step
        self.success = True
        s_new = {"h": sol[0], "v": sol[1], "m": sol[2], "t": t_eval[-1], "step": state["step"] + 1}
        
        return s_new
    

    def collect_reward(self,
                       prev_state: Optional[Any]=None, 
                       state: Optional[Any]=None, 
                       control: Optional[Any]=None) -> Tuple[float, bool]:
        """
        Get current reward and done signal.
        """

        done = False

        reward = state["m"] - prev_state["m"]

        if state["step"] == self.H:
            done = True
        if not self.success:
            done = True
        if state["h"] <= 0. or state["m"] <= 0.:
            done = True
        
        if done:
            cstr_viol = max(abs(state["h"] - self.hf), abs(state["v"] - self.vf) - 0.005)
            state["cstr_viol"] = max(cstr_viol, 0.)

            reward = reward - 10.*cstr_viol
        
        return reward, done
    

    def get_info(self,
                 prev_state: Optional[Any]=None,
                 state: Optional[Any]=None,
                 observation: Optional[Any]=None,
                 control: Optional[Any]=None,
                 reward: Optional[float]=None,
                 done: Optional[bool]=None) -> Dict[str, float]:
        """
        Get current info.
        """

        info = {}
        info["episode_step_data"] = {}
        info["episode_step_data"]["h"] = [prev_state["h"]] 
        info["episode_step_data"]["v"] = [prev_state["v"]] 
        info["episode_step_data"]["m"] = [prev_state["m"]] 
        info["episode_step_data"]["t"] = [prev_state["t"]] 
        info["episode_step_data"]["T"] = [control]
        if done:
            info["custom_metrics"] = {}
            info["episode_step_data"]["h"].append(state["h"]) 
            info["episode_step_data"]["v"].append(state["v"]) 
            info["episode_step_data"]["m"].append(state["m"]) 
            info["episode_step_data"]["t"].append(state["t"]) 
            info["episode_step_data"]["T"].append(control)
            info["custom_metrics"]["cstr_viol"] = state["cstr_viol"]

        return info
    

    def step(self, action):
        """
        Step of the MDP.

        Args:
            action: current action
        """

        # Invalid action
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Previous state
        self.prev_state = self.state

        # Get control
        control = self.get_control(action, self.prev_state)

        # Next state
        self.state = self.next_state(self.prev_state, control)

        # Get observation
        observation = self.get_observation(self.state, control)

        # Get reward and done signal
        reward, done = self.collect_reward(self.prev_state, self.state, control)

        # Compute infos
        info = self.get_info(self.prev_state, self.state, observation, control, reward, done)

        return observation, float(reward), done, info
    

    def reset(self) -> np.ndarray:
        """ 
        Reset the environment
        """

        self.state = {}
        self.state["h"] = uniform(self.h0_min, self.h0_max)
        self.state["v"] = uniform(self.v0_min, self.v0_max)
        self.state["m"] = self.m0
        self.state["t"] = 0.
        self.state["step"] = 0

        control = 0.

        observation = self.get_observation(self.state, control)

        return observation
    

    def render(self, mode='human'):
        pass


    
