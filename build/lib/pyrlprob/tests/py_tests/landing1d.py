import numpy as np
from numpy.random import uniform
from typing import *

import gym
from gym import spaces

from pyrlprob.mdp import AbstractMDP

from pyrlprob.tests.py_tests.landing1d_gym import *
from pyrlprob.tests.py_tests.landing1d_dyn import *


class pyLanding1DEnv(AbstractMDP):
    """
    One-Dimensional Landing Problem.
    Reference: https://doi.org/10.2514/6.2008-6615
    """

    def __init__(self, config) -> None:
        """
        Definition of observation and action spaces
        """

        if (not config) or (config is None):
            config = DEFAULT_CONFIG_LANDING1D

        super().__init__(config=config)

        #Time step
        self.time_step = self.tf/(float(self.H))

        #Maximum episode steps
        self.max_episode_steps = self.H

        #Observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        #Action space
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

        #Reward range
        self.reward_range = (-float('inf'), 0.)


    def get_observation(self,
                        state,
                        control) -> np.ndarray:
        """
        Get current observation: height and velocity
        """

        observation = np.array([state["h"], state["v"], state["m"], state["t"]], dtype=np.float32)

        return observation
    

    def get_control(self,
                    action,
                    state) -> float:
        """
        Get current control: thrust value
        """

        control = 0.5*(action[0] + 1.)*self.Tmax

        return control


    def next_state(self,
                   state: Optional[Any], 
                   control: Any,
                   time_step: float) -> Dict[str, float]:
        """
        Propagate state: integration of system dynamics
        """

        #State at current time-step
        s = np.array([state["h"], state["v"], state["m"]])

        #Integration of equations of motion
        data = np.array([self.g, control, self.c])

        # Integration
        t_eval = np.array([state['t'], state['t'] + time_step])
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
                 prev_state,
                 state,
                 observation,
                 control,
                 reward,
                 done) -> Dict[str, float]:
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


    
