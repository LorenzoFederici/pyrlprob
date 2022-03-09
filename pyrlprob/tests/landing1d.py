import numpy as np
from numpy.random import uniform
from typing import *

import gym
from gym import spaces

import ray
from ray.tune.registry import register_env

from pyrlprob.mdp import AbstractMDP

import scipy
from scipy.integrate import solve_ivp


#Default config for Landing1DEnv
DEFAULT_CONFIG_LANDING1D = {
    "H": 20,
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


def dynamics(t, 
             s,
             g, 
             T, 
             c) -> np.ndarray:

    """
    System dynamics: vertical landing on planetary body
        with constant gravity g and thrust T
    """
    #State
    h = s[0]
    v = s[1]
    m = s[2]

    #Equations of motion
    h_dot = v
    v_dot = - g + T/m
    m_dot = - T/c

    s_dot = np.array([h_dot, v_dot, m_dot], dtype=np.float32)

    return s_dot


def landed_or_empty_event(t, 
                          s,
                          g, 
                          T, 
                          c) -> float:
    """
    Event for dynamics: it stops the integration if h < 0 (landed) or m < 0 (empy prop tanks)
    """

    #State
    h = s[0]
    m = s[2]

    return min(h, m)


class Landing1DEnv(AbstractMDP):
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

        #Class attributes
        self.time_step = self.tf/(float(self.H))

        #Observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        #Action space
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

        #Reward range
        self.reward_range = (-float('inf'), 0.)

        #Maximum episode steps
        self.max_episode_steps = self.H
    

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
                   state, 
                   control,
                   time_step) -> Dict[str, float]:
        """
        Propagate state: integration of system dynamics
        """

        #State at current time-step
        s = np.array([state["h"], state["v"], state["m"]], dtype=np.float32)

        #Integration of equations of motion
        landed_or_empty_event.terminal = True
        sol = solve_ivp(fun=dynamics, t_span=[state["t"], state["t"]+time_step], y0=s, method='RK45', 
            args=(self.g, control, self.c), events=landed_or_empty_event, rtol=1e-6, atol=1e-6)

        #State at next time-step
        self.event = sol.status
        s_new = {"h": sol.y[0][-1], "v": sol.y[1][-1], "m": sol.y[2][-1], "t": sol.t[-1], "step": state["step"] + 1}
        
        return s_new
    

    def collect_reward(self,
                       prev_state, 
                       state, 
                       control) -> Tuple[float, bool]:
        """
        Get current reward and done signal.
        """

        done = False

        reward = state["m"] - prev_state["m"]

        if state["step"] == self.H:
            done = True
        if self.event == 1:
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

#Register environment
def landing_env_creator(env_config):
    return Landing1DEnv(env_config)

register_env("Landing1D", landing_env_creator)

    
