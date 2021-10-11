import numpy as np
from typing import *

import gym
from gym import spaces

from mdp import AbstractMDP

import scipy
from scipy.integrate import solve_ivp


class Landing1DEnv(AbstractMDP):
    """
    One-Dimensional Landing Problem.
    Reference: https://doi.org/10.2514/6.2008-6615
    """

    def __init__(self, config) -> None:
        """
        Definition of observation and action spaces
        """
        super().__init__(config=config)

        #Class attributes
        self.dt = self.tf/(float(self.H))

        #Observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        #Action space
        self.action_space = spaces.Box(low=-1., high=1., dtype=np.float64)

        #Reward range
        self.reward_range = (-float('inf'), 0.)
    

    def get_observation(self,
                        state,
                        control) -> np.ndarray:
        """
        Get current observation: height and velocity
        """

        observation = np.array([state["h"], state["v"], state["m"], state["t"]], dtype=np.float64)

        return observation
    

    def get_control(self,
                    action,
                    state) -> float:
        """
        Get current control: thrust value
        """

        control = 0.5*(action + 1.)*self.Tmax

        return control


    def dynamics(self, 
                 t, 
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

        s_dot = np.array([h_dot, v_dot, m_dot], dtype=np.float64)


    def next_state(self,
                   state, 
                   control) -> Dict[str, float]:
        """
        Propagate state: integration of system dynamics
        """

        #State at current time-step
        s = np.array([state["h"], state["v"], state["m"]], dtype=np.float64)

        #Integration of equations of motion
        sol = solve_ivp(fun=self.dynamics, t_span=[state["t"], state["t"]+self.dt], y0=s, method='RK45', 
            args=(self.g, control, self.c), rtol=1e-6, atol=1e-6)

        #State at next time-step
        s_new = {"h": sol.y[0][-1], "v": sol.y[1][-1], "m": sol.y[2][-1], "t": sol.t[-1]}
        
        return s_new
    

    def collect_reward(self,
                       prev_state, 
                       state, 
                       control) -> Tuple[float, bool]:
        """
        Get current reward and done signal.
        """

        reward = state["m"] - prev_state["m"]

        if state["t"] == self.tf:
            done = True
        if state["h"] <= 0:
            done = True
        if state["m"] <= 0:
            done = True
        
        if done:
            cstr_viol = max(abs(state["h"] - self.hf), abs(state["v"] - self.vf))
            state["cstr_viol"] = cstr_viol

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

        info = prev_state
        info["T"] = control

        if done:
            info["h"] = [prev_state["h"], state["h"]]
            info["v"] = [prev_state["v"], state["v"]]
            info["m"] = [prev_state["m"], state["m"]]
            info["t"] = [prev_state["t"], state["t"]]
            info["T"] = [control, control]

        return info
    

    def reset(self) -> np.ndarray:
        """ 
        Reset the environment
        """

        self.state = {}
        self.state["h"] = self.h0
        self.state["v"] = self.v0
        self.state["m"] = self.m0
        self.state["t"] = 0.

        control = 0.

        observation = self.get_observation(self.state, control)

        return observation


    def render(self, mode='human'):
        pass



    
