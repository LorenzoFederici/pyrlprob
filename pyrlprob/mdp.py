import gym

import numpy as np
from typing import *

from pyrlprob.utils.auxiliary import *

class AbstractMDP(gym.Env):
    """
    Abstract Markov Decision Process, based on OpenAI Gym.
    """

    def __init__(self, 
                 config = Dict[str, Any]) -> None:
        """ 
        Class constructor 
        
        Args:
            config (dict): environment configs
        """

        super().__init__()

        #Create class attributes
        for key, item in config.items():
            setattr(self, key, item)
        
        self.epsilon = 0.
        self.epsilon0 = 0.
        self.epsilonf = 0.
        self.iter0 = 0
        self.iterf = 1

        if hasattr(self, 'prng_seed'):
            seeds = self.seed(self.prng_seed)
        else:
            seeds = self.seed()


    def seed(self,
             seed: Optional[int]=None) -> List[int]:
        """
        Set global seeds.

        Args:
            seed (int): seed
        """

        seeds = set_global_seeds(seed)

        return seeds
    

    def get_observation(self,
                        state: Optional[Any]=None,
                        control: Optional[Any]=None) -> Any:
        """
        Get current observation.

        Args:
            state (Any): current system state
            control (Any): current control action
        
        Return:
            observation (Any): current observation

        """

        raise NotImplementedError(
            "The Environment must implement the get_observation function!")
    

    def get_control(self,
                    action: Any,
                    state: Optional[Any]=None) -> Any:
        """
        Get current control.

        Args:
            action (Any): current action
            state (Any): current system state
        
        Return:
            control (Any): current control

        """

        raise NotImplementedError(
            "The Environment must implement the get_control function!")
    

    def next_state(self,
                   state: Optional[Any], 
                   control: Any) -> Any:
        """
        Propagate state

        Args:
            state (Any): current system state
            control (Any): current control
        
        Return:
            next_state (Any): next system state

        """

        raise NotImplementedError(
            "The Environment must implement the next_state function!")
    

    def collect_reward(self,
                       prev_state: Optional[Any]=None, 
                       state: Optional[Any]=None, 
                       control: Optional[Any]=None) -> Tuple[float, bool]:
        """
        Get current reward and done signal.

        Args:
            prev_state (Any): previous system state
            control (Any): current control
            state (Any): current system state
            
        Return:
            reward (float): current reward
            done (bool): is episode done?

        """

        raise NotImplementedError(
            "The Environment must implement the collect_reward function!")
    

    def get_info(self,
                 prev_state: Optional[Any]=None,
                 state: Optional[Any]=None,
                 observation: Optional[Any]=None,
                 control: Optional[Any]=None,
                 reward: Optional[float]=None,
                 done: Optional[bool]=None) -> Dict[str, Any]:
        """
        Get current info.

        Args:
            prev_state (Any): previous system state
            state (Any): current system state
            observation (Any): current observation
            control (Any): current control
            reward (float): last reward
            done (bool): last done signal
            
        Return:
            info (dict): current info

        """

        raise NotImplementedError(
            "The Environment must implement the get_info function!")
    

    def set_cstr_tolerance(self,
                           iter: int) -> None:
        """
        Set current constraint satisfation tolerance epsilon.

        Args:
            iter (int): current training iteration
        """

        if iter <= self.iter0:
            self.epsilon = self.epsilon0
        elif iter >= self.iterf:
            self.epsilon = self.epsilonf
        else:
            self.epsilon = \
                self.epsilon0*(self.epsilonf/self.epsilon0)**(float(iter - \
                self.iter0)/float(self.iterf - self.iter0))
    

    def step(self, action):
        """
        Step of the MDP.

        Args:
            action: current action
        """

        # Invalid action
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Get control
        control = self.get_control(action, self.state)

        # Next state
        self.prev_state = self.state
        self.state = self.next_state(self.prev_state, control)

        # Get observation
        observation = self.get_observation(self.state, control)

        # Get reward and done signal
        reward, done = self.collect_reward(self.prev_state, self.state, control)

        # Compute infos
        info = self.get_info(self.prev_state, self.state, observation, control, reward, done)

        return observation, float(reward), done, info





    