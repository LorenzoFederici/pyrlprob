from typing import *

import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy


class TrainingCallbacks(DefaultCallbacks):
    """ 
    TrainingCallbacksCallbacks class. Contains 
    the definition of callbacks used during model training 
    """

    def __init__(self, 
                 custom_metrics: List[str]) -> None:
        """
        Class constructor

        Args:
            custom_metrics (list): metrics saved as custom_metrics at the end
                of each episode
        """
        
        super().__init__()
        self.custom_metrics = custom_metrics


    def on_episode_end(self, 
                       *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, 
                       **kwargs) -> None:
        """
        Runs when an episode is done.

        Args: -> check DefaultCallbacks class
        """

        #Info returned by the episode
        info = episode.last_info_for()
        if info is not None:
            for metric in self.custom_metrics:
                episode.custom_metrics[metric] = info[metric]


class epsConstraintCallbacks(TrainingCallbacks):
    """ 
    epsConstraintCallbacks class. Contains 
    the on_train_results callbacks used to decrease the
    constraint satisfaction tolerance during training
    """

    def on_train_results(self, 
                         *, 
                         trainer, 
                         result: dict, 
                         **kwargs) -> None:
        """
        Called at the end of Trainable.train().

        Args: -> check DefaultCallbacks class
        """

        training_iter = result["training_iteration"]
        
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_cstr_tolerance(training_iter)))


class EvaluationCallbacks(DefaultCallbacks):
    """ 
    EvaluationCallbacksCallbacks class. Contains 
    the definition of callbacks used during model evaluation 
    """

    def __init__(self, 
                 episode_step_data: List[str],
                 episode_end_data: List[str], 
                 custom_metrics: List[str]) -> None:
        """
        Class constructor

        Args:
            episode_step_data (list): data saved as hist_stats collected at each step
                of the episode
            episode_end_data (list): data saved as hist_stats collected just at the end 
                of the episode
            custom_metrics (list): metrics saved as custom_metrics at the end
                of each episode
        """
        
        super().__init__()
        self.episode_step_data = episode_step_data
        self.episode_end_data = episode_end_data
        self.custom_metrics = custom_metrics
    

    def on_episode_start(self, 
                         *, worker: RolloutWorker, 
                         base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, 
                         env_index: int, 
                         **kwargs) -> None:
        """
        Callback run on the rollout worker before each episode starts.

        Args: -> check DefaultCallbacks class
        """

        for metric in self.episode_step_data:
            episode.user_data[metric] = []
            episode.hist_data[metric] = []
        for metric in self.episode_end_data:
            episode.hist_data[metric] = []
        for metric in self.custom_metrics:
            episode.hist_data[metric] = []


    def on_episode_step(self, 
                        *, 
                        worker: RolloutWorker, 
                        base_env: BaseEnv,
                        policies: Dict[str, Policy],
                        episode: MultiAgentEpisode, 
                        env_index: int, 
                        **kwargs) -> None:
        """
        Runs on each episode step.

        Args: -> check DefaultCallbacks class
        """

        #Info and done returned by the episode
        info = episode.last_info_for()
        done = episode.last_done_for()

        if info is not None:
            for metric in self.episode_step_data:
                if not done:
                    episode.user_data[metric].append(info[metric])
                else:
                    episode.user_data[metric].append(info[metric][0])
                    episode.user_data[metric].append(info[metric][1])
    

    def on_episode_end(self, 
                       *, worker: RolloutWorker, 
                       base_env: BaseEnv,
                       policies: Dict[str, Policy], 
                       episode: MultiAgentEpisode,
                       env_index: int, **kwargs) -> None:
        """
        Runs when an episode is done.

        Args: -> check DefaultCallbacks class
        """

        #Info and done returned by the episode
        info = episode.last_info_for()

        if info is not None:
            for metric in self.episode_step_data:
                episode.hist_data[metric] = episode.user_data[metric]
            for metric in self.episode_end_data:
                episode.hist_data[metric] = [info[metric]]
            for metric in self.custom_metrics:
                episode.hist_data[metric] = [info[metric]]
                episode.custom_metrics[metric] = info[metric]


