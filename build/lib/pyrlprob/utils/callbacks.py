from typing import *
from collections.abc import Iterable

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
            if "custom_metrics" in info:
                for key, item in info["custom_metrics"].items():
                    episode.custom_metrics[key] = item


class epsConstraintCallbacks(TrainingCallbacks):
    """ 
    epsConstraintCallbacks class. Contains 
    the on_train_results callbacks used to decrease the
    constraint satisfaction tolerance during training
    """

    def on_train_result(self, 
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


    def on_episode_step(self, 
                        *, 
                        worker: RolloutWorker, 
                        base_env: BaseEnv,
                        episode: MultiAgentEpisode, 
                        env_index: int, 
                        **kwargs) -> None:
        """
        Runs on each episode step.

        Args: -> check DefaultCallbacks class
        """

        #Info and done returned by the episode
        info = episode.last_info_for()

        if info is not None:
            if "episode_step_data" in info:
                for key, item in info["episode_step_data"].items():
                    if isinstance(item, Iterable):
                        if episode.length == 1:
                            episode.user_data[key] = list(item)
                        else:
                            episode.user_data[key] = episode.user_data[key] + list(item)
                    else:
                        if episode.length == 1:
                            episode.user_data[key] = [item]
                        else:
                            episode.user_data[key].append(item)
    

    def on_episode_end(self, 
                       *, worker: RolloutWorker, 
                       base_env: BaseEnv,
                       policies: Dict[str, Policy], 
                       episode: MultiAgentEpisode,
                       env_index: int, 
                       **kwargs) -> None:
        """
        Runs when an episode is done.

        Args: -> check DefaultCallbacks class
        """

        #Info and done returned by the episode
        info = episode.last_info_for()

        if info is not None:
            if "episode_step_data" in info:
                for key in info["episode_step_data"].keys():
                    episode.hist_data[key] = episode.user_data[key]
                    episode.hist_data[key + "_length"] = [len(episode.user_data[key])]
            if "episode_end_data" in info:
                for key, item in info["episode_end_data"].items():
                    if isinstance(item, Iterable):
                        episode.hist_data[key] = [item[-1]]
                    else:
                        episode.hist_data[key] = [item]
            if "custom_metrics" in info:
                for key, item in info["custom_metrics"].items():
                    if isinstance(item, Iterable):
                        episode.hist_data[key] = [item[-1]]
                        episode.custom_metrics[key] = item[-1]
                    else:
                        episode.hist_data[key] = [item]
                        episode.custom_metrics[key] = item


