from pyrlprob.problem import RLProblem
from pyrlprob.tests.landing1d import Landing1DEnv

from typing import *
import matplotlib
import matplotlib.pyplot as plt
import yaml
import os

from pyrlprob.utils.plots import plot_metric


def test_landing_env_train(res_dir: Optional[str]=None) -> None:
    """
    Test pyrlprob training functionalities in the Landing1D environment.

    Args:
        res_dir: path where results are saved. Current directory if not specified.
    """

    #Config file
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    config = os.path.join(__location__, "landing1d.yaml")

    #Problem definition
    LandingProblem = RLProblem(config)

    #Training
    trainer_dir, exp_dirs, last_cps, _ = \
        LandingProblem.solve(res_dir, 
                             evaluate=False, 
                             postprocess=False)

    #Plot of metric trend
    plt.style.use("seaborn")
    fig = plot_metric("episode_reward",
                      exp_dirs,
                      last_cps)
    plt.xlabel('training iteration', fontsize=20)
    plt.ylabel('episode reward', fontsize=20)
    fig.savefig(trainer_dir + "episode_reward.png")


def test_landing_env_train_eval(res_dir: Optional[str]=None) -> None:
    """
    Test pyrlprob training, evaluation and post-processing functionalities 
        in the Landing1D environment.

    Args:
        res_dir: path where results are saved. Current directory if not specified.
    """

    #Config file
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    config = os.path.join(__location__, "landing1d.yaml")

    #Problem definition
    LandingProblem = RLProblem(config)

    #Training
    trainer_dir, exp_dirs, last_cps, _ = \
        LandingProblem.solve(res_dir, 
                             evaluate=False, 
                             postprocess=False)

    #Create new config file for model re-training
    load = {"load": {"trainer_dir": trainer_dir,
            "prev_exp_dirs": exp_dirs,
            "prev_last_cps": last_cps}}
    stop = {"stop": {"training_iteration": 2*last_cps[-1]}}
    config_new_dict = {**LandingProblem.input_config, **stop, **load}
    with open(os.path.join(__location__, "landing1d_load.yaml"), 'w') as outfile:
        yaml.dump(config_new_dict, outfile)
    
    #Training, evaluation and postprocessing of pre-trained model
    config_new = os.path.join(__location__, "landing1d_load.yaml")
    LandingProblemPretrained = RLProblem(config_new)
    trainer_dir, exp_dirs, last_cps, _ = \
        LandingProblemPretrained.solve()

    #Plot of metric trend
    plt.style.use("seaborn")
    fig = plot_metric("episode_reward",
                      exp_dirs,
                      last_cps)
    plt.xlabel('training iteration', fontsize=20)
    plt.ylabel('episode reward', fontsize=20)
    fig.savefig(trainer_dir + "episode_reward.png")


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))



