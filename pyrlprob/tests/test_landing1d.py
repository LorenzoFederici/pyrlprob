from pyrlprob.problem import RLProblem
from pyrlprob.tests.landing1d import Landing1DEnv

import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
import yaml

from pyrlprob.utils.plots import plot_metric


def test_landing_env():

    #Results directory
    res_dir = "pyrlprob/tests/results/"
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)

    #Config file
    config = "pyrlprob/tests/landing1d.yaml"

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
    with open("pyrlprob/tests/landing1d_load.yaml", 'w') as outfile:
        yaml.dump(config_new_dict, outfile)
    
    #Training, evaluation and postprocessing of pre-trained model
    config_new = "pyrlprob/tests/landing1d_load.yaml"
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



