from pyrlprob.problem import RLProblem
from pyrlprob.tests import *

from typing import *
import matplotlib
import matplotlib.pyplot as plt
import os

from pyrlprob.utils.plots import plot_metric


def test_multiple_gym_envs(envs: List[str]=["CartPole-v1"], 
                           algs: List[List[str]]=[["PPO"]], 
                           res_dir: Optional[str]=None) -> None:
    """
    Compare multiple RL algorithms in different Gym environments through pyrlprob.

    Args:
        envs: list of Gym environments' names, among those in tuned_examples/
        algs: list of RL algorithms names, among those in tuned_examples/
        res_dir: path where results are saved. Current directory if not specified.
    """

    #Current location
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    #Results dir
    if res_dir is None:
        res_dir = "./results/"

    #Plot style
    plt.style.use("seaborn")
    matplotlib.rc('font', size=24)
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    #For every env, do:
    for e, env in enumerate(envs):

        #Create res_dir
        res_dir_env = res_dir + env + "/"
        res_dir_plot = res_dir + "plots/"
        os.makedirs(res_dir_env, exist_ok=True)

        #Create figure
        fig = plt.figure()
        fig.set_size_inches(10.4,6.4)

        #For every alg, do:
        for a, alg in enumerate(algs[e]):
            
            #Config file
            config = os.path.join(__location__, "py_tests/tuned_examples/" + env + "_" + alg + ".yaml")

            #Problem definition
            Problem = RLProblem(config)

            #Training
            _, exp_dirs, last_cps, _ = \
                Problem.solve(res_dir_env, 
                              evaluate=False, 
                              postprocess=False)

            #Plot of metric trend
            fig = plot_metric("episode_reward",
                            exp_dirs,
                            last_cps,
                            fig=fig,
                            label=alg,
                            color=palette[a])
            ax = fig.gca()
            ax.tick_params(labelsize=24)
            ax.yaxis.grid(True, which='major', linewidth=0.5, linestyle='dashed')
            ax.xaxis.grid(True, which='major', linewidth=0.5, linestyle='dashed')
            plt.xlabel('Training iteration', fontsize=24)
            plt.ylabel('Episode return', fontsize=24)

        #Save figure
        ax.legend(fontsize=24, bbox_to_anchor=(0,1.01,1,0.2), loc="lower left", mode="expand", ncol=5)
        plt.tight_layout(pad=0)
        fig.savefig(res_dir_plot + env + "_reward.pdf", dpi=300)


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))



