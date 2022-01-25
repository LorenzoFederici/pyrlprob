from pyrlprob.problem import RLProblem
from pyrlprob.tests.landing1d import *

from typing import *
import matplotlib
import matplotlib.pyplot as plt
import yaml
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
    matplotlib.rc('font', size=20)
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('text.latex', preamble='\\usepackage{amsmath} \\usepackage{bm}')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    #For every env, do:
    for e, env in enumerate(envs):

        #Create res_dir
        res_dir_env = res_dir + env + "/"
        os.system("rm -r " + res_dir_env)
        os.makedirs(res_dir_env, exist_ok=True)

        #Create figure
        fig = plt.figure()
        fig.set_size_inches(9.7,6.4)

        #For every alg, do:
        for a, alg in enumerate(algs[e]):
            
            #Config file
            config = os.path.join(__location__, "tuned_examples/" + env + "_" + alg + ".yaml")

            #Problem definition
            # config["stop"]["training_iteration"] *= 2
            Problem = RLProblem(config)

            #Training
            trainer_dir, exp_dirs, last_cps, _ = \
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
            ax.tick_params(labelsize=20)
            ax.yaxis.grid(True, which='major', linewidth=0.5, linestyle='dashed')
            ax.xaxis.grid(True, which='major', linewidth=0.5, linestyle='dashed')
            #plt.xlabel('training iteration', fontsize=20)
            #plt.ylabel('episode reward', fontsize=20)

        #Save figure
        ax.legend(fontsize=20, loc = 'best')
        fig.savefig(res_dir_env + env + "_reward.png", dpi=300)


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))



