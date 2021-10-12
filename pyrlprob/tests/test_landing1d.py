from pyrlprob.problem import RLProblem
from pyrlprob.tests.landing1d import Landing1DEnv

import os
import shutil

def test_landing_env():

    #Results directory
    res_dir = "pyrlprob/tests/results/"
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)

    #Config file
    config = "pyrlprob/tests/landing1d.yaml"

    #Problem definition
    LandingProblem = RLProblem(config)

    #Training, evaluation and postprocessing
    _, _, _ = LandingProblem.solve(res_dir, graphs=True)


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))



