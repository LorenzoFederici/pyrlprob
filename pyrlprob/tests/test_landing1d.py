from pyrlprob.problem import RLProblem
from pyrlprob.tests.landing1d import Landing1DEnv

def test_landing_env():

    #Config file
    config = "pyrlprob/tests/landing1d.yaml"

    #Problem definition
    LandingProblem = RLProblem(config)

    #Training, evaluation and postprocessing
    _, _, _ = LandingProblem.solve()


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))



