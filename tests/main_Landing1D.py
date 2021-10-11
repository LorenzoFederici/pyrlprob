from RLproblem import RLProblem
from tests.Landing1D import Landing1DEnv

if __name__ == "__main__":

    #Config file
    config = "tests/Landing1D.yaml"

    #Problem definition
    LandingProblem = RLProblem(config)

    #Training, evaluation and postprocessing
    _, _, _ = LandingProblem.solve()



