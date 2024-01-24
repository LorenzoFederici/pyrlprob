#Finds the best combination of cpu/gpu to set in the config file
import argparse
import ray
from pyrlprob.tune import tune_workers_envs

def main():

    # Input config file and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml", \
        help='Input file with algorithm, model and environment config')
    parser.add_argument('--cpus', type=float, default=32.0, \
        help='Number of CPUs available')
    parser.add_argument('--gpus', type=int, default=1, \
        help='Number of GPUs available')
    parser.add_argument('--envs', type=int, default=100, \
        help='Number of environments')
    parser.add_argument('--min_w', type=int, default=1, \
        help='Minimum number of workers')
    parser.add_argument('--max_w', type=int, default=50, \
        help='Maximum number of workers')
    args = parser.parse_args()
    config_file = args.config
    cpus = args.cpus
    gpus = args.gpus
    envs = args.envs
    min_w = args.min_w
    max_w = args.max_w

    tune_workers_envs(config_file, cpus, gpus, envs, min_w, max_w)


if __name__ == "__main__":
    main()