import os
from sympy import divisors
import yaml
import ray
from pyrlprob.problem import RLProblem

def tune_workers_envs(config_file, cpus, gpus, envs, min_w, max_w):
    
    ray.init(logging_level="ERROR", log_to_driver=False)

    config = yaml.safe_load(open(config_file))

    # Tests
    if gpus > 0:
        hardware = ["cpu_only", "gpu_d_cpu_w"]
    else:
        hardware = ["cpu_only"]

    # Output file
    os.makedirs("./tuning", exist_ok=True)
    f_log = open("./tuning/cpu_times.txt", "w")
    f_log.write("%20s %20s %20s %20s %20s %20s %20s %20s %20s %20s\n" \
        % ("# hardware", "workers", "envs_per_worker", \
        "eval_workers", "eval_epis", "cpus_per_w", "gpus_per_w", "cpus_per_d", "gpus_per_d", "time[s]"))

    # Run simulation
    for h in hardware:
        for w in divisors(envs):
            
            if w > max_w or w < min_w:
                continue
            
            eval_w = max(1, int(w/2))
            eval_epis = "auto"
            total_w = w + eval_w

            if h == "cpu_only":
                cpus_per_w = (cpus - 1.)/total_w if (cpus - 1.)/total_w < 1 else int((cpus - 1.)/total_w)
                cpus_per_d = int(cpus - cpus_per_w*total_w)
                gpus_per_w = 0
                gpus_per_d = 0
            elif h == "gpu_only":
                cpus_per_w = 0
                cpus_per_d = 0
                gpus_per_w = gpus/(total_w + 1)
                gpus_per_d = gpus_per_w
            elif h == "gpu_d_cpu_w":
                cpus_per_w = cpus/total_w if cpus/total_w < 1 else int(cpus/total_w)
                cpus_per_d = int(cpus - cpus_per_w*total_w)
                gpus_per_w = 0
                gpus_per_d = gpus
            elif h == "gpu_w_cpu_d":
                cpus_per_w = 0
                cpus_per_d = cpus
                gpus_per_w = gpus/total_w
                gpus_per_d = 0
            
            # Training config
            config["stop"]["training_iteration"] = 1
            config["config"]["num_rollout_workers"] = w
            config["config"]["num_envs_per_worker"] = int(envs / w)
            config["config"]["num_cpus_per_worker"] = cpus_per_w
            config["config"]["num_cpus_for_local_worker"] = cpus_per_d
            config["config"]["num_gpus_per_worker"] = gpus_per_w
            config["config"]["num_gpus"] = gpus_per_d
            config["config"]["create_env_on_local_worker"] = False

            #Evaluation config
            config["config"]["evaluation_parallel_to_training"] = True
            config["config"]["evaluation_interval"] = 1
            config["config"]["evaluation_duration_unit"] = "episodes"
            config["config"]["evaluation_duration"] = eval_epis
            config["config"]["evaluation_num_workers"] = eval_w
            config["config"]["evaluation_config"]["explore"] = False

            # Define RL problem
            Prb = RLProblem(config)

            # Solve RL problem
            best_results, trainer_dir, exp_dirs, last_cps, best_cp_dir, run_time = \
                    Prb.solve(evaluate=False, postprocess=False, debug=False, 
                        open_ray=False, return_time=True)
            
            # Print results
            f_log.write("%20s %20d %20d %20d %20s %20.5f %20.5f %20.5f %20.5f %20.5f\n" \
                % (h, w, int(envs / w), eval_w, eval_epis, cpus_per_w, gpus_per_w, cpus_per_d, gpus_per_d, run_time))
            
            print("Done case: w = %d, cpu_per_w = %4.3f" % (w, cpus_per_w))
        
    f_log.close()

    ray.shutdown()

    return