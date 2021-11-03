import numpy as np
import time
from typing import *
from itertools import islice
import importlib
import os

import ray
from ray import tune

from pyrlprob.utils.auxiliary import *
import pyrlprob.utils.callbacks as callbacks


def training(trainer: Union[str, Callable, Type], 
             config: Dict[str, Any], 
             stop: Dict[str, Any], 
             logdir: Optional[str]=None,
             create_out_file: bool=True, 
             load: Optional[Dict[str, Any]]=None, 
             debug: bool=False) -> Tuple[str, str, int]:
    """
    Train the current model with ray.tune, using the specified trainer and configs.

    Args:
        trainer (str or callable): trainer (i.e., RL algorithm) to train the model with
        config (dict): config file (dictionary)
        stop (dict): stopping conditions (dictionary)
        logdir (str): name of the directory where training results are saved
        create_out_file (bool): whether to create an outfile with run time and best result
        load (dict): dictionary containing the directory and checkpoint where the 
            pre-trained model to load is located
        debug (bool): whether to print worker's logs.
    
    Return:
        trainer_dir (str): trainer's output directory
        best_exp_dir (str): directory containing the best experiment's output
        last_checkpoint (int): last checkpoint saved
    """

    #Create output folder
    if load is not None:
        outdir = load["logdir"]
        restore = load["checkpoint_dir"]
    else:
        if logdir is None:
            outdir = "./results/"
        else:
            outdir = logdir
        os.makedirs(outdir, exist_ok=True)
        restore = None

    #Initialize ray
    ray.init(log_to_driver=debug)
    
    #Train the model
    start_time = time.time()
    analysis = tune.run(trainer,
                        config=config,
                        local_dir=outdir,
                        restore=restore,
                        stop=stop,
                        metric="training_iteration",
                        mode="max",
                        checkpoint_freq=1,
                        checkpoint_at_end=True,
                        keep_checkpoints_num=None)
    end_time = time.time()

    #Last iteration stats
    last_result = analysis.best_result
    best_exp_dir = analysis.best_logdir
    trainer_dir = best_exp_dir[:best_exp_dir.rfind("/")+1]
    best_exp_dir = best_exp_dir + "/"

    #Last checkpoint
    last_checkpoint = last_result["training_iteration"]

    # Save elapsed time and results
    if create_out_file:
        f_out_res = open(best_exp_dir + "result.txt", "w")
        f_out_res.write("%22s %22s %22s %22s %22s\n" \
            % ("# elapsed time [s]", "training_iteration", \
            "episode_reward_mean", "episode_reward_max", "episode_reward_min"))
        f_out_res.write("%22.7f %22d %22.7f %22.7f %22.7f\n" \
            % (end_time - start_time, last_result["training_iteration"], \
                last_result["episode_reward_mean"], \
                last_result["episode_reward_max"], \
                last_result["episode_reward_min"]))
        f_out_res.close()

    #Terminate ray
    ray.shutdown()

    #Return trainer and best experiment directory
    return trainer_dir, best_exp_dir, last_checkpoint


def evaluation(trainer_dir: str,
               exp_dirs: List[str],
               last_cps: List[int],
               model: Dict[str, Any],
               gamma: float,
               max_ep_length: int, 
               env: Union[Callable, str], 
               env_config: Dict[str, Any],
               evaluation_num_episodes: int, 
               evaluation_config: Dict[str, Any],
               custom_eval_function: Optional[Union[Callable, str]]=None,
               metrics_and_data: Optional[Dict[str, Any]]=None,
               is_evaluation_env: bool=False,
               do_postprocess: bool=True) -> str:
    """
    Evaluate a model, and find the best checkpoint

    Args:
        trainer_dir (str): trainer directory
        exp_dirs (list): list with experiments directories
        last_cps (list): list with last checkpoint number of each experiment in exp_dirs
        model (dict): dict with current model configs
        gamma (float): disconut factor
        max_ep_length (int): maximum episode length
        env (callable or str): Environment class (or class name)
        env_config (dict): dictionary containing the environment configs
        evaluation_num_episodes (int): number of evaluation episodes
        evaluation_config (dict): dictionary containing the evaluation configs
        custom_eval_function (callable or str): Custom evaluation function (or function name)
        metrics_and_data (dict): dictionary containing the metrics and data to save
            in the new file progress.csv
        is_evaluation_env (bool): are metrics computed through an evaluation environment?
        do_postprocess (bool): whether to do postprocessing
    """
    
    #Path of metrics
    metric_path = ""
    if is_evaluation_env:
        metric_path = "evaluation/"

    if trainer_dir is not None:
        #Determine the best checkpoint
        episode_reward = metric_training_trend(metric_path + "episode_reward_mean",
                                            exp_dirs,
                                            last_cps)
        best_cp = np.argmax(episode_reward)+1
        best_exp = next(exp for exp, cp in enumerate(last_cps) if cp >= best_cp)
        best_exp_dir = exp_dirs[best_exp]

        #Define load properties
        load = {}
        load["logdir"] = trainer_dir
        _, load["checkpoint_dir"] = \
            get_cp_dir_and_model(best_exp_dir, best_cp)
    else:
        load = None

    #Check what metrics/data are defined
    if metrics_and_data is None:
        metrics_and_data = {}
    for key in ["episode_step_data", "episode_end_data", "custom_metrics"]:
        if not key in metrics_and_data:
            metrics_and_data[key] = []      
                
    #Define standard PG trainer and configs for evaluation
    trainer = ray.rllib.agents.ppo.PPOTrainer
    config = {}
    config["num_workers"] = 0
    config["num_envs_per_worker"] = 1
    config["create_env_on_driver"] = True
    config["model"] = model
    config["gamma"] = gamma
    config["batch_mode"] = "complete_episodes"
    config["horizon"] = max_ep_length
    config["train_batch_size"] = max_ep_length
    config["sgd_minibatch_size"] = max_ep_length
    config["lr"] = 0.
    if callable(env):
        config["env"] = env
    else:
        mod_name, fun_name = env.rsplit('.',1)
        mod = importlib.import_module(mod_name)
        config["env"] = getattr(mod, fun_name)
    config["env_config"] = env_config
    config["evaluation_interval"] = 1
    config["evaluation_num_episodes"] = evaluation_num_episodes
    config["evaluation_num_workers"] = 0
    config["evaluation_config"] = evaluation_config
    if custom_eval_function is not None:
        if callable(custom_eval_function):
                config["custom_eval_function"] = custom_eval_function
        else:
            mod_name, fun_name = custom_eval_function.rsplit('.',1)
            mod = importlib.import_module(mod_name)
            config["custom_eval_function"] = getattr(mod, fun_name)
    config["callbacks"] = callbacks.EvaluationCallbacks
    stop = {"training_iteration": 1}


    #Evaluation
    _, new_best_exp_dir, last_checkpoint  = training(trainer=trainer, 
                                                     config=config,
                                                     stop=stop,
                                                     load=load)
    
    #Postprocessing
    if do_postprocess:
        cp_dir = postprocessing(best_exp_dir=new_best_exp_dir, 
                                checkpoint=last_checkpoint, 
                                metrics_and_data=metrics_and_data, 
                                is_evaluation_env=is_evaluation_env)
    else:
        cp_dir, _ = get_cp_dir_and_model(new_best_exp_dir, best_cp)
    
    return cp_dir


def postprocessing(best_exp_dir: str,
                   checkpoint: int,
                   metrics_and_data: Optional[Dict[str, Any]]=None,
                   is_evaluation_env: bool=False) -> str:
    """
    Default postprocessing.

    Args:
        exp_dir (str): experiment directory
        checkpoint (int): number of the checkpoint to postprocess
        custom_metrics (list): custom metrics to include in the postprocessing
        metrics_and_data (dict): dictionary containing the data and metrics to postprocess
        is_evaluation_env (bool): are metrics computed through an evaluation environment?
    """

    #Get checkpoint directory
    cp_dir, _ = get_cp_dir_and_model(best_exp_dir, checkpoint)

    #Path of metrics and data
    metric_path = ""
    if is_evaluation_env:
        metric_path = "evaluation/"

    if metrics_and_data is None:
        metrics_and_data = {}
        metrics_and_data["custom_metrics"] = []
        metrics_and_data["episode_step_data"] = []
        metrics_and_data["episode_end_data"] = []

    #Metrics and data to log
    values = ["min", "mean", "max"]
    metrics = {
        "metrics": {"episode_reward": {value: 0. for value in values}},
        "custom_metrics": {metric: {value: 0. for value in values} 
            for metric in metrics_and_data["custom_metrics"]},
        "episode_step_data": {metric: []
            for metric in metrics_and_data["episode_step_data"]}, 
        "episode_end_data": {metric: []
            for metric in metrics_and_data["custom_metrics"] + metrics_and_data["episode_end_data"]}
        }

    #Create output files
    f_log = open(cp_dir + "metrics.txt", "w") # open file
    f_end_data = open(cp_dir + "episode_end_data.txt", "w") # open file
    f_step_data = open(cp_dir + "episode_step_data.txt", "w") # open file

    f_log.write("%20s " % ("# checkpoint"))
    for key, item in metrics.items():
        for key_in in item.keys():
            if key == "episode_step_data":
                f_step_data.write("%20s " % (key_in))
            elif key == "episode_end_data":
                f_end_data.write("%20s " % (key_in))
            else:
                for value in values:
                    f_log.write("%20s " % (key_in + "_" + value))
    f_log.write("\n")
    f_step_data.write("\n")
    f_end_data.write("\n")
    f_log.write("%20d " % (checkpoint))

    #Episode lengths
    ep_length = column_progress(best_exp_dir+"progress.csv", \
                        metric_path + "hist_stats/episode_lengths")
    ep_length = ep_length[0].strip("[").strip("]").split(", ")
    ep_length = np.array(ep_length, dtype=int)

    #Retrieve metrics and data from progress.csv
    for key, item in metrics.items():
        for key_in in item.keys():
            if "episode" in key:
                q = column_progress(best_exp_dir+"progress.csv", metric_path + "hist_stats/" + key_in)
                q = q[0].strip("[").strip("]").split(", ")
                if key == "episode_step_data":
                    start = 0
                    stop = 0
                    for e_num, e in enumerate(ep_length):
                        stop = stop + (e+1)
                        metrics[key][key_in].append(q[start:stop])
                        start = start + (e+1)
                else:
                    metrics[key][key_in] = q
            else:
                if key == "custom_metrics":
                    metric_type = "custom_metrics/"
                else:
                    metric_type = ""
                for value in values:
                    metrics[key][key_in][value] = column_progress(best_exp_dir+"progress.csv", \
                        metric_path + metric_type + key_in + "_" + value)
                    f_log.write("%20.7f " % (metrics[key][key_in][value][-1]))
    f_log.write("\n")
    f_log.close()

    #Print episode_step_data
    for e_num, e in enumerate(ep_length):
        for h in range(e+1):
            for key_in in metrics["episode_step_data"].keys():
                f_step_data.write("%20s " % (metrics["episode_step_data"][key_in][e_num][h]))
            f_step_data.write("\n")
        f_step_data.write("\n\n")
    f_step_data.close()

    #Print episode_end_data
    for e_num, _ in enumerate(ep_length):
        for key_in in metrics["episode_end_data"].keys():
            f_end_data.write("%20s " % (metrics["episode_end_data"][key_in][e_num]))
        f_end_data.write("\n")
    f_end_data.close()

    print("\nResults printed\n")

    return cp_dir