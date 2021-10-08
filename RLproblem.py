import numpy as np
import time
from typing import *
from itertools import islice

import ray
from ray import tune

import yaml
import os

from auxiliary import *
from callbacks import *


class RLProblem:
    """
    Abstract class defining a generic Reinforcement Learning Problem
    """

    def __init__(self, 
                 config_file: str) -> None:
        """ 
        Class constructor 
        
        Args:
            config_file (str): name of the config file (.yaml) that constains the trainer, environment and
                post-processing settings
        """

        #Open config file
        settings = yaml.safe_load(open(config_file))

        #Trainer definition
        self.__algorithm = settings["run"]
        self.trainer = globals()["ray.rllib.agents." + self.__algorithm + "." + self.__algorithm.upper() +  "Trainer"]

        #Config and stopping criteria definition
        self.stop = settings["stop"]
        self.config = settings["config"]

        #Evironment definition
        self.env = globals()[self.config["env"]]
        self.config["env"] = self.env
        self.env_config = self.config["env_config"]

        #Evaluation config definition
        self.evaluation = False
        if "evaluation_interval" in self.config:
            if self.config["evaluation_interval"]:
                self.evaluation = True

        self.evaluation_config = {}
        if "evaluation_config" in self.config:
            self.evaluation_config = self.config["evaluation_config"]

        #Custom metrics
        self.custom_metrics = []
        if "custom_metrics" in settings:
            self.custom_metrics = settings["custom_metrics"]   

        #Callbacks and eval functions definition
        if "callbacks" in self.config:
            self.config["callbacks"] = globals()[self.config["callbacks"]]
            if self.config["callbacks"] in [TrainingCallbacks, epsConstraintCallbacks]:
                callback = self.config["callbacks"](self.custom_metrics)
                self.config["callbacks"] = callback
        self.config["custom_eval_function"] = None
        if "custom_eval_function" in self.config:
            self.config["custom_eval_function"] = globals()[self.config["custom_eval_function"]]            
        
        #Postprocessing definition
        self.postproc = {"custom_metrics": self.custom_metrics}
        if "prostproc" in settings:
            self.postproc = {}
            self.postproc["episode_step_data"] = []
            self.postproc["episode_end_data"] = []
            if "episode_step_data" in settings["prostproc"]:
                self.postproc["episode_step_data"] = settings["prostproc"]["episode_step_data"]
            if "episode_step_data" in settings["prostproc"]:
                self.postproc["episode_end_data"] = settings["prostproc"]["episode_end_data"]
            if "plot_traj" in settings["prostproc"]:
                self.postproc["plot_traj"] = settings["prostproc"]["plot_traj"]
            if "plot_ctrl" in settings["prostproc"]:
                self.postproc["plot_ctrl"] = settings["prostproc"]["plot_ctrl"]
        
        #Pre-trained model definition
        self.load = None
        if "load" in settings:
            self.load = {}
            self.load["logdir"] = settings["load"]["trainer_dir"]
            prev_exp_dirs = []
            prev_last_cps = []
            if "prev_exp_dirs" in settings["load"]:
                prev_exp_dirs = settings["load"]["prev_exp_dirs"]
            if "prev_last_cp" in settings["load"]:
                prev_last_cps = settings["load"]["prev_last_cps"]
            self.load["exp_dirs"] = prev_exp_dirs.append(settings["load"]["last_exp_dir"])
            self.load["last_cps"] = prev_last_cps.append(settings["load"]["checkpoint"])
            _, self.load["checkpoint_dir"] = \
                get_cp_dir_and_model(settings["load"]["last_exp_dir"], settings["load"]["checkpoint"])
    

    def train(self,
              trainer: Union[str, Callable, Type], 
              config: Dict[str, Any], 
              stop: Dict[str, Any], 
              logdir: Optional[str]=None,
              create_out_file: bool=True, 
              load: Optional[Dict[str, Any]]=None, 
              debug: bool=False) -> Tuple[str, str]:
        """
        Train the current model with ray.tune, using the specified trainer and configs.

        Args:
            trainer (str or callable): trainer (i.e., RL algorithm) to train the model with
            config (dict): config file (dictionary)
            stop (dict): stopping conditions (dictionary)
            logdir (str): name of the directory where training results are saved
            create_out_file (bool): whether to create an outfile with run time and best result
            load (dict): dictionary containing the directory and checkpoint where the pre-trained model to load is located
            debug (bool): whether to print worker's logs.
        
        Return:
            trainer_dir (str): trainer's output directory
            best_exp_dir (str): directory containing the best experiment's output
        """

        #Create output folder
        if load is not None:
            outdir = load["logdir"]
            restore = load["checkpoint_dir"]
        else:
            if logdir is None:
                outdir = "./sol_saved/"
            else:
                outdir = logdir
            os.makedirs(outdir, exist_ok=True)
            restore = None

        #Initialize ray
        ray.init(log_to_driver=debug)
        
        #Train the model
        start_time = time.time()
        analysis = tune.run(
            trainer,
            config=config,
            local_dir=outdir,
            restore=restore,
            stop=stop,
            metric="episode_reward_mean",
            mode="max",
            checkpoint_freq=1,
            checkpoint_at_end=True,
            keep_checkpoints_num=None)
        end_time = time.time()

        #Best trial
        best_result = analysis.best_result
        best_exp_dir = analysis.best_logdir
        trainer_dir = best_exp_dir[:best_exp_dir.rfind("/")+1]
        best_exp_dir = best_exp_dir + "/"

        #Save config file
        with open(best_exp_dir + "config.yaml", 'w') as outfile:
            yaml.dump({"run": str(trainer), **stop, **config}, outfile)

        # Save elapsed time and results
        if create_out_file:
            f_out_res = open(best_exp_dir + "result.txt", "w")
            f_out_res.write("%22s %22s %22s %22s %22s\n" \
                % ("# elapsed time [s]", "training_iteration", \
                "episode_reward_mean", "episode_reward_max", "episode_reward_min"))
            f_out_res.write("%22.7f %22d %22.7f %22.7f %22.7f\n" \
                % (end_time - start_time, self.stop["training_iteration"], best_result["episode_reward_mean"], \
                    best_result["episode_reward_max"], best_result["episode_reward_min"]))
            f_out_res.close()

        #Terminate ray
        ray.shutdown()

        #Return trainer and best experiment directory
        return trainer_dir, best_exp_dir
    

    def evaluate(self, 
                 trainer_dir: str,
                 exp_dirs: List[str],
                 last_cps: List[int],
                 env: Union[Callable, str], 
                 env_config: Dict[str, Any],
                 evaluation_num_episodes: int, 
                 evaluation_config: Dict[str, Any],
                 custom_eval_function: Optional[Union[Callable, str]]=None,
                 metrics_and_data: Optional[Dict[str, Any]]=None,
                 evaluation: bool=False, 
                 postprocess: bool=True) -> None:
        """
        Evaluate a model, and find the best checkpoint

        Args:
            trainer_dir (str): trainer directory
            exp_dirs (list): list with experiments directories
            last_cps (list): list with last checkpoint number of each experiment in exp_dirs
            env (callable or str): Environment class (or class name)
            env_config (dict): dictionary containing the environment configs
            evaluation_num_episodes (int): number of evaluation episodes
            evaluation_config (dict): dictionary containing the evaluation configs
            custom_eval_function (callable or str): Custom evaluation function (or function name)
            metrics_and_data (dict): dictionary containing the metrics and data to save
                in the new file progress.csv
            evaluation (bool): are metrics computed through an evaluation environment?
            postprocess (bool): whether to do postprocessing
        """
        
        #Path of metrics
        metric_path = ""
        if evaluation:
            metric_path = "evaluation/"

        #Determine the best checkpoint
        episode_reward = []
        for exp_num, exp_dir in enumerate(exp_dirs):
            episode_reward = episode_reward + column_progress(exp_dir+"progress.csv", metric_path + "episode_reward_mean", last_cps[exp_num]+1)
        best_cp = np.argmax(episode_reward)+1
        best_exp = next(exp for exp, cp in enumerate(last_cps) if cp >= best_cp)
        best_exp_dir = exp_dirs[best_exp]

        #Check what metrics/data are defined
        if metrics_and_data is None:
            metrics_and_data = {}
        for key in ["episode_step_data", "episode_end_data", "custom_metrics"]:
            if not key in metrics_and_data:
                metrics_and_data[key] = []

        #Define standard PG trainer and configs for evaluation
        trainer = ray.rllib.agents.pg.PGTrainer
        config = trainer.DEFAULT_CONFIG.copy()
        config["rollout_fragment_length"] = 1
        config["train_batch_size"] = 1
        config["lr"] = 0.
        config["env"] = env if callable(env) else globals()[env]
        config["env_config"] = env_config
        config["evaluation_interval"] = 1
        config["evaluation_num_episodes"] = evaluation_num_episodes
        config["evaluation_num_workers"] = 0
        config["evaluation_config"] = evaluation_config
        config["custom_eval_function"] = custom_eval_function if callable(custom_eval_function) \
            else globals()[custom_eval_function]
        callback = EvaluationCallbacks(episode_step_data = metrics_and_data["episode_step_data"], \
            episode_end_data = metrics_and_data["episode_end_data"], \
            custom_metrics = metrics_and_data["custom_metrics"]
            )
        config["callbacks"] = callback
        stop = {"training_iteration": 1}

        #Define load properties
        load = {}
        load["logdir"] = trainer_dir
        _, load["checkpoint_dir"] = \
            get_cp_dir_and_model(best_exp_dir, best_cp)


        #Evaluation
        _, new_best_exp_dir = self.train(trainer=trainer, 
              config=config,
              stop=stop,
              load=load)
        
        #Postprocessing
        if postprocess:
            self.postprocess(best_exp_dir=new_best_exp_dir, 
                    checkpoint=best_cp, 
                    custom_metrics=metrics_and_data["custom_metrics"], 
                    postproc=metrics_and_data, 
                    evaluation=evaluation)


    def postprocess(self,
                    best_exp_dir: str,
                    checkpoint: int,
                    custom_metrics: Optional[List[str]]=None,
                    postproc: Optional[Dict[str, Any]]=None,
                    evaluation: bool=False) -> None:
        """
        Default postprocessing.

        Args:
            exp_dir (str): experiment directory
            checkpoint (int): number of the checkpoint to postprocess
            custom_metrics (list): custom metrics to include in the postprocessing
            postproc (dict): dictionary containing the data to postprocess and plotting files
            evaluation (bool): are metrics computed through an evaluation environment?
        """

        #Get checkpoint directory
        cp_dir, _ = get_cp_dir_and_model(best_exp_dir, checkpoint)

        #Path of metrics
        metric_path = ""
        if evaluation:
            metric_path = "evaluation/"
        
        if custom_metrics is None:
            custom_metrics = []

        if postproc is None:
            postproc = {}
            postproc["custom_metrics"] = []
            postproc["episode_step_data"] = []
            postproc["episode_end_data"] = []

        #Metrics to log
        values = ["min", "mean", "max"]
        metrics = {
            "metrics": {"episode_reward": {value: 0. for value in values}},
            "custom_metrics": {metric: {value: 0. for value in values} 
                for metric in custom_metrics},
            "episode_step_data": {metric: []
                for metric in postproc["episode_step_data"]}, 
            "episode_end_data": {metric: []
                for metric in postproc["episode_end_data"]}
            }

        #Save metrics of the best checkpoint
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

        for key, item in metrics.items():
            for key_in in item.keys():
                if "episode" in key:
                    q = column_progress(best_exp_dir+"progress.csv", metric_path + "hist_data/" + key_in)
                    q = q[0].strip("[").strip("]").split(", ")
                    #q = np.array(q, dtype=np.float64)
                    if key == "episode_step_data":
                        ep_length = column_progress(best_exp_dir+"progress.csv", \
                            metric_path + "hist_data/episode_lengths")
                        metrics[key][key_in] = [list(islice(q, i)) for i in ep_length]
                    else:
                        metrics[key][key_in] = q
                else:
                    if "custom" in key:
                        metric_type = "custom_metrics/"
                    else:
                        metric_type = ""
                    for value in values:
                        metrics[key][key_in][value] = column_progress(best_exp_dir+"progress.csv", \
                            metric_path + metric_type + key_in + "_" + value)
                        f_log.write("%20.7f " % (metrics[key][key_in][value]))
        f_log.write("\n")
        f_log.close()

        for e in ep_length:
            for h in range(e):
                    for key_in in metrics["episode_step_data"].keys():
                        f_step_data.write("%20s " % (metrics["episode_step_data"][key_in][e][h]))
                    f_step_data.write("\n")
            f_step_data.write("\n\n")
        f_step_data.close()

        for e in ep_length:
            for key_in in metrics["episode_end_data"].keys():
                f_end_data.write("%20s " % (metrics["episode_end_data"][key_in][e]))
            f_end_data.write("\n")
        f_end_data.close()

        if "plot_traj" in postproc:
            os.system(postproc["plot_traj"] + " " + cp_dir)
        if "plot_ctrl" in postproc:
            os.system(postproc["plot_ctrl"] + " " + cp_dir)

        print("\nResults printed, graphs plotted.\n")
    

    def solve(self,
              logdir: Optional[str]=None,
              evaluate: bool=True, 
              evaluation_num_episodes: int=1,
              postprocess: bool=True,
              debug: bool=False) -> None:
        """
        Solve a RL problem.
        It include pre-processing and training, 
            and may include evaluation and post-processing.

        Args:
            logdir (str): name of the directory where training results are saved
            evaluate (bool): whether to do evaluation
            evaluation_num_episodes (int): number of evaluation episodes
            evaluate (bool): whether to do postprocessing
            debug (bool): whether to print worker's logs.
        """
        
        #Training
        trainer_dir, best_exp_dir = self.train(trainer=self.trainer, 
                                               config=self.config, 
                                               stop=self.stop,
                                               logdir=logdir,
                                               load=self.load,
                                               debug=debug)
        
        #Evaluation and Postprocessing
        if evaluate:
            last_checkpoint = self.stop["training_iteration"]
            exp_dirs = [best_exp_dir]
            last_cps = [last_checkpoint]
            if self.load is not None:
                exp_dirs = self.load["exp_dirs"].append(best_exp_dir)
                last_cps = self.load["last_cps"].append(last_checkpoint)

            self.evaluate(trainer_dir=trainer_dir, 
                 exp_dirs=exp_dirs,
                 last_cps=last_cps,
                 env=self.env, 
                 env_config=self.env_config,
                 evaluation_num_episodes=evaluation_num_episodes,
                 evaluation_config=self.evaluation_config, 
                 custom_eval_function=self.config["custom_eval_function"], 
                 metrics_and_data=self.postproc, 
                 evaluation=self.evaluation, 
                 postprocess=postprocess)








    
    



    




