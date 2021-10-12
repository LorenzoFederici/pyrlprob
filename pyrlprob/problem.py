import numpy as np
import time
from typing import *
from itertools import islice
import importlib

import ray
import ray.rllib.agents as agents
from ray import tune

import yaml
import os

from pyrlprob.utils.auxiliary import *
import pyrlprob.utils.callbacks as callbacks
from pyrlprob.utils.plots import plot_metric


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
        self.input_config = settings

        #Trainer definition
        self.__algorithm = settings["run"]
        alg_module = importlib.import_module("ray.rllib.agents." + self.__algorithm)
        self.trainer = getattr(alg_module, self.__algorithm.upper() + "Trainer")

        #Stopping criteria definition
        self.stop = settings["stop"]

        #Config definition
        self.config = alg_module.DEFAULT_CONFIG.copy()
        self.config.update(settings["config"])

        #Evironment definition
        mod_name, env_name = self.config["env"].rsplit('.',1)
        mod = importlib.import_module(mod_name)
        self.env = getattr(mod, env_name)
        self.config["env"] = self.env
        self.env_config = self.config["env_config"]

        #Model definition
        self.model = self.config["model"]

        #Evaluation config definition
        self.evaluation = bool(self.config["evaluation_interval"])
        self.evaluation_config = self.config["evaluation_config"]

        #Custom metrics
        self.custom_metrics = []
        self.metric_names = None
        if "custom_metrics" in settings:
            self.custom_metrics = settings["custom_metrics"]   
        if "metric_names" in settings:
            self.metric_names = settings["metric_names"]

        #Callbacks and eval functions definition
        if self.config["callbacks"] != "DefaultCallbacks":
            if self.config["callbacks"] in ["TrainingCallbacks", "epsConstraintCallbacks"]:
                self.config["callbacks"] = getattr(callbacks, self.config["callbacks"])
            else:
                mod_name, fun_name = self.config["callbacks"].rsplit('.',1)
                mod = importlib.import_module(mod_name)
                self.config["callbacks"] = getattr(mod, fun_name)
        if self.config["custom_eval_function"] is not None:
            mod_name, fun_name = self.config["custom_eval_function"].rsplit('.',1)
            mod = importlib.import_module(mod_name)
            self.config["custom_eval_function"] = getattr(mod, fun_name)
        
        #Postprocessing definition
        self.postproc = {"custom_metrics": self.custom_metrics}
        if "postproc" in settings:
            self.postproc["episode_step_data"] = []
            self.postproc["episode_end_data"] = []
            if "episode_step_data" in settings["postproc"]:
                self.postproc["episode_step_data"] = settings["postproc"]["episode_step_data"]
            if "episode_end_data" in settings["postproc"]:
                self.postproc["episode_end_data"] = settings["postproc"]["episode_end_data"]
        
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
            load (dict): dictionary containing the directory and checkpoint where the 
                pre-trained model to load is located
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

        # Save elapsed time and results
        if create_out_file:
            f_out_res = open(best_exp_dir + "result.txt", "w")
            f_out_res.write("%22s %22s %22s %22s %22s\n" \
                % ("# elapsed time [s]", "training_iteration", \
                "episode_reward_mean", "episode_reward_max", "episode_reward_min"))
            f_out_res.write("%22.7f %22d %22.7f %22.7f %22.7f\n" \
                % (end_time - start_time, self.stop["training_iteration"], \
                    best_result["episode_reward_mean"], \
                    best_result["episode_reward_max"], \
                    best_result["episode_reward_min"]))
            f_out_res.close()

        #Terminate ray
        ray.shutdown()

        #Return trainer and best experiment directory
        return trainer_dir, best_exp_dir
    

    def evaluate(self, 
                 trainer_dir: str,
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
                 metric_names: Optional[List[str]]=None,
                 evaluation: bool=False,
                 graphs: bool=False,
                 postprocess: bool=True) -> str:
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
            metric_names (list): list with the names of the metrics (necessary just if graphs = True)
            evaluation (bool): are metrics computed through an evaluation environment?
            graphs (bool): whether to realize graphs of the metrics' trend along training
            postprocess (bool): whether to do postprocessing
        """
        
        #Path of metrics
        metric_path = ""
        if evaluation:
            metric_path = "evaluation/"

        #Determine the best checkpoint
        episode_reward = metric_training_trend(metric_path + "episode_reward_mean",
                                               exp_dirs,
                                               last_cps)
        best_cp = np.argmax(episode_reward)+1
        best_exp = next(exp for exp, cp in enumerate(last_cps) if cp >= best_cp)
        best_exp_dir = exp_dirs[best_exp]

        #Check what metrics/data are defined
        if metrics_and_data is None:
            metrics_and_data = {}
        for key in ["episode_step_data", "episode_end_data", "custom_metrics"]:
            if not key in metrics_and_data:
                metrics_and_data[key] = []
        
        #Load metrics and realize graphs
        if graphs:
            for m_num, metric in enumerate(["episode_reward"] + metrics_and_data["custom_metrics"]):
                if m_num > 0:
                    path = "custom_metrics/"
                else:
                    path = ""
                metric_trend_min = metric_training_trend(path + metric + "_min",
                                                        exp_dirs,
                                                        last_cps)
                metric_trend_mean = metric_training_trend(path + metric + "_mean",
                                                        exp_dirs,
                                                        last_cps)
                metric_trend_max = metric_training_trend(path + metric + "_max",
                                                        exp_dirs,
                                                        last_cps)
                plot_metric(metric_mean=metric_trend_mean,
                            out_dir=trainer_dir,
                            metric_min=metric_trend_min,
                            metric_max=metric_trend_max,
                            title=metric,
                            label=metric_names[m_num])
                    
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
            cp_dir = self.postprocess(best_exp_dir=new_best_exp_dir, 
                                      checkpoint=best_cp+1, 
                                      custom_metrics=metrics_and_data["custom_metrics"], 
                                      postproc=metrics_and_data, 
                                      evaluation=evaluation)
        else:
            cp_dir, _ = get_cp_dir_and_model(new_best_exp_dir, best_cp)
        
        return cp_dir


    def postprocess(self,
                    best_exp_dir: str,
                    checkpoint: int,
                    custom_metrics: Optional[List[str]]=None,
                    postproc: Optional[Dict[str, Any]]=None,
                    evaluation: bool=False) -> str:
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
                for metric in custom_metrics + postproc["episode_end_data"]}
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
                    q = column_progress(best_exp_dir+"progress.csv", metric_path + "hist_stats/" + key_in)
                    q = q[0].strip("[").strip("]").split(", ")
                    if key == "episode_step_data":
                        ep_length = column_progress(best_exp_dir+"progress.csv", \
                            metric_path + "hist_stats/episode_lengths")
                        ep_length = ep_length[0].strip("[").strip("]").split(", ")
                        ep_length = np.array(ep_length, dtype=int)
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
                        f_log.write("%20.7f " % (metrics[key][key_in][value][-1]))
        f_log.write("\n")
        f_log.close()

        for e_num, e in enumerate(ep_length):
            for h in range(e):
                for key_in in metrics["episode_step_data"].keys():
                    f_step_data.write("%20s " % (metrics["episode_step_data"][key_in][e_num][h]))
                f_step_data.write("\n")
            f_step_data.write("\n\n")
        f_step_data.close()

        for e_num, _ in enumerate(ep_length):
            for key_in in metrics["episode_end_data"].keys():
                f_end_data.write("%20s " % (metrics["episode_end_data"][key_in][e_num]))
            f_end_data.write("\n")
        f_end_data.close()

        print("\nResults printed\n")

        return cp_dir
    

    def solve(self,
              logdir: Optional[str]=None,
              evaluate: bool=True, 
              evaluation_num_episodes: int=1,
              postprocess: bool=True,
              graphs: bool=False,
              debug: bool=False) -> Tuple[str, str, str]:
        """
        Solve a RL problem.
        It include pre-processing and training, 
        and may include evaluation and post-processing.

        Args:
            logdir (str): name of the directory where training results are saved
            evaluate (bool): whether to do evaluation
            evaluation_num_episodes (int): number of evaluation episodes
            postprocess (bool): whether to do postprocessing
            graphs (bool): whether to realize graphs of metrics' trend during training
            debug (bool): whether to print worker's logs.
        
        Return:
            trainer_dir (str): trainer directory
            best_exp_dir (str): best experiment directory
            best_cp_dir (str): best checkpoint directory
        """
        
        #Training
        trainer_dir, best_exp_dir = self.train(trainer=self.trainer, 
                                               config=self.config, 
                                               stop=self.stop,
                                               logdir=logdir,
                                               load=self.load,
                                               debug=debug)
        
        #Save config file
        with open(best_exp_dir + "config.yaml", 'w') as outfile:
            yaml.dump(self.input_config, outfile)
        
        #Evaluation and Postprocessing
        if evaluate:
            last_checkpoint = self.stop["training_iteration"]
            exp_dirs = [best_exp_dir]
            last_cps = [last_checkpoint]
            if self.load is not None:
                exp_dirs = self.load["exp_dirs"].append(best_exp_dir)
                last_cps = self.load["last_cps"].append(last_checkpoint)

            env = self.env(self.env_config)
            best_cp_dir = self.evaluate(trainer_dir=trainer_dir, 
                                        exp_dirs=exp_dirs,
                                        last_cps=last_cps,
                                        model=self.model,
                                        gamma=self.config["gamma"],
                                        max_ep_length=env.max_episode_steps, 
                                        env=self.env, 
                                        env_config=self.env_config,
                                        evaluation_num_episodes=evaluation_num_episodes,
                                        evaluation_config=self.evaluation_config, 
                                        custom_eval_function=self.config["custom_eval_function"], 
                                        metrics_and_data=self.postproc, 
                                        metric_names=self.metric_names,
                                        evaluation=self.evaluation, 
                                        postprocess=postprocess,
                                        graphs=graphs)
        else:
            best_cp_dir = best_exp_dir

        return trainer_dir, best_exp_dir, best_cp_dir








    
    



    




