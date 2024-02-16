import numpy as np
from typing import *
import importlib
import yaml
import copy

from ray.rllib.models import ModelCatalog

from pyrlprob.base_funs import *
from pyrlprob.utils import update
from pyrlprob.models import models


class RLProblem:
    """
    Abstract class defining a generic Reinforcement Learning Problem
    """

    def __init__(self, 
                 config_file: Union[str, Dict[str, Any]]) -> None:
        """ 
        Class constructor 
        
        Args:
            config_file (str or dict): name of the config file (.yaml) that constains the trainer, environment and
                post-processing settings or the dictionary with all the information
        """

        #Open config file
        if isinstance(config_file, str):
            settings = yaml.safe_load(open(config_file))
        else:
            settings = config_file
        self.input_config = settings

        #Trainer definition
        if isinstance(settings["run"], dict):
            self.__method = settings["run"]["method"]
            self.__algorithm = settings["run"]["algorithm"]
        else:
            self.__method = settings["run"]
            self.__algorithm = self.__method.upper()
        alg_module = importlib.import_module("ray.rllib.algorithms." + self.__method)
        self.alg_config = getattr(alg_module, self.__algorithm + "Config")
        self.trainer = self.__algorithm

        #Stopping criteria definition
        self.stop = settings["stop"]

        #Config definition
        self.config = self.alg_config().to_dict()
        update(self.config, settings["config"])

        #Evironment definition
        if "." in self.config["env"]:
            mod_name, env_name = self.config["env"].rsplit('.',1)
            mod = importlib.import_module(mod_name)
            env = getattr(mod, env_name)
            tune.register_env(self.config["env"], lambda config: env(config))
        self.env = self.config["env"]
        self.env_config = self.config["env_config"]

        #Model definition
        self.model = self.config["model"]
        if self.model["custom_model"] is not None:
            if self.model["custom_model"] in ["FCModelforRNNs", "MLPplusLSTM", "MLPplusGTrXL"]:
                custom_model = getattr(models, self.model["custom_model"])
                self.model["custom_model"] = "pyrlprob.models." + self.model["custom_model"]

            if "." in self.model["custom_model"]:
                mod_name, model_name = self.model["custom_model"].rsplit('.',1)
                mod = importlib.import_module(mod_name)
                custom_model = getattr(mod, model_name)
                ModelCatalog.register_custom_model(self.model["custom_model"], custom_model)
        
        #Framework definition
        self.framework = self.config["framework"]

        #Gamma definition
        self.gamma = self.config["gamma"]

        #Evaluation during training
        self.evaluation = bool(self.config["evaluation_interval"])
        self.evaluation_config = copy.deepcopy(self.config["evaluation_config"])
        if self.config["custom_eval_function"] is not None:
            mod_name, fun_name = self.config["custom_eval_function"].rsplit('.',1)
            mod = importlib.import_module(mod_name)
            self.config["custom_eval_function"] = getattr(mod, fun_name)
        self.custom_eval_function = self.config["custom_eval_function"]

        #Final evaluation
        self.final_evaluation_config = {}
        self.final_evaluation_duration = 1
        self.final_evaluation_duration_unit = "episodes"
        if "final_evaluation" in settings:
            if settings["final_evaluation"]:
                if "final_evaluation_duration" in settings:
                    self.final_evaluation_duration = settings["final_evaluation_duration"]
                    self.final_evaluation_duration_unit = settings["final_evaluation_duration_unit"]
                if "final_evaluation_config" in settings:
                    self.final_evaluation_config = settings["final_evaluation_config"]
        if self.evaluation_config is not None and self.evaluation_config != {}:
            update(self.evaluation_config, self.final_evaluation_config)
        else:
            self.evaluation_config = self.final_evaluation_config

        #Custom metrics
        self.custom_metrics = []
        if "custom_metrics" in settings:
            self.custom_metrics = settings["custom_metrics"]

        #Callbacks definition
        if not callable(self.config["callbacks"]):
            if self.config["callbacks"] != "DefaultCallbacks":
                if self.config["callbacks"] in ["TrainingCallbacks", "epsConstraintCallbacks"]:
                    self.config["callbacks"] = getattr(callbacks, self.config["callbacks"])
                else:
                    mod_name, fun_name = self.config["callbacks"].rsplit('.',1)
                    mod = importlib.import_module(mod_name)
                    self.config["callbacks"] = getattr(mod, fun_name)
        
        #Postprocessing definition
        self.postproc_data = {"custom_metrics": self.custom_metrics}
        if "postproc_data" in settings:
            self.postproc_data["episode_step_data"] = []
            self.postproc_data["episode_end_data"] = []
            if "episode_step_data" in settings["postproc_data"]:
                self.postproc_data["episode_step_data"] = settings["postproc_data"]["episode_step_data"]
            if "episode_end_data" in settings["postproc_data"]:
                self.postproc_data["episode_end_data"] = settings["postproc_data"]["episode_end_data"]
        
        #Pre-trained model definition
        self.load = None
        if "load" in settings:
            self.load = {}
            self.load["logdir"] = settings["load"]["trainer_dir"]
            self.load["exp_dirs"] = settings["load"]["prev_exp_dirs"]
            self.load["last_cps"] = settings["load"]["prev_last_cps"]
            _, self.load["checkpoint_dir"] = \
                get_cp_dir_and_model(self.load["exp_dirs"][-1], self.load["last_cps"][-1])
        
        # Update alg_config with the self.config
        self.alg_config = self.alg_config().from_dict(self.config)
    

    def solve(self,
              logdir: Optional[str]=None,
              evaluate: bool=True, 
              best_metric: str="episode_reward_mean",
              min_or_max: str="max",
              postprocess: bool=True,
              debug: bool=False,
              open_ray: bool=True,
              return_time: bool=False) ->  \
                  Union[Tuple[Dict[str, Any], str, List[str], List[int], str, float], \
                        Tuple[Dict[str, Any], str, List[str], List[int], str]]:
        """
        Solve a RL problem.
        It include pre-processing and training, 
        and may include evaluation and post-processing.

        Args:
            logdir (str): name of the directory where training results are saved
            evaluate (bool): whether to do evaluation
            best_metric (str): metric to be used to determine the best checkpoint in exp_dir during evaluation
            min_or_max (str): if best_metric must be minimized or maximized
            postprocess (bool): whether to do postprocessing
            debug (bool): whether to print worker's logs.
            open_ray (bool): whether to open/close ray
            return_time (bool): whether to return run time per iter
        
        Return:
            best_result (str): best result of the experiment
            trainer_dir (str): trainer directory
            exp_dirs (list): experiment directories
            last_cps (list): last checkpoints of the experiments
            best_cp_dir (str): best checkpoint directory
            (optional) run_time (float): total run time
        """
        
        #Training
        if return_time:
            best_result, trainer_dir, best_exp_dir, last_checkpoint, run_time = training(trainer=self.trainer, 
                                                                            config=self.config, 
                                                                            stop=self.stop,
                                                                            logdir=logdir,
                                                                            load=self.load,
                                                                            debug=debug,
                                                                            open_ray=open_ray,
                                                                            return_time=return_time)
        else:
            best_result, trainer_dir, best_exp_dir, last_checkpoint = training(trainer=self.trainer, 
                                                                  config=self.config, 
                                                                  stop=self.stop,
                                                                  logdir=logdir,
                                                                  load=self.load,
                                                                  debug=debug,
                                                                  open_ray=open_ray,
                                                                  return_time=return_time)
        
        #Save config file
        with open(best_exp_dir + "config.yaml", 'w') as outfile:
            yaml.dump(self.input_config, outfile)
        
        #Evaluation and Postprocessing
        if evaluate:
            exp_dirs, last_cps, best_cp_dir = self.evaluate(trainer_dir=trainer_dir,
                                                            exp_dir=best_exp_dir,
                                                            last_checkpoint=last_checkpoint,
                                                            best_metric=best_metric,
                                                            min_or_max=min_or_max,
                                                            do_postprocess=postprocess,
                                                            debug=debug)
        else:
            exp_dirs = [best_exp_dir]
            last_cps = [last_checkpoint]
            if self.load is not None:
                exp_dirs = self.load["exp_dirs"] + exp_dirs
                last_cps = self.load["last_cps"] + last_cps
            best_cp_dir, _ = get_cp_dir_and_model(best_exp_dir, last_checkpoint)

        if return_time:
            return best_result, trainer_dir, exp_dirs, last_cps, best_cp_dir, run_time
        else:
            return best_result, trainer_dir, exp_dirs, last_cps, best_cp_dir


    def evaluate(self,
                 trainer_dir: Optional[str] = None,
                 exp_dir: Optional[str] = None,
                 last_checkpoint: Optional[int] = None,
                 best_metric: str = "episode_reward_mean",
                 min_or_max: str = "max",
                 do_postprocess: Optional[bool] = True,
                 debug: bool = False) -> Tuple[List[str], List[int], str]:
        """
        Evaluate current model.
        It may include postprocessing.

        Args:
            trainer_dir (str): trainer directory
            exp_dir (str): experiment directory
            last_checkpoint (int): last checkpoint of the experiment
            best_metric (str): metric to be used to determine the best checkpoint in exp_dir
            min_or_max (str): if best_metric must be minimized or maximized
            do_postprocess (bool): whether to do postprocessing
            debug (bool): is debugging mode on?
        """

        exp_dirs = [exp_dir]
        last_cps = [last_checkpoint]
        if self.load is not None:
            exp_dirs = self.load["exp_dirs"] + exp_dirs
            last_cps = self.load["last_cps"] + last_cps
        best_cp_dir = evaluation(trainer=self.trainer,
                                 trainer_dir=trainer_dir, 
                                 exp_dirs=exp_dirs,
                                 last_cps=last_cps,
                                 config=self.config,
                                 env_name=self.env, 
                                 env_config=self.env_config,
                                 evaluation_duration=self.final_evaluation_duration,
                                 evaluation_duration_unit=self.final_evaluation_duration_unit,
                                 evaluation_config=self.evaluation_config, 
                                 custom_eval_function=self.custom_eval_function, 
                                 best_metric=best_metric,
                                 min_or_max=min_or_max,
                                 metrics_and_data=self.postproc_data, 
                                 is_evaluation_env=self.evaluation, 
                                 do_postprocess=do_postprocess,
                                 debug=debug)
        
        return exp_dirs, last_cps, best_cp_dir
    

    def postprocess(self,
                    exp_dir: str,
                    checkpoint: int) -> str:
        """
        Postprocess the experiment.

        Args:
            exp_dir (str): experiment directory
            checkpoint (int): experiment's checkpoint to postprocess
        
        Returns:
            cp_dir (str): postprocessed checkpoint directory
        """

        cp_dir = postprocessing(best_exp_dir=exp_dir, 
                                checkpoint=checkpoint, 
                                metrics_and_data=self.postproc_data, 
                                is_evaluation_env=self.evaluation)
        
        return cp_dir
