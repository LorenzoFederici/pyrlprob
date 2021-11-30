from typing import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pyrlprob.utils.auxiliary import moving_average, metric_training_trend

def plot_metric(metric_name: str,
                experiment_dirs: List[str],
                last_checkpoints: List[int],
                window: int = 1,
                fig: Optional[Callable] = None,
                label: Optional[str] = None,
                color: Optional[str] = None) -> Callable:
    """
    plot the trend during training of a metric

    Args:
        metric_name (str): full name of the metric (with the path)
        experiment_dirs (list): list of the experiment directories
        last_checkpoints (list): list with the last checkpoints of the experiments
        window (int): window to evaluate the moving average
        fig (callable): figure object
        label (str): the label of the plotted curve
        color (str): color of the curve (if None -> random color)
    
    Return:
        fig: figure object
    """

    #Retrieve metric min, mean and max trend
    if "_mean" in metric_name:
        metric_mean = metric_training_trend(metric_name,
                                        experiment_dirs,
                                        last_checkpoints)
        metric_min = None
        metric_max = None
    else:
        metric_min = metric_training_trend(metric_name + "_min",
                                        experiment_dirs,
                                        last_checkpoints)
        metric_mean = metric_training_trend(metric_name + "_mean",
                                            experiment_dirs,
                                            last_checkpoints)
        metric_max = metric_training_trend(metric_name + "_max",
                                        experiment_dirs,
                                        last_checkpoints)

    #Training iterations
    training_iter = [i for i in range(len(metric_mean))]
    
    #Evaluate the moving average
    metric_mean_mov = moving_average(metric_mean, window=window)
    training_iter_mov = training_iter[len(training_iter) - len(metric_mean_mov):]
    if metric_min != None:
        metric_min_mov = moving_average(metric_min, window=window)
        metric_max_mov = moving_average(metric_max, window=window)
        metric_std_min_mov = np.array(metric_mean_mov) - abs(np.array(metric_mean_mov) - np.array(metric_min_mov))/4.
        metric_std_max_mov = np.array(metric_mean_mov) + abs(np.array(metric_mean_mov) - np.array(metric_max_mov))/4.

    #Plot
    if fig is None:
        fig = plt.figure()
        fig.set_size_inches(9.7,6.4)
    ax = fig.gca()
    ax.plot(training_iter_mov, metric_mean_mov, '-', linewidth='2.5', color=color, label=label)
    if metric_min !=  None:
        plt.fill_between(training_iter_mov, metric_std_min_mov, metric_std_max_mov, alpha=0.3, color=color)
    
    return fig