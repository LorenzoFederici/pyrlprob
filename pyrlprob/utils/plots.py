from typing import *
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from pyrlprob.utils.auxiliary import moving_average

def plot_metric(metric_mean: List[Any],
                out_dir: str,
                metric_min: Optional[List[Any]] = None,
                metric_max: Optional[List[Any]] = None,
                window: int = 1,
                title: str = "metric",
                label: Optional[str] = None,
                color: Optional[str] = None,
                plot_style: str="seaborn",
                latex_preamble: str="\\usepackage{amsmath}") -> None:
    """
    plot the trend during training of a metric

    Args:
        metric_mean (list): list containing the mean values of the metric to plot
        out_folder (str): the directory where the plot is saved
        metric_min (list): list containing the min values of the metric to plot
        metric_max (list): list containing the max values of the metric to plot
        window (int): window to evaluate the moving average
        title (str): title of the plot
        label (str): the label of the plotted curve
        color (str): color of the curve (if None -> random color)
        plot_style (str): plotting style
        latex_preamble (str): latex preamble
    """
    #Training iterations
    training_iter = [i for i in range(len(metric_mean))]
    
    #Evaluate the moving average
    metric_mean_mov = moving_average(metric_mean, window=window)
    training_iter_mov = training_iter[len(training_iter) - len(metric_mean_mov):]
    if metric_min is not None:
        metric_min_mov = moving_average(metric_min, window=window)
        metric_max_mov = moving_average(metric_max, window=window)
        metric_std_min_mov = np.array(metric_mean_mov) - abs(np.array(metric_mean_mov) - np.array(metric_min_mov))/4.
        metric_std_max_mov = np.array(metric_mean_mov) + abs(np.array(metric_mean_mov) - np.array(metric_max_mov))/4.

    #Setting plot style
    plt.style.use(plot_style)
    matplotlib.rc('font', size=22)
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('text.latex', preamble=latex_preamble)
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    #Plot
    fig = plt.figure()
    fig.set_size_inches(9.7,6.4)
    ax = fig.gca()
    ax.plot(training_iter_mov, metric_mean_mov, '-', linewidth='2.5', color=color)
    if metric_min is not None:
        plt.fill_between(training_iter_mov, metric_std_min_mov, metric_std_max_mov, alpha=0.3, color=color)
    plt.xlabel('Training iteration, $k$', fontsize=22)
    plt.ylabel(label, fontsize=22)
    plt.yscale('symlog', subsy=[2,3,4,5,6,7,8,9], linthreshy=0.005, linscaley=2.0)
    plt.xlim(0, training_iter_mov[-1])
    plt.xticks(np.arange(0, training_iter_mov[-1]+1, step=training_iter_mov[-1]/5))
    ax.tick_params(labelsize=22)
    ax.yaxis.grid(True, which='minor', linewidth=0.5, linestyle='dashed')
    ax.yaxis.grid(True, which='major', linewidth=0.5, linestyle='dashed')
    ax.xaxis.grid(True, which='major', linewidth=0.5, linestyle='dashed')
    ax.legend(fontsize=22, loc = 'best')
    fig.savefig(out_dir + title + ".pdf", dpi=300)