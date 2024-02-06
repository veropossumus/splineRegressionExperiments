import scipy
import numpy as np
import matplotlib.colors as mcolors
from utils.spline import evaluate_spline
import matplotlib.pyplot as plt

from utils.fit import fit_max_spline, fit_max_l1_spline
from utils.spline import calculate_max_dist


def plot_data(data):
    for elem in data:
        plt.scatter([d[0] for d in elem], [d[1] for d in elem], marker='.')
    plt.show()


def plot_splines(axis, knots, degree, data, eps=0.0000001, plot_max=True, plot_max_l1=False, plot_LSQ=False):
    results = []
    labels = []

    if plot_LSQ:
        labels.append(r'$L_2$')
        results.append(scipy.interpolate.make_lsq_spline([x[0] for x in data], [x[1] for x in data], knots, k=degree).c)

    if plot_max:
        labels.append(r'$L_{\infty}$')
        max_dist, result_max = fit_max_spline(data, knots, degree)
        results.append(result_max)

        if plot_max_l1:
            labels.append(r'$L_{\infty}$ and $L_1$')
            results.append(fit_max_l1_spline(data, knots, degree, eps=eps, t=max_dist)[1])
    else:
        if plot_max_l1:
            labels.append(r'$L_{\infty}$ and $L_1$')
            results.append(fit_max_l1_spline(data, knots, degree, eps=eps)[1])

    """if plot_DFT:
        labels.append('DFT')
        results.append(fit_DFT(data, num_coeffs))

    if plot_PAA:
        labels.append('PAA')
        results.append(fit_PAA(data, num_coeffs))"""

    max_dists = [calculate_max_dist(knots, result, degree, data)[0] for result in results]
    colors = list(mcolors.BASE_COLORS.keys())

    xs = np.linspace(0, 1, num=1000)

    print("opt distance", max_dist)

    for i in range(len(results)):
        axis.plot(xs, [evaluate_spline(knots, results[i], degree, x) for x in xs], colors[i % len(colors)] + '-',
                  label=labels[i])
        if len(max_dists) > 0:
            axis.plot(xs, [evaluate_spline(knots, results[i], degree, x) + max_dists[i] for x in xs],
                      colors[i % len(colors)] + '--')
            axis.plot(xs, [evaluate_spline(knots, results[i], degree, x) - max_dists[i] for x in xs],
                      colors[i % len(colors)] + '--')

    axis.scatter([d[0] for d in data], [d[1] for d in data], marker='.')
    print("number of data points:", len(data))
    axis.legend()


def plot_errors_against_degrees(dataframe):
    metrics = ['max_dist', 'MSE', 'MAE']
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i, metric in enumerate(metrics):
        avg_metric_by_degree = dataframe.groupby('degree')[metric].mean()
        axs[i].plot(avg_metric_by_degree.index, avg_metric_by_degree.values, marker='o', linestyle='-')
        axs[i].set_xlabel('Degree')
        axs[i].set_ylabel('Average ' + metric)
        axs[i].set_title('avg. ' + metric + ' for degrees 0 to ' + str(max(dataframe['degree'].unique())))
        axs[i].set_xticks(list(avg_metric_by_degree.index))
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


def plot_errors_against_compression_rates_avg_degree(dataframe):
    metrics = ['max_dist', 'MSE', 'MAE']
    compression_ratios = dataframe['compression_rate'].unique()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i, metric in enumerate(metrics):
        avg_mse_by_compression_rate = dataframe.groupby('compression_rate')[metric].mean()
        axs[i].plot(avg_mse_by_compression_rate.index, avg_mse_by_compression_rate.values, marker='o',
                    linestyle='-')
        axs[i].set_xlabel('compression_rate')
        axs[i].set_ylabel('avg. ' + metric)
        axs[i].set_title(
            'avg. ' + metric + ' vs. compression_rate (over degrees 0 to ' + str(
                max(dataframe['degree'].unique())) + ')')
        axs[i].set_xticks(compression_ratios)
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


def plot_errors_against_compression_rates_for_each_degree(dataframe):
    metrics = ['max_dist', 'MSE', 'MAE']
    for degree, group in dataframe.groupby('degree'):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        sub_df = dataframe[dataframe['degree'] == degree]
        for i, metric in enumerate(metrics):
            avg_metric_by_compression = sub_df.groupby('compression_rate')[metric].mean()
            print()
            axs[i].plot(avg_metric_by_compression.index, avg_metric_by_compression.values, marker='o',
                        linestyle='-')
            axs[i].set_xlabel('compression_rate')
            axs[i].set_ylabel('avg. ' + metric)
            axs[i].set_title('avg. ' + metric + ' for degree ' + str(degree))
            axs[i].set_xticks(list(avg_metric_by_compression.index))
            axs[i].grid(True)
            plt.tight_layout()

    #plt.tight_layout()
    plt.show()
