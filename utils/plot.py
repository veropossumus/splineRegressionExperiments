import scipy
import numpy as np
import matplotlib.colors as mcolors

from utils.data import replace_outliers
from utils.spline import evaluate_spline
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.titlesize':14, 'axes.labelsize':14})

from utils.fit import fit_max_spline, fit_max_l1_spline, calculate_inverse_DFT
from utils.spline import calculate_max_dist
from math import sqrt
import pandas as pd

import scipy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.spline import evaluate_spline, calculate_max_dist
from utils.fit import fit_max_spline, fit_max_l1_spline

plot_path = "../plots/e1/"


def calculate_errors(data_tuples, knots, n):
    result1 = scipy.interpolate.make_lsq_spline([x[0] for x in data_tuples], [x[1] for x in data_tuples], knots, k=n).c
    result2 = fit_max_spline(data_tuples, knots, n)[1]
    result3 = fit_max_l1_spline(data_tuples, knots, n, eps=1e-6)[1]

    results = [result1, result2, result3]

    errors = []
    data_points = [x[1] for x in data_tuples]
    counter = 0
    for result in results:
        counter += 1
        fitted_spline = [evaluate_spline(knots, result, n, x[0]) for x in data_tuples]  # data_points]
        max_dist = calculate_max_dist(knots, result, n, data_tuples)[0]
        errors.append([max_dist,  # maximum absolute distance,
                       mean_squared_error(data_points, fitted_spline),  # MSE
                       sqrt(mean_squared_error(data_points, fitted_spline)),  # RMSE
                       mean_absolute_error(data_points, fitted_spline)  # MAE
                       ])

    return errors


def plot_time_series(time_series):
    plt.scatter([d[0] for d in time_series], [d[1] for d in time_series], marker='.')
    plt.show()


def plot_data(data):
    for elem in data:
        plt.scatter([d[0] for d in elem], [d[1] for d in elem], marker='.')
    plt.show()


def plot_splines_with_without_outliers(data, data_lof, knots, degree, eps=0.000001, plot_max=True, plot_max_l1=True,
                                       plot_LSQ=False, plot_PAA=False, plot_PLA=False):
    row_names = ['LSQ', 'Max', 'Max and L1']

    assert (len(data) == len(data_lof))

    for i in range(len(data)):
        f, axes = plt.subplots(1, 2, sharey=True)
        f.set_figwidth(12)

        df1 = pd.DataFrame(error_metrics, columns=['max_dist', 'MSE', 'RMSE', 'MAE'])
        df1.index = row_names
        print(df1)

        # plot_splines(axes[0], data[i], fitting_methods)
        plot_splines(knots=knots, degree=degree, data=data[i], axis=axes[0], eps=eps, plot_max=plot_max,
                     plot_max_l1=plot_max_l1, plot_LSQ=plot_LSQ, plot_PAA=plot_PAA, plot_PLA=plot_PLA)
        axes[0].set_title("Data including outliers")

        error_metrics = calculate_errors(data_lof[i], knots, degree)
        df2 = pd.DataFrame(error_metrics, columns=['max_dist', 'MSE', 'RMSE', 'MAE'])
        df2.index = row_names
        print(df2)

        # plot_splines(axes[1], data_lof[i], fitting_methods)
        plot_splines(knots=knots, degree=degree, data=data_lof[i], axis=axes[1], eps=eps, plot_max=plot_max,
                     plot_max_l1=plot_max_l1, plot_LSQ=plot_LSQ, plot_PAA=plot_PAA, plot_PLA=plot_PLA,
                     outliers_removed=True, original_xs=[tup[0] for tup in data])
        axes[1].set_title("Data without outliers")

        plt.subplots_adjust(bottom=0.25, top=0.95)

        plt.show()


def add_fitted_curve_to_plot(axis, xs, fitted_curve: [float], max_dist: float, label: None, color: str = None):
    if label is not None and color is None:
        match label:
            case 'PAA':
                color = 'tab:gray'
            case 'PLA':
                color = 'tab:olive'
            case 'L8':
                #color = 'tab:pink'
                #color = 'xkcd:bordeaux'
                color='red'
                label = r'$L_\infty$'
            case 'L8 and L1':
                label = r'$L_\infty^*$'
                color = 'tab:blue'
                #color = 'xkcd:pumpkin'
            case 'LSQ':
                color = 'tab:purple'
            case 'DFT':
                color = 'tab:green'
            case 'PR':
                color = 'tab:orange'
            case 1:
                color = 'tab:olive'
            case 3:
                color = 'tab:pink'

    print("max_dist for method", label, ":", max_dist)

    axis.plot(xs, fitted_curve, color=color, linestyle='solid', label=label)
    #if abs(max_dist) > 0:
        #axis.plot(xs, [y + max_dist for y in fitted_curve], color=color, linestyle='dashed')
        #axis.plot(xs, [y - max_dist for y in fitted_curve], color=color, linestyle='dashed')


def plot_fitted_curve(xs, fitted_curve: [float], max_dist: float, label=None):
    color = "tab:blue"
    match label:
        case 'L8':
            label = r'$L_\infty$'
        case 'L8 and L1':
            label = r'$L_\infty$'
    plt.plot(xs, fitted_curve, color=color, linestyle='solid', label=label)
    if abs(max_dist) > 0:
        plt.plot(xs, [y + max_dist for y in fitted_curve], color=color, linestyle='dashed')
        plt.plot(xs, [y - max_dist for y in fitted_curve], color=color, linestyle='dashed')

    plt.legend()
    plt.show()


def plot_splines(knots, degree, data, axis=None, eps=0.000001, plot_max=True, plot_max_l1=True, plot_LSQ=True,
                 plot_PAA=True, plot_PLA=True, outliers_removed=False, original_xs=None):
    results = []
    labels = []

    if axis is None:
        axis = plt

    if plot_LSQ:
        print("outliers_removed?", outliers_removed)
        if (outliers_removed):  # & (original_xs is not None):
            if original_xs is None:
                print("please provide the complete list of x-values of the original time series")
                return
            else:
                labels.append(r'$L_2$')
                ts_with_replacements = replace_outliers(data, original_xs)
                results.append((degree, scipy.interpolate.make_lsq_spline([x[0] for x in ts_with_replacements],
                                                                          [x[1] for x in ts_with_replacements], knots,
                                                                          k=degree).c))
        elif not outliers_removed:
            labels.append(r'$L_2$')
            results.append((degree, scipy.interpolate.make_lsq_spline([x[0] for x in data], [x[1] for x in data], knots,
                                                                      k=degree).c))
        """try:
            labels.append(r'$L_2$')
            results.append((degree, scipy.interpolate.make_lsq_spline([x[0] for x in data], [x[1] for x in data], knots,
                                                                      k=degree).c))
        except:
            try:
                labels.append(r'$L_2$')
            results.append((degree, scipy.interpolate.make_lsq_spline([x[0] for x in data], [x[1] for x in data], knots,
                                                                      k=degree).c))
            print("LSQ problem")"""

    if plot_max:
        labels.append(r'$L_{\infty}$')
        max_dist, result_max = fit_max_spline(data, knots, degree)
        results.append((degree, result_max))

        if plot_max_l1:
            labels.append(r'$L_{\infty}$ and $L_1$')
            results.append((degree, fit_max_l1_spline(data, knots, degree, eps=eps, t=max_dist)[1]))
    else:
        if plot_max_l1:
            labels.append(r'$L_{\infty}$ and $L_1$')
            results.append((degree, fit_max_l1_spline(data, knots, degree, eps=eps)[1]))

    if plot_PLA:
        labels.append('PLA')
        max_dist, result_max = fit_max_spline(data, knots, 1)
        results.append((1, result_max))

    if plot_PAA:
        labels.append('PAA')
        max_dist, result_max = fit_max_spline(data, knots, 0)
        results.append((0, result_max))

    max_dists = [calculate_max_dist(knots, result, degree, data)[0] for degree, result in results]
    colors = list(mcolors.BASE_COLORS.keys())

    xs = np.linspace(0, 1, num=1000)

    # print("optimum distance", max_dist)

    for i in range(len(results)):
        degree = results[i][0]
        coeffs = results[i][1]
        axis.plot(xs, [evaluate_spline(knots, coeffs, degree, x) for x in xs], colors[i % len(colors)] + '-',
                  label=labels[i])
        if len(max_dists) > 0:
            axis.plot(xs, [evaluate_spline(knots, coeffs, degree, x) + max_dists[i] for x in xs],
                      colors[i % len(colors)] + '--')
            axis.plot(xs, [evaluate_spline(knots, coeffs, degree, x) - max_dists[i] for x in xs],
                      colors[i % len(colors)] + '--')

    """for i in range(len(results)):
        axis.plot(xs, [evaluate_spline(knots, results[i], degree, x) for x in xs], colors[i % len(colors)] + '-',
                  label=labels[i])
        if len(max_dists) > 0:
            axis.plot(xs, [evaluate_spline(knots, results[i], degree, x) + max_dists[i] for x in xs],
                      colors[i % len(colors)] + '--')
            axis.plot(xs, [evaluate_spline(knots, results[i], degree, x) - max_dists[i] for x in xs],
                      colors[i % len(colors)] + '--')"""

    axis.scatter([d[0] for d in data], [d[1] for d in data], marker='.')
    print("number of data points:", len(data))
    axis.legend()
    plt.show()


def plot_errors_against_degrees(dataframe, savefig=False):
    metrics = ['max_dist', 'MSE', 'MAE']
    metric_names = ['max. distance', 'MSE', 'MAE']
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    #fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    df_list = []

    for i, metric in enumerate(metrics):
        avg_metric_by_degree = dataframe.groupby('degree')[metric].mean()
        df_list.append(avg_metric_by_degree)
        axs[i].plot(avg_metric_by_degree.index, avg_metric_by_degree.values, marker='o', linestyle='-')
        axs[i].set_xlabel('Degree')
        axs[i].set_ylabel('Mean ' + metric_names[i])
        axs[i].set_title('Mean ' + metric_names[i])  # + ' for degrees 0 to ' + str(max(dataframe['degree'].unique())))
        axs[i].set_xticks(list(avg_metric_by_degree.index))
        axs[i].grid(True)
        #axs[i].set_ylim(ymin=0)

    plt.tight_layout(pad=3)

    if savefig:
        plt.savefig(plot_path + "plot_errors_against_degrees.pdf")

    plt.show()

    df = pd.concat(df_list,axis=1)
    print(df)
    print(df.to_latex())

    return


def plot_errors_against_compression_rates_avg_degree(dataframe, savefig=False):
    metrics = ['max_dist', 'MSE', 'MAE']
    metric_names = ['max. distance', 'MSE', 'MAE']
    compression_ratios = dataframe['compression_rate'].unique()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    #fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    df_list = []

    for i, metric in enumerate(metrics):
        avg_mse_by_compression_rate = dataframe.groupby('compression_rate')[metric].mean()
        df_list.append(avg_mse_by_compression_rate)
        axs[i].plot(avg_mse_by_compression_rate.index, avg_mse_by_compression_rate.values, marker='o', linestyle='-')
        axs[i].set_xlabel('Compression ratio')
        axs[i].set_ylabel('Mean ' + metric_names[i])
        axs[i].set_title('Mean ' + metric_names[
            i] + ' vs. compression ratio')  # + ' vs. compression_rate (over degrees 0 to ' + str(max(dataframe['degree'].unique())) + ')')
        axs[i].set_xticks(compression_ratios)
        axs[i].grid(True)
        #axs[i].set_ylim(ymin=0)

    plt.tight_layout(pad=3)

    if savefig:
        plt.savefig(plot_path + "plot_errors_against_compression_rates_avg_degree.pdf")

    plt.show()

    df = pd.concat(df_list,axis=1)
    print(df)
    print(df.to_latex())

    return



def plot_errors_against_compression_rates_for_each_degree(dataframe, savefig=False):
    metrics = ['max_dist', 'MSE', 'MAE']
    metric_names = ['max. distance', 'MSE', 'MAE']
    for degree, group in dataframe.groupby('degree'):

        df_list = []

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        #fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        sub_df = dataframe[dataframe['degree'] == degree]
        for i, metric in enumerate(metrics):
            avg_metric_by_compression = sub_df.groupby('compression_rate')[metric].mean()
            df_list.append(avg_metric_by_compression)
            axs[i].plot(avg_metric_by_compression.index, avg_metric_by_compression.values, marker='o', linestyle='-')
            axs[i].set_xlabel('Compression ratio')
            axs[i].set_ylabel('Mean ' + metric_names[i])
            axs[i].set_title('Mean ' + metric_names[i] + ' for degree ' + str(degree))
            axs[i].set_xticks(list(avg_metric_by_compression.index))
            axs[i].grid(True)
            #axs[i].set_ylim(ymin=0)
            plt.tight_layout(pad=3)

        if savefig:
            plt.savefig(plot_path + "plot_errors_against_compression_rates_for_degree_" + str(degree) + ".pdf")

        df = pd.concat(df_list,axis=1)
        print("degree:",degree)
        print(df)
        print(df.to_latex())

    plt.show()
    return

def plot_inverse_DFT(dft_coeffs, time_series, num_coeffs):
    x_values = [tup[0] for tup in time_series]
    y_values = [tup[1] for tup in time_series]
    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, 'o--', ms=4, label='Original')
    plt.plot(x_values, dft_coeffs, 'o--', ms=4, label='DFT - {0} coefs'.format(num_coeffs))
    plt.legend(loc='best', fontsize=10)
    plt.xlabel('Time', fontsize=14)
    plt.title('Discrete Fourier Transform', fontsize=16)
    plt.show()
