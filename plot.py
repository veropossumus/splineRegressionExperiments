import scipy
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from spline_utils import evaluate_spline

from fit import fit_max_spline, fit_max_l1_spline, fit_DFT
from spline_utils import calculate_max_dist


def plot_data(data):
    for elem in data:
        plt.scatter([d[0] for d in elem], [d[1] for d in elem], marker='.')
    plt.show()


def plot_splines(axis, knots, n, data, plot_LSQ=False, plot_max=True, plot_max_l1=False, plot_DFT=False, eps=0.0000001):
    results = []
    labels = []
    if plot_LSQ:
        labels.append(r'$L_2$')
        results.append(scipy.interpolate.make_lsq_spline([x[0] for x in data], [x[1] for x in data], knots, k=n).c)

    if plot_max:
        labels.append('r$L_{\infty}$')
        max_dist, result_max = fit_max_spline(data, knots, n)
        results.append(result_max)

        if plot_max_l1:
            labels.append(r'$L_{\infty}$ and $L_1$')
            results.append(fit_max_l1_spline(data, knots, n, eps=eps, t=max_dist)[1])

    if plot_max_l1:
        labels.append(r'$L_{\infty}$ and $L_1$')
        results.append(fit_max_l1_spline(data, knots, n, eps=eps)[1])

    if plot_DFT:
        labels.append('DFT')
        results.append(fit_DFT(data, knots, n))

    max_dists = [calculate_max_dist(knots, result, n, data)[0] for result in results]
    colors = list(mcolors.BASE_COLORS.keys())

    xs = np.linspace(0, 1, num=1000)

    print("opt distance", max_dist)

    for i in range(len(results)):
        axis.plot(xs, [evaluate_spline(knots, results[i], n, x) for x in xs], colors[i % len(colors)] + '-',
                  label=labels[i])
        axis.plot(xs, [evaluate_spline(knots, results[i], n, x) + max_dists[i] for x in xs],
                  colors[i % len(colors)] + '--')
        axis.plot(xs, [evaluate_spline(knots, results[i], n, x) - max_dists[i] for x in xs],
                  colors[i % len(colors)] + '--')

    axis.scatter([d[0] for d in data], [d[1] for d in data], marker='.')
    print("number of data points:", len(data))
    axis.legend()
