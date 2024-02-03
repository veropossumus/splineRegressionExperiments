from spline_utils import evaluate_b_spline
import numpy as np
from pyts.approximation import DiscreteFourierTransform, PiecewiseAggregateApproximation, SymbolicAggregateApproximation
from scipy.optimize import linprog


def fit_max_spline(data, knots, n):
    m = len(knots) - n - 1
    k = len(data)

    c = np.array([0] * m + [1])
    b = np.array([data[i][1] for i in range(k)] + [-data[i][1] for i in range(k)])

    A = []
    for i in range(2 * k):
        row = []

        sgn = 1 if i < k else -1
        for j in range(m):
            row.append(sgn * evaluate_b_spline(knots, n, j, data[i % k][0], m))

        row.append(-1)
        A.append(row)

    bounds = (None, None)

    x = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if x['status'] != 0:
        print("x status", x['status'])
        print("x['message']", x['message'])
        print("problem for knot count", len(knots), "and degree", n)
        return None, None

    return x['fun'], x['x'][:-1]


def fit_max_l1_spline(data, knots, n, eps=0, t=None):
    m = len(knots) - n - 1
    k = len(data)

    if t is None:
        t = fit_max_spline(data, knots, n)[0]
    bounds = [(None, None) for _ in range(m)] + [(0, t + eps)] + [(None, None) for _ in range(k)]

    c = np.array([0] * m + [0] + [1] * k)
    b = np.array(
        [data[i][1] for i in range(k)] + [-data[i][1] for i in range(k)] +
        [data[i][1] for i in range(k)] + [-data[i][1] for i in range(k)])

    A = []
    for i in range(2 * k):
        row = []
        sgn = 1 if i < k else -1
        for j in range(m):
            row.append(sgn * evaluate_b_spline(knots, n, j, data[i % k][0], m))

        row.append(-1)
        row.extend([0] * k)
        A.append(row)

    for i in range(2 * k):
        row = []
        sgn = 1 if i < k else -1
        for j in range(m):
            row.append(sgn * evaluate_b_spline(knots, n, j, data[i % k][0], m))

        row.append(0)
        row.extend([0] * (i % k) + [-1] + [0] * (k - (i % k) - 1))
        A.append(row)

    x2 = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if x2['status'] != 0:
        print("x2 status", x2['status'])
        print("NOT SUCCESSFUL")
        print(x2['message'])

    return x2['fun'], x2['x'][:m]


def fit_DFT(data, num_coeffs):
    return DiscreteFourierTransform(n_coefs=num_coeffs).fit_transform(data)


def fit_PAA(data, num_coeffs):
    window_size = int(len(data) / num_coeffs)
    return PiecewiseAggregateApproximation(window_size=window_size).fit_transform(data)


def fit_SAX(data, num_coeffs):
    return SymbolicAggregateApproximation(n_bins=num_coeffs).fit_transform(data)

# %%
