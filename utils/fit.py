import scipy
from utils.spline import evaluate_b_spline, calculate_max_dist
import numpy as np
from pyts.approximation import DiscreteFourierTransform, PiecewiseAggregateApproximation
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def fit_max_spline(data, knots, n) -> [float, [float]]:
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

    A_sparse = csr_matrix(A)

    bounds = (None, None)

    x = linprog(c, A_ub=A_sparse, b_ub=b, bounds=bounds, options={'disp': True})

    if x['status'] != 0:
        print("problem for knot count", len(knots), "and degree", n, "i.e. for num_coeff =", len(knots) - n - 1)
        print(x)
        return None, None

    return x['fun'], x['x'][:-1]


def fit_max_l1_spline(data, knots, n, eps=0, t=None) -> [float, [float]]:
    m = len(knots) - n - 1
    k = len(data)

    if t is None:
        t = fit_max_spline(data, knots, n)[0]
    bounds = [(None, None) for _ in range(m)] + [(0, t + eps)] + [(None, None) for _ in range(k)]

    c = np.array([0] * m + [0] + [1] * k)
    b = np.array([data[i][1] for i in range(k)]
                 + [-data[i][1] for i in range(k)]
                 + [data[i][1] for i in range(k)]
                 + [-data[i][1] for i in range(k)])

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

    A_sparse = csr_matrix(A)

    x2 = linprog(c, A_ub=A_sparse, b_ub=b, bounds=bounds)

    if x2['status'] != 0:
        print("problem for knot count", len(knots), "and degree", n, "i.e. for num_coeff =", len(knots) - n - 1)

        while eps < 1e-5:
            print("increasing eps from", eps, "to", eps + 1e-8)
            eps += 1e-7
            print("new eps:", eps)
            bounds = [(None, None) for _ in range(m)] + [(0, t + eps)] + [(None, None) for _ in range(k)]
            x2 = linprog(c, A_ub=A_sparse, b_ub=b, bounds=bounds, options={'disp': True})
            if x2['status'] == 0:
                print("success for eps =", eps)
                return x2['fun'], x2['x'][:m]

        print("NOT SUCCESSFUL")
        print("x2 status", x2['status'])
        print(x2['message'])
        print("x2 \n:", x2)
        return None, None

    return x2['fun'], x2['x'][:m]


def fit_LSQ_spline(time_series: [(int, int)], knots: [int], degree: int) -> [float]:
    xs = [x[0] for x in time_series]
    ys = [x[1] for x in time_series]
    result = scipy.interpolate.make_lsq_spline(xs, ys, knots, k=degree).c
    return result


def fit_DFT(data, num_coeffs) -> [float]:
    X = [[x[1] for x in data]]
    dft = DiscreteFourierTransform(n_coefs=num_coeffs)
    X_dft = dft.fit_transform(X)
    return X_dft


def calculate_inverse_DFT(num_data_pts, num_coeffs, X_dft):
    n_coefs = num_coeffs
    n_samples = 1
    n_timestamps = num_data_pts

    if n_coefs % 2 == 0:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs, 2)
        X_dft_new = np.c_[X_dft[:, :1], X_dft[:, real_idx] + 1j * np.c_[X_dft[:, imag_idx], np.zeros((n_samples,))]]
    else:
        real_idx = np.arange(1, n_coefs, 2)
        imag_idx = np.arange(2, n_coefs + 1, 2)
        X_dft_new = np.c_[X_dft[:, :1], X_dft[:, real_idx] + 1j * X_dft[:, imag_idx]]

    X_irfft = np.fft.irfft(X_dft_new, n_timestamps)[0]

    result = scipy.stats.zscore(X_irfft)
    if np.isnan(result).all():
        result = np.nan_to_num(result, nan=0)

    return result
