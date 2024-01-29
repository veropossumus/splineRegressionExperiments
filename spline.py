def evaluate_b_spline(a, n, i, u, m):
    if i == m - 1 and u == 1:
        return 1

    if n == 0:
        return 1 if a[i] <= u < a[i + 1] else 0

    left_alpha = (u - a[i]) / (a[i + n] - a[i]) if i + n < len(a) and a[i + n] != a[i] else 0
    left_term = left_alpha * evaluate_b_spline(a, n - 1, i, u, m)
    right_alpha = (u - a[i + 1]) / (a[i + n + 1] - a[i + 1]) if i + n + 1 < len(a) and a[i + n + 1] != a[
        i + 1] else 1
    right_term = (1 - right_alpha) * evaluate_b_spline(a, n - 1, i + 1, u, m)

    return left_term + right_term


def evaluate_spline(knots, control_points, n, u):
    result = 0
    m = len(knots) - 1 - n
    for i in range(0, len(control_points)):
        result += control_points[i] * evaluate_b_spline(knots, n, i, u, m)

    return result


def calculate_max_dist(knots, coeffs, n, data):
    max_dist = 0
    max_dist_idx = -1
    for i in range(len(data)):
        dist = abs(evaluate_spline(knots, coeffs, n, data[i][0]) - data[i][1])
        max_dist = max(dist, max_dist)
        if dist == max_dist:
            max_dist_idx = i

    return max_dist, max_dist_idx


"""def calculate_max_dist_splinter(spline, data):
    max_dist = 0
    max_dist_idx = -1
    for i in range(len(data)):
        dist = abs(spline.eval(data[i][0]) - data[i][1])
        max_dist = max(dist, max_dist)
        if dist == max_dist:
            max_dist_idx = i

    return max_dist, max_dist_idx"""
# %%
