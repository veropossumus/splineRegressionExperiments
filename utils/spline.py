def evaluate_b_spline(a, n, i, u, m):
    if i == m - 1 and u == 1:
        return 1

    if n == 0:
        return 1 if a[i] <= u < a[i + 1] else 0

    left_alpha = (u - a[i]) / (a[i + n] - a[i]) if i + n < len(a) and a[i + n] != a[i] else 0
    left_term = left_alpha * evaluate_b_spline(a, n - 1, i, u, m)
    right_alpha = (u - a[i + 1]) / (a[i + n + 1] - a[i + 1]) if i + n + 1 < len(a) and a[i + n + 1] != a[i + 1] else 1
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


def generate_coeff_counts(num_data_pts, degree, compression_ratios):
    counts = [int(ratio * num_data_pts) for ratio in compression_ratios]
    for i, count in enumerate(counts):
        if count < degree + 1:
            print("problem: count", count, "for degree", degree, ", but should be >=", degree + 1, "comp_ratio:",
                  compression_ratios[i])
    return counts


def generate_internal_knots(num_internal_knots):
    return [((x + 1) / (num_internal_knots + 1)) for x in range(num_internal_knots)]


def generate_knot_vector_from_coeff_count(degree, num_coeffs):
    num_knots = num_coeffs + degree + 1
    num_end_knots_each = degree + 1
    assert num_knots >= 2 * num_end_knots_each
    num_internal_knots = num_knots - 2 * num_end_knots_each

    internal_knots = generate_internal_knots(num_internal_knots)
    knot_vector = num_end_knots_each * [0] + internal_knots + num_end_knots_each * [1]
    return knot_vector


def generate_knot_vector_for_discontinuous_spline(degree, num_coeffs):
    num_knots = num_coeffs + degree + 1
    num_end_knots_each = degree + 1
    assert num_knots >= 2 * num_end_knots_each
    num_internal_knots = num_knots - 2 * num_end_knots_each
    internal_knots = generate_internal_knots(num_internal_knots)
    internal_knots_disc = [x for x in internal_knots for _ in range(degree + 1)]

    knot_vector = num_end_knots_each * [0] + internal_knots_disc + num_end_knots_each * [1]
    return knot_vector
