def calculate_max_compression(min_ts_length, max_degree):
    return (max_degree + 1) / min_ts_length


def generate_compression_ratios(dataframe, max_degree, compression_ratios=None):
    min_ts_length = dataframe['data'].apply(len).min()

    if compression_ratios is None:
        compression_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

    max_compression_ratio = calculate_max_compression(min_ts_length, max_degree)

    if (max_compression_ratio < min(compression_ratios)) & (max_compression_ratio not in compression_ratios):
        return [max_compression_ratio] + compression_ratios
    else:
        return compression_ratios


def round_to_nearest_ratio(number, ratios, num_dec_pts=2):
    ratios.sort()
    if number <= ratios[0]:
        return ratios[0]
    if number >= ratios[-1]:
        return ratios[-1]

    for i in range(len(ratios) - 1):
        if ratios[i] < number <= ratios[i + 1]:
            val_1 = round(number - ratios[i], num_dec_pts)
            val_2 = round(ratios[i + 1] - number, num_dec_pts)

            if val_1 == val_2:
                return ratios[i + 1]
            else:
                min_val = round(min(val_1, val_2), num_dec_pts)

                if min_val == val_1:
                    return ratios[i]
                elif min_val == val_2:
                    return ratios[i + 1]
