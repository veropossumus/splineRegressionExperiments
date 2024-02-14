import math
import os

import numpy as np
import pandas as pd
import scipy
import math
from aeon.datasets import load_from_tsfile
from sklearn.neighbors import LocalOutlierFactor

path = "../data/Univariate2018_ts"


def load_gbnc_data(file_name="nopos_al_excerpt.txt"):
    dir_path = "../data/nopos"
    file_path = os.path.join(dir_path, file_name)

    return pd.DataFrame(load_from_gbnc_file(file_path))


def load_from_gbnc_file(file_path):
    file = open(file_path, 'r', encoding="utf-8")
    data_list = []

    for line in file:
        ngram = line.strip().split('\t')
        ngram_name = ngram[0]
        values = [tuple(map(int, x.split(','))) for x in ngram[1:]]
        values.sort(key=lambda x: x[0])

        data_tuples = [(values[0][0], values[0][1])]

        # fill in missing years with word count of 0
        for i in range(1, len(values)):
            year_diff = values[i][0] - data_tuples[-1][0]
            assert (year_diff > 0)
            if year_diff != 1:
                for j in range(1, year_diff):
                    data_tuples.append((data_tuples[-1][0] + 1, 0))

            data_tuples.append((values[i][0], values[i][1]))

        data = normalize([x[1] for x in data_tuples])
        data_list.append({'dataset': 'GBNC', 'num': ngram_name, 'data': data})
    return data_list


def load_ucr_dataset_as_dict(number):
    file_contents = []
    try:
        file_name = os.listdir(path)[number]
        file_path = os.path.join(path, file_name, file_name + "_TRAIN.ts")
        dataset = load_from_tsfile(file_path, return_type="numpy2D")[0]

        counter = 0
        for i in range(len(dataset)):
            time_series = dataset[i].squeeze()
            if not np.isnan(time_series).any():
                data = normalize(time_series)
                file_contents.append({'dataset': file_name, 'num': counter, 'data': data})
                counter += 1

        return file_contents

    except Exception as e:
        print(f'Could not fetch file number {number}. Error: {str(e)}')
    return None


def load_ucr_dataset(number):
    dataset = load_ucr_dataset_as_dict(number)
    return pd.DataFrame(dataset)


def load_ucr_archive(min_ts_length=None, max_ts_length=None):
    datasets = []
    for i in range(len(os.listdir(path))):
        datasets.extend(load_ucr_dataset_as_dict(i))

    df = pd.DataFrame(datasets)

    if min_ts_length is not None:
        df = df[df['data'].apply(len) >= min_ts_length]
    if max_ts_length is not None:
        df = df[df['data'].apply(len) <= max_ts_length]

    return df


def load_ucr_data_short():
    return load_ucr_archive(50, 199)


def load_ucr_data_medium():
    return load_ucr_archive(200, 499)


def load_ucr_data_short_and_medium():
    return load_ucr_archive(50, 499)


def normalize(time_series: [float]) -> [(float, float)]:
    """
    Normalizes a time series.
    :param time_series: a list of y-values
    :return data_tuples: normalized y-values and the corresponding x-values
    """
    normalized_ts = scipy.stats.zscore(time_series)
    indices = np.linspace(0, 1, len(normalized_ts))
    data_tuples = list(zip(indices, normalized_ts))
    return data_tuples


"""def remove_outliers(time_series: [(int, int)]):
    lof = LocalOutlierFactor()
    filtered_ts = []

    y_values = [[tup[1]] for tup in time_series]

    outliers = lof.fit_predict(y_values)
    assert (len(outliers) == len(time_series))

    for i in range(len(outliers)):
        if outliers[i] == 1:
            filtered_ts.append(time_series[i])

    print("remove_outliers version: only feature: y-values")
    print("lof.n_features_in_", lof.n_features_in_)
    print("lof.n_samples_fit_", lof.n_samples_fit_)

    return filtered_ts"""


def remove_outliers(time_series: [(int, int)]):  # old version
    lof = LocalOutlierFactor()
    filtered_ts = []

    outliers = lof.fit_predict(time_series)
    assert (len(outliers) == len(time_series))

    for i in range(len(outliers)):
        if outliers[i] == 1:
            filtered_ts.append(time_series[i])

    """print("remove_outliers version: outliers = lof.fit_predict(time_series)")
    print("lof.n_features_in_", lof.n_features_in_)
    print("lof.n_samples_fit_",lof.n_samples_fit_)"""

    return filtered_ts


def replace_outliers(ts_without_outliers: [(int, int)], original_xs: [int]):
    ts_with_replacements = [(x, "nan") for x in original_xs]

    for i in range(len(ts_with_replacements)):
        x = ts_with_replacements[i][0]
        for j in range(len(ts_without_outliers)):
            if x == ts_without_outliers[j][0]:
                ts_with_replacements[i] = ts_without_outliers[j]

    first_number_idx = 0  # contains index of first tuple where y-value is not "nan"
    while ts_with_replacements[first_number_idx][1] == "nan":
        first_number_idx += 1

    # fill gaps at the beginning with copy of first number
    for i in range(first_number_idx):
        new_tuple = (ts_with_replacements[i][0], ts_with_replacements[first_number_idx][1])
        ts_with_replacements[i] = new_tuple

    # same for last number (but in reverse)
    last_number_idx = len(ts_with_replacements) - 1
    while ts_with_replacements[last_number_idx][1] == "nan":
        last_number_idx -= 1

    # fill gaps at the end with copy of last number
    for i in range(len(ts_with_replacements) - 1, last_number_idx, -1):
        new_tuple = (ts_with_replacements[i][0], ts_with_replacements[last_number_idx][1])
        ts_with_replacements[i] = new_tuple

    for i in range(first_number_idx, last_number_idx):
        if ts_with_replacements[i][1] == "nan":

            # find next tuple where y-value is not "nan"
            next_number_idx = i + 1

            while ts_with_replacements[next_number_idx][1] == "nan":
                next_number_idx += 1

            gap_len = next_number_idx - i
            print("gap_len", gap_len)

            previous_y = ts_with_replacements[i - 1][1]
            next_y = ts_with_replacements[next_number_idx][1]

            assert type(previous_y) is not str
            assert type(next_y) is not str

            # lin. interpolation for gaps in the middle (incl. gaps >= 2!)
            if previous_y == next_y:
                for j in range(gap_len):
                    ts_with_replacements[i + j] = ts_with_replacements[i - 1]

            else:
                increment_size = (abs(previous_y - next_y)) / (gap_len + 1)
                for j in range(gap_len):

                    if previous_y < next_y:
                        new_y_value = previous_y + increment_size * (j + 1)
                    else:
                        new_y_value = previous_y - increment_size * (j + 1)

                    new_tuple = (ts_with_replacements[i + j][0], new_y_value)
                    ts_with_replacements[i + j] = new_tuple

    return ts_with_replacements
