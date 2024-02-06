import os

import numpy as np
import pandas as pd
import scipy
from aeon.datasets import load_from_tsfile
from sklearn.neighbors import LocalOutlierFactor

path = "../../data/Univariate2018_ts"


def load_gbnc_data(file_name="nopos_al_excerpt.txt"):
    dir_path = "../../data/nopos"
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


def normalize(time_series):
    """
    Normalizes a time series.
    :param [float] time_series: a list of y-values
    :return [(float,float)] data_tuples: normalized y-values and the corresponding x-values
    """
    normalized_ts = scipy.stats.zscore(time_series)
    data_tuples = [(i / (len(normalized_ts) - 1), normalized_ts[i]) for i in range(len(normalized_ts))]
    return data_tuples


def remove_outliers(time_series):
    lof = LocalOutlierFactor()
    filtered_ts = []

    outliers = lof.fit_predict(time_series)
    assert (len(outliers) == len(time_series))

    for i in range(len(outliers)):
        if outliers[i] == 1:
            filtered_ts.append(time_series[i])

    return filtered_ts
