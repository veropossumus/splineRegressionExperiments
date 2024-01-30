import pandas as pd
import scipy
from aeon.datasets import load_from_tsfile
from sklearn.neighbors import LocalOutlierFactor
import os

path = "../data/Univariate2018_ts"


def load_ucr_dataset_as_dict(number):
    chunk_size = 50
    file_contents = []
    try:
        file_name = os.listdir(path)[number]
        file_path = os.path.join(path, file_name, file_name + "_TRAIN.ts")
        dataset = load_from_tsfile(file_path, return_type="numpy2D")[0]

        counter = 0
        for i in range(0, len(dataset), chunk_size):
            chunk = dataset[i:i + chunk_size]

            for j, time_series in enumerate(chunk):
                data = normalize(time_series.squeeze())
                file_contents.append({'dataset': file_name, 'num': counter, 'data': data})
                counter += 1

        return file_contents

    except Exception as e:
        print(f'Could not fetch file number {number}. Error: {str(e)}')
    return None


def load_ucr_dataset(number):
    dataset = load_ucr_dataset_as_dict(number)
    return pd.DataFrame(dataset)


def load_ucr_archive():
    datasets = []
    for i in range(len(os.listdir(path))):
        datasets.extend(load_ucr_dataset_as_dict(i))

    return pd.DataFrame(datasets)


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
