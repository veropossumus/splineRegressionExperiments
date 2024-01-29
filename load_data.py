import scipy
from sklearn.neighbors import LocalOutlierFactor
from aeon.datasets import load_classification
from pyts.datasets import ucr_dataset_list, ucr_dataset_info
import numpy as np

"""def get_data_train_from_dataset(datasets):
    data_train = []
    for dataset in datasets:
        data_train.append(dataset.data_train)
    return data_train"""


def remove_outliers(dataset):
    lof = LocalOutlierFactor()
    filtered_data = []

    for data_tuples in dataset:
        outliers = lof.fit_predict(data_tuples)
        assert (len(outliers) == len(data_tuples))
        filtered_data_tuples = []
        for i in range(len(outliers)):
            if outliers[i] == 1:
                filtered_data_tuples.append(data_tuples[i])

        filtered_data.append(filtered_data_tuples)

    return filtered_data


def normalize_datasets(datasets):
    normalized_datasets = []
    for dataset in datasets:
        normalized_datasets.append(dataset.normalize())
    return normalized_datasets


def create_dataset_from_ucr_name(dataset_name):
    dataset_info = ucr_dataset_info(dataset_name)
    try:
        X, y, meta_data = load_classification(dataset_name)
        dataset = Dataset(name=dataset_name,
                          data_train=[np.squeeze(time_series) for time_series in X],
                          target_train=y,
                          n_classes=dataset_info['n_classes'],
                          n_timestamps=dataset_info['n_timestamps'],
                          test_size=dataset_info['test_size'],
                          train_size=dataset_info['train_size'],
                          dataset_type=dataset_info['type'])
        return dataset
    except Exception as e:
        print(f'Could not fetch {dataset_name}. Error: {str(e)}')
    return None


def load_ucr_archive():
    dataset_list = ucr_dataset_list()
    ucr_datasets = []
    for dataset_name in dataset_list:
        new_dataset = create_dataset_from_ucr_name(dataset_name)
        if new_dataset is not None:
            ucr_datasets.append(new_dataset)
    return ucr_datasets


def load_ucr_dataset(number):
    dataset_list = ucr_dataset_list()
    dataset_name = dataset_list[number]

    return create_dataset_from_ucr_name(dataset_name)


def load_gbnc_ngrams_as_datasets(file_path="nopos/nopos_al_excerpt.txt"):
    file = open(file_path, 'r', encoding="utf-8")
    gbnc_ngrams = []
    for line in file:
        ngram = line.strip().split('\t')
        ngram_name = ngram[0]
        values = [tuple(map(int, x.split(','))) for x in ngram[1:]]
        values.sort(key=lambda x: x[0])

        # while values[0][0] < 1800:
        # values.pop(0)

        data_tuples = [(values[0][0], values[0][1])]

        # fill in missing years with word count of 0
        for i in range(1, len(values)):
            year_diff = values[i][0] - data_tuples[-1][0]
            assert (year_diff > 0)
            if year_diff != 1:
                for j in range(1, year_diff):
                    data_tuples.append((data_tuples[-1][0] + 1, 0))

            data_tuples.append((values[i][0], values[i][1]))

        # add list level for comparability with ucr (multiple ts in one dataset there!)
        dataset = Dataset(ngram_name, [[x[1] for x in data_tuples]])
        gbnc_ngrams.append(dataset)

    return gbnc_ngrams


class Dataset:
    def __init__(self, name, data_train, target_train=None, data_test=None, target_test=None, n_classes=None,
                 n_timestamps=None, test_size=None, train_size=None,
                 dataset_type=None):
        self.name = name
        self.data_train = data_train
        self.data_test = data_test
        self.target_train = target_train
        self.target_test = target_test
        self.n_classes = n_classes
        self.n_timestamps = n_timestamps
        self.test_size = test_size
        self.train_size = train_size
        self.dataset_type = dataset_type

    def normalize(self):
        normalized_dataset = []
        for time_series in self.data_train:
            normalized_ts = scipy.stats.zscore(np.array(time_series))
            # data_tuples = [(i / (len(normalized_dataset)), normalized_dataset[i]) for i in range(len(
            # normalized_dataset))]
            data_tuples = [(i / (len(normalized_ts) - 1), normalized_ts[i]) for i in range(len(normalized_ts))]
            normalized_dataset.append(data_tuples)
        return normalized_dataset

# %%
