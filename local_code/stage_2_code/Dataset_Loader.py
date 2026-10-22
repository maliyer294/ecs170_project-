'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import csv
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Dataset_Loader(dataset):
    dataset_source_folder_path = PROJECT_ROOT / 'data' / 'stage_2_data'
    train_file_name = 'train.csv'
    test_file_name = 'test.csv'
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        return {
            'train': self._load_file(self.train_file_name),
            'test': self._load_file(self.test_file_name),
        }

    def _load_file(self, file_name):
        X = []
        y = []
        source_path = Path(self.dataset_source_folder_path) / file_name
        print(f'loading data from {source_path}...')
        with source_path.open(newline='') as f:
            for line_number, row in enumerate(csv.reader(f), start=1):
                if len(row) != 785:
                    raise ValueError(
                        f'{source_path}:{line_number} has {len(row)} values; expected 785'
                    )
                values = [int(value) for value in row]
                label, features = values[0], values[1:]
                if label not in range(10):
                    raise ValueError(
                        f'{source_path}:{line_number} has invalid digit label {label}'
                    )
                if any(value < 0 or value > 255 for value in features):
                    raise ValueError(
                        f'{source_path}:{line_number} has a pixel value outside 0-255'
                    )
                X.append(np.asarray(features, dtype=np.float32) / 255.0)
                y.append(label)
        return {'X': np.stack(X), 'y': np.asarray(y, dtype=np.int64)}