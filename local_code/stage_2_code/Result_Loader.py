'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.result import result
import pickle
from pathlib import Path


class Result_Loader(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def load(self):
        print('loading results...')
        destination_name = Path(self.result_destination_file_name).name
        if not destination_name.endswith('.pkl'):
            destination_name += '.pkl'
        if self.fold_count is not None:
            destination_name = f'{destination_name[:-4]}_{self.fold_count}.pkl'
        result_path = Path(self.result_destination_folder_path) / destination_name
        with result_path.open('rb') as f:
            self.data = pickle.load(f)
        return self.data