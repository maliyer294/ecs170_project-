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
        destination = Path(self.result_destination_file_name)
        if self.fold_count is not None:
            destination = destination.with_name(
                f'{destination.stem}_{self.fold_count}{destination.suffix}'
            )
        if destination.suffix != '.pkl':
            destination = destination.with_suffix('.pkl')
        result_path = Path(self.result_destination_folder_path) / destination
        with result_path.open('rb') as f:
            self.data = pickle.load(f)
        return self.data