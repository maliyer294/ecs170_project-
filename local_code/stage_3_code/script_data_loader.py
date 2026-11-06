import pickle
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_ROOT / 'data' / 'stage_3_data'

DATASET_SPECS = {
    'ORL': {
        'file_name': 'ORL',
        'image_shape': (112, 92, 3),
        'label_range': (1, 40),
        'label_offset': 1,
        'description': 'ORL grayscale face images for 40 people',
    },
    'MNIST': {
        'file_name': 'MNIST',
        'image_shape': (28, 28),
        'label_range': (0, 9),
        'label_offset': 0,
        'description': 'MNIST handwritten digit images',
    },
    'CIFAR': {
        'file_name': 'CIFAR',
        'image_shape': (32, 32, 3),
        'label_range': (0, 9),
        'label_offset': 0,
        'description': 'CIFAR-10 colored object images',
    },
}


class Stage3DatasetLoader(dataset):
    def __init__(self, dataset_name):
        name = dataset_name.upper()
        if name not in DATASET_SPECS:
            raise ValueError(
                f'Unknown dataset {dataset_name!r}; expected one of '
                f'{", ".join(DATASET_SPECS)}'
            )

        self.spec = DATASET_SPECS[name]
        self.dataset_source_folder_path = DATA_FOLDER
        self.dataset_source_file_name = self.spec['file_name']
        super().__init__(name, self.spec['description'])

    def load(self):
        source_path = self.dataset_source_folder_path / self.dataset_source_file_name
        with source_path.open('rb') as source_file:
            data = pickle.load(source_file)

        if not isinstance(data, dict):
            raise TypeError(f'{source_path} must contain a dictionary')

        loaded_splits = {}
        for split in ('train', 'test'):
            if split not in data:
                raise KeyError(f'{source_path} is missing the {split!r} split')
            loaded_splits[split] = self._load_split(
                data[split], split, source_path
            )
        return loaded_splits

    def _load_split(self, instances, split, source_path):
        if not isinstance(instances, (list, tuple)) or not instances:
            raise ValueError(
                f'{source_path} split {split!r} must be a non-empty list or tuple'
            )

        image_shape = self.spec['image_shape']
        if len(image_shape) == 2:
            channels, height, width = 1, image_shape[0], image_shape[1]
        else:
            height, width, channels = image_shape

        images = np.empty(
            (len(instances), channels, height, width), dtype=np.float32
        )
        labels = np.empty(len(instances), dtype=np.int64)
        label_min, label_max = self.spec['label_range']
        label_offset = self.spec['label_offset']

        for index, instance in enumerate(instances):
            if not isinstance(instance, dict):
                raise TypeError(
                    f'{source_path} split {split!r} item {index} must be a dictionary'
                )
            if 'image' not in instance or 'label' not in instance:
                raise KeyError(
                    f'{source_path} split {split!r} item {index} must contain '
                    "'image' and 'label'"
                )

            image = np.asarray(instance['image'])
            if image.shape != image_shape:
                raise ValueError(
                    f'{source_path} split {split!r} item {index} has image shape '
                    f'{image.shape}; expected {image_shape}'
                )
            if not np.issubdtype(image.dtype, np.number):
                raise TypeError(
                    f'{source_path} split {split!r} item {index} image must be numeric'
                )
            if not np.isfinite(image).all():
                raise ValueError(
                    f'{source_path} split {split!r} item {index} has non-finite pixels'
                )
            if image.min() < 0 or image.max() > 255:
                raise ValueError(
                    f'{source_path} split {split!r} item {index} has pixels outside 0-255'
                )

            label = instance['label']
            if isinstance(label, (bool, np.bool_)) or not isinstance(
                label, (int, np.integer)
            ):
                raise TypeError(
                    f'{source_path} split {split!r} item {index} label must be an integer'
                )
            if not label_min <= label <= label_max:
                raise ValueError(
                    f'{source_path} split {split!r} item {index} has label {label}; '
                    f'expected {label_min}-{label_max}'
                )

            if channels == 1:
                images[index, 0] = image / 255.0
            else:
                images[index] = np.moveaxis(image, -1, 0) / 255.0
            labels[index] = label - label_offset

        return {'X': images, 'y': labels}


class ORL_Dataset_Loader(Stage3DatasetLoader):
    def __init__(self):
        super().__init__('ORL')


class MNIST_Dataset_Loader(Stage3DatasetLoader):
    def __init__(self):
        super().__init__('MNIST')


class CIFAR_Dataset_Loader(Stage3DatasetLoader):
    def __init__(self):
        super().__init__('CIFAR')


if __name__ == '__main__':
    for loader_class in (
        ORL_Dataset_Loader,
        MNIST_Dataset_Loader,
        CIFAR_Dataset_Loader,
    ):
        loaded_data = loader_class().load()
        print(loader_class.__name__)
        for split, split_data in loaded_data.items():
            images, labels = split_data['X'], split_data['y']
            print(
                f'  {split}: X={images.shape} ({images.dtype}, '
                f'{images.min():.3f}-{images.max():.3f}), '
                f'y={labels.shape} ({labels.dtype}, '
                f'{labels.min()}-{labels.max()})'
            )
        del loaded_data
