from torch import nn

from local_code.stage_3_code.Method_CNN_Base import Method_CNN_Base


class Method_CNN_MNIST(Method_CNN_Base):
    def __init__(self, model_id='model_1'):
        configurations = {
            'model_1': {
                'channels': (32, 64),
                'kernel_size': 3,
                'padding': 1,
                'stride': 1,
                'pooling': 'max',
                'hidden_units': 128,
                'dropout_rate': 0.25,
                'optimizer': 'Adam',
                'learning_rate': 1e-3,
                'loss_function': 'CrossEntropyLoss',
            },
            'model_2': {
                'channels': (32, 64, 128),
                'kernel_size': 5,
                'padding': 2,
                'stride': 1,
                'pooling': 'max',
                'hidden_units': 256,
                'dropout_rate': 0.25,
                'optimizer': 'Adam',
                'learning_rate': 1e-3,
                'loss_function': 'CrossEntropyLoss',
                'epochs': 25,
            },
            'model_3': {
                'channels': (32, 64),
                'kernel_size': 3,
                'padding': 1,
                'stride': 2,
                'pooling': 'max',
                'hidden_units': 128,
                'dropout_rate': 0.25,
                'optimizer': 'Adam',
                'learning_rate': 5e-4,
                'loss_function': 'NLLLoss',
                'epochs': 15,
            },
            'model_4': {
                'channels': (32, 64),
                'kernel_size': 3,
                'padding': 1,
                'stride': 1,
                'pooling': 'average',
                'hidden_units': 64,
                'dropout_rate': 0.4,
                'optimizer': 'SGD',
                'optimizer_momentum': 0.9,
                'learning_rate': 1e-2,
                'loss_function': 'NLLLoss',
                'epochs': 20,
            },
        }
        if model_id not in configurations:
            raise ValueError(
                f'Unknown MNIST model {model_id!r}; expected one of '
                f'{", ".join(configurations)}'
            )

        config = configurations[model_id]
        super().__init__(
            f'MNIST CNN {model_id.replace("_", " ")}',
            f'CNN classifier for handwritten digits ({model_id})',
            10,
        )
        self.model_id = model_id
        self.learning_rate = config['learning_rate']
        self.dropout_rate = config['dropout_rate']
        self.optimizer_name = config['optimizer']
        self.optimizer_momentum = config.get('optimizer_momentum')
        self.loss_function_name = config['loss_function']
        self.max_epoch = config.get('epochs', 20)
        self.batch_size = 128

        feature_height = 28
        feature_width = 28
        input_channels = 1
        layers = []
        convolution_config = []
        for output_channels in config['channels']:
            layers.extend((
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=config['kernel_size'],
                    stride=config['stride'],
                    padding=config['padding'],
                ),
                nn.ReLU(inplace=True),
                (
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    if config['pooling'] == 'max'
                    else nn.AvgPool2d(kernel_size=2, stride=2)
                ),
            ))
            feature_height = (
                feature_height + 2 * config['padding'] - config['kernel_size']
            ) // config['stride'] + 1
            feature_width = (
                feature_width + 2 * config['padding'] - config['kernel_size']
            ) // config['stride'] + 1
            feature_height = (feature_height - 2) // 2 + 1
            feature_width = (feature_width - 2) // 2 + 1
            if feature_height < 1 or feature_width < 1:
                raise ValueError(
                    f'{model_id} reduces the feature map to an invalid size'
                )
            convolution_config.append({
                'in_channels': input_channels,
                'out_channels': output_channels,
                'kernel_size': config['kernel_size'],
                'stride': config['stride'],
                'padding': config['padding'],
                'activation': 'ReLU',
                'pooling': (
                    f'{config["pooling"].capitalize()}Pool2d('
                    'kernel_size=2, stride=2)'
                ),
            })
            input_channels = output_channels

        flattened_features = input_channels * feature_height * feature_width
        layers.extend((
            nn.Flatten(),
            nn.Linear(flattened_features, config['hidden_units']),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(config['hidden_units'], self.num_classes),
        ))
        self.network = nn.Sequential(*layers)
        self.configuration = {
            'dataset': 'MNIST',
            'model_id': self.model_id,
            'input_shape': [1, 28, 28],
            'depth': len(convolution_config),
            'convolution_layers': convolution_config,
            'pooling_layer': config['pooling'],
            'flattened_features': flattened_features,
            'hidden_layer': {
                'units': config['hidden_units'],
                'activation': 'ReLU',
            },
            'dropout_rate': self.dropout_rate,
            'output_classes': self.num_classes,
            'optimizer': self.optimizer_name,
            'optimizer_momentum': self.optimizer_momentum,
            'learning_rate': self.learning_rate,
            'loss_function': self.loss_function_name,
            'epochs': self.max_epoch,
            'batch_size': self.batch_size,
        }

    def forward(self, X):
        return self.network(X)
