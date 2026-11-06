from torch import nn

from local_code.stage_3_code.Method_CNN_Base import Method_CNN_Base


class Method_CNN_ORL(Method_CNN_Base):
    def __init__(self, model_id='model_1'):
        if model_id not in ('model_1', 'model_2', 'model_3'):
            raise ValueError(
                "model_id must be 'model_1', 'model_2', or 'model_3'"
            )

        model_name = model_id.replace('_', ' ')
        super().__init__(
            f'ORL CNN {model_name}',
            f'{model_name.capitalize()} CNN classifier for ORL face identities',
            40,
        )
        self.dropout_rate = 0.5
        self.model_id = model_id

        convolution_stride = 1
        pooling_size = 2
        final_pooling = 'AvgPool2d(kernel_size=2, stride=2)'
        final_pooling_layer = nn.AvgPool2d(kernel_size=2)

        if model_id == 'model_1':
            convolution_layers = [
                {
                    'in_channels': 3,
                    'out_channels': 16,
                    'kernel_size': 3,
                    'padding': 1,
                },
                {
                    'in_channels': 16,
                    'out_channels': 32,
                    'kernel_size': 3,
                    'padding': 1,
                },
                {
                    'in_channels': 32,
                    'out_channels': 64,
                    'kernel_size': 3,
                    'padding': 1,
                },
            ]
            flattened_features = 64 * 7 * 5
        else:
            convolution_layers = [
                {
                    'in_channels': 3,
                    'out_channels': 32,
                    'kernel_size': 5,
                    'padding': 2,
                },
                {
                    'in_channels': 32,
                    'out_channels': 64,
                    'kernel_size': 5,
                    'padding': 2,
                },
            ]
            flattened_features = 64 * 14 * 11

        if model_id == 'model_3':
            hidden_units = 256
            self.dropout_rate = 0.3
            self.learning_rate = 1e-4
            self.max_epoch = 10
            convolution_stride = 3
            pooling_size = 3
            final_pooling = None
            final_pooling_layer = None
            flattened_features = 64
            self.optimizer_name = 'SGD'
            self.optimizer_momentum = 0.9
            self.loss_function_name = 'NLLLoss'
        else:
            hidden_units = 128

        self.configuration = {
            'dataset': 'ORL',
            'model_id': self.model_id,
            'input_shape': [3, 112, 92],
            'convolution_layers': [
                {
                    **layer,
                    'stride': convolution_stride,
                    'activation': 'ReLU',
                    'pooling': (
                        f'MaxPool2d(kernel_size={pooling_size}, '
                        f'stride={pooling_size})'
                    ),
                }
                for layer in convolution_layers
            ],
            'final_pooling': final_pooling,
            'flattened_features': flattened_features,
            'hidden_layer': {'units': hidden_units, 'activation': 'ReLU'},
            'dropout_rate': self.dropout_rate,
            'output_classes': self.num_classes,
            'optimizer': self.optimizer_name,
            'optimizer_momentum': self.optimizer_momentum,
            'learning_rate': self.learning_rate,
            'loss_function': self.loss_function_name,
            'epochs': self.max_epoch,
            'batch_size': self.batch_size,
        }

        layers = []
        for layer in convolution_layers:
            layers.extend((
                nn.Conv2d(
                    layer['in_channels'],
                    layer['out_channels'],
                    kernel_size=layer['kernel_size'],
                    stride=convolution_stride,
                    padding=layer['padding'],
                ),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=pooling_size),
            ))
        if final_pooling_layer is not None:
            layers.append(final_pooling_layer)
        layers.extend((
            nn.Flatten(),
            nn.Linear(flattened_features, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(hidden_units, self.num_classes),
        ))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)
