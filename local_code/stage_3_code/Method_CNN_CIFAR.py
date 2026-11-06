from torch import nn

from local_code.stage_3_code.Method_CNN_Base import Method_CNN_Base


class Method_CNN_CIFAR(Method_CNN_Base):
    def __init__(self):
        super().__init__('CIFAR-10 CNN', 'CNN classifier for CIFAR-10 objects', 10)
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, X):
        return self.network(X)
