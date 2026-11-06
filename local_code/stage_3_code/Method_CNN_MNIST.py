from torch import nn

from local_code.stage_3_code.Method_CNN_Base import Method_CNN_Base


class Method_CNN_MNIST(Method_CNN_Base):
    def __init__(self):
        super().__init__('MNIST CNN', 'CNN classifier for handwritten digits', 10)
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, X):
        return self.network(X)
