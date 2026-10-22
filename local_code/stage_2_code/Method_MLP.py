'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 20
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 128

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 128)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(128, 10)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        h = self.activation_func_1(self.fc_layer_1(x))
        return self.fc_layer_2(h)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        train_data = TensorDataset(
            torch.tensor(np.asarray(X), dtype=torch.float32),
            torch.tensor(np.asarray(y), dtype=torch.long),
        )
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.max_epoch):
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                logits = self.forward(X_batch)
                train_loss = loss_function(logits, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                total_loss += train_loss.item() * y_batch.size(0)

            if epoch % 5 == 0 or epoch == self.max_epoch - 1:
                with torch.no_grad():
                    train_pred = self.test(X)
                accuracy_evaluator.data = {
                    'true_y': np.asarray(y),
                    'pred_y': train_pred.numpy(),
                }
                mean_loss = total_loss / len(train_data)
                train_metrics = accuracy_evaluator.evaluate()
                print(
                    'Epoch:', epoch + 1,
                    'Accuracy:', train_metrics['accuracy'],
                    'Loss:', mean_loss,
                )
    
    def test(self, X):
        with torch.no_grad():
            logits = self.forward(torch.tensor(np.asarray(X), dtype=torch.float32))
            return logits.argmax(dim=1)
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {
            'pred_y': pred_y.numpy(),
            'true_y': np.asarray(self.data['test']['y']),
        }
            