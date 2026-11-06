import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method


class Method_CNN_Base(method, nn.Module):
    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 128
    optimizer_name = 'Adam'
    optimizer_momentum = None
    loss_function_name = 'CrossEntropyLoss'

    def __init__(self, mName, mDescription, num_classes):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.num_classes = num_classes
        self.train_loss_history = []

    @staticmethod
    def _as_tensor(data, dtype):
        array = np.asarray(data)
        if array.dtype != np.dtype(dtype):
            array = array.astype(dtype, copy=False)
        return torch.from_numpy(array)

    def fit(self, X, y):
        device = next(self.parameters()).device
        train_data = TensorDataset(
            self._as_tensor(X, np.float32),
            self._as_tensor(y, np.int64),
        )
        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=device.type == 'cuda',
        )
        if self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate
            )
        elif self.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=self.optimizer_momentum or 0.0,
            )
        else:
            raise ValueError(f'Unsupported optimizer: {self.optimizer_name}')

        if self.loss_function_name == 'CrossEntropyLoss':
            loss_function = nn.CrossEntropyLoss()
        elif self.loss_function_name == 'NLLLoss':
            loss_function = nn.NLLLoss()
        else:
            raise ValueError(
                f'Unsupported loss function: {self.loss_function_name}'
            )
        self.train_loss_history = []

        for epoch in range(self.max_epoch):
            self.train()
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = self(X_batch)
                loss_input = (
                    torch.log_softmax(logits, dim=1)
                    if self.loss_function_name == 'NLLLoss'
                    else logits
                )
                loss = loss_function(loss_input, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * y_batch.size(0)

            mean_loss = total_loss / len(train_data)
            self.train_loss_history.append(float(mean_loss))
            print(
                f'Epoch {epoch + 1}/{self.max_epoch} '
                f'- mean training loss: {mean_loss:.6f}'
            )

    def test(self, X):
        device = next(self.parameters()).device
        test_data = TensorDataset(self._as_tensor(X, np.float32))
        test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=device.type == 'cuda',
        )
        predictions = []
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for (X_batch,) in test_loader:
                    logits = self(X_batch.to(device))
                    predictions.append(logits.argmax(dim=1).cpu())
        finally:
            self.train(was_training)

        return torch.cat(predictions)

    def run(self):
        if self.data is None:
            raise ValueError('Set model.data to the loaded train/test splits before run()')

        device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        self.to(device)
        print(f'Training {self.method_name} on {device}...')
        self.fit(self.data['train']['X'], self.data['train']['y'])
        predictions = self.test(self.data['test']['X'])
        return {
            'pred_y': predictions.numpy(),
            'true_y': np.asarray(self.data['test']['y']),
        }
