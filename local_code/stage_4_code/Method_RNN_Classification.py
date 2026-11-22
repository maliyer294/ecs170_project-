'''
Concrete MethodModule class: RNN / LSTM / GRU classifier for IMDb binary sentiment classification
'''

import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method

RECURRENT_LAYERS = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}


class Method_RNN_Classification(method, nn.Module):
    PAD_ID = 0
    max_epoch = 10
    learning_rate = 1e-3
    batch_size = 64
    clip_norm = 5.0
    device = None  # None = MPS if available, otherwise CPU

    # rnn_type: 'RNN' (vanilla, task 4-2/4-3), 'LSTM' or 'GRU' (task 4-5)
    def __init__(self, mName, mDescription, vocab_size, embed_dim=64, hidden_dim=128,
                 num_classes=2, rnn_type='RNN'):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.PAD_ID)
        self.rnn = RECURRENT_LAYERS[rnn_type](embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.train_loss_history = []
        self.train_accuracy_history = []

    def forward(self, x):
        '''x: [batch, seq_len] word ids, right-padded with PAD_ID -> logits [batch, num_classes]'''
        lengths = (x != self.PAD_ID).sum(dim=1).clamp(min=1)         # real length of each review
        embedded = self.embedding(x)                                  # [batch, seq_len, embed_dim]
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)                                     # h_n: [1, batch, hidden_dim]
        if isinstance(h_n, tuple):                                    # LSTM returns (h_n, c_n)
            h_n = h_n[0]
        return self.fc(h_n[-1])                                       # [batch, num_classes]

    def fit(self, X, y, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        train_data = TensorDataset(torch.from_numpy(np.asarray(X)), torch.from_numpy(np.asarray(y)))
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.train_loss_history = []
        self.train_accuracy_history = []

        for epoch in range(self.max_epoch):
            self.train()
            total_loss, total_correct = 0.0, 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = self(X_batch)
                loss = loss_function(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)  # RNNs can have exploding gradients
                optimizer.step()
                total_loss += loss.item() * y_batch.size(0)
                total_correct += (logits.argmax(dim=1) == y_batch).sum().item()

            self.train_loss_history.append(total_loss / len(train_data))
            self.train_accuracy_history.append(total_correct / len(train_data))
            print(f'Epoch {epoch + 1}/{self.max_epoch} - {self.rnn_type} '
                  f'training loss: {self.train_loss_history[-1]:.4f} '
                  f'training accuracy: {self.train_accuracy_history[-1]:.4f}')

    def test(self, X, device):
        test_loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(X))), batch_size=self.batch_size)
        predictions = []
        self.eval()
        with torch.no_grad():
            for (X_batch,) in test_loader:
                predictions.append(self(X_batch.to(device)).argmax(dim=1).cpu())
        return torch.cat(predictions)

    def run(self):
        device = self.device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(device)
        print(f'Training {self.method_name} ({self.rnn_type}) on {device}...')
        self.method_start_time = time.time()
        self.fit(self.data['train']['X'], self.data['train']['y'], device)
        self.method_training_time = time.time() - self.method_start_time
        pred_y = self.test(self.data['test']['X'], device)
        self.method_stop_time = time.time()
        self.method_testing_time = self.method_stop_time - self.method_start_time - self.method_training_time
        return {'pred_y': pred_y.numpy(), 'true_y': np.asarray(self.data['test']['y'])}
