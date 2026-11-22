'''
Concrete MethodModule class: RNN / LSTM / GRU next-word generator trained on the jokes dataset
'''

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method
from local_code.stage_4_code.Dataset_Loader import tokenize_joke
from local_code.stage_4_code.Method_RNN_Classification import RECURRENT_LAYERS


class Method_RNN_Generation(method, nn.Module):
    PAD_ID, UNK_ID, EOS_ID = 0, 1, 2
    max_epoch = 40
    learning_rate = 1e-3
    batch_size = 32
    clip_norm = 5.0
    patience = 5  # stop when the validation loss has not improved for this many epochs
    device = None  # None = MPS if available, otherwise CPU

    def __init__(self, mName, mDescription, vocab_size, embed_dim=128, hidden_dim=256,
                 dropout=0.3, rnn_type='RNN'):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.PAD_ID)
        self.rnn = RECURRENT_LAYERS[rnn_type](embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_epoch = None

    def forward(self, x, hidden=None):
        '''x: [batch, seq_len] word ids -> (logits [batch, seq_len, vocab] for the NEXT word, hidden state)'''
        output, hidden = self.rnn(self.embedding(x), hidden)         # output: [batch, seq_len, hidden_dim]
        return self.fc(self.dropout(output)), hidden

    def _loader(self, X, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(np.asarray(X))),
                          batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, train_X, val_X, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.train_loss_history, self.val_loss_history = [], []
        best_val, best_state = float('inf'), None

        for epoch in range(self.max_epoch):
            self.train()
            total_loss, total_tokens = 0.0, 0
            for (batch,) in self._loader(train_X, shuffle=True):
                batch = batch.to(device)
                inputs, targets = batch[:, :-1], batch[:, 1:]        # predict each next word
                logits, _ = self(inputs)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                       ignore_index=self.PAD_ID)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
                optimizer.step()
                n = (targets != self.PAD_ID).sum().item()
                total_loss += loss.item() * n
                total_tokens += n

            self.train_loss_history.append(total_loss / total_tokens)
            self.val_loss_history.append(self.evaluate(val_X, device)['loss'])
            print(f'Epoch {epoch + 1}/{self.max_epoch} - {self.rnn_type} '
                  f'train loss: {self.train_loss_history[-1]:.4f} val loss: {self.val_loss_history[-1]:.4f}')
            if self.val_loss_history[-1] < best_val:
                best_val, self.best_epoch = self.val_loss_history[-1], epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
            elif epoch + 1 - self.best_epoch >= self.patience:
                print('early stopping')
                break
        self.load_state_dict(best_state)  # keep the epoch with the lowest validation loss

    def evaluate(self, X, device):
        '''Next-word metrics over all non-pad target tokens of the given jokes.
        *_all_tokens: every non-pad target (a <unk> target counts as a normal class).
        *_known_tokens: only targets that are real vocabulary words (<unk> targets skipped).
        Token-level precision/recall/F1 treat each vocabulary word as a class (all non-pad targets);
        micro-averaged P/R/F1 equal top1_accuracy_all_tokens.'''
        self.eval()
        loss_sum, all_top5, all_targets = 0.0, [], []
        with torch.no_grad():
            for (batch,) in self._loader(X, shuffle=False):
                batch = batch.to(device)
                logits, _ = self(batch[:, :-1])
                targets = batch[:, 1:]
                mask = targets != self.PAD_ID
                logits, targets = logits[mask], targets[mask]
                loss_sum += F.cross_entropy(logits, targets, reduction='sum').item()
                all_top5.append(logits.topk(5, dim=1).indices.cpu())
                all_targets.append(targets.cpu())
        top5, targets = torch.cat(all_top5), torch.cat(all_targets).unsqueeze(1)
        known = (targets != self.UNK_ID).squeeze(1)
        hit1, hit5 = top5[:, :1] == targets, top5 == targets.expand_as(top5)
        n_tokens = targets.numel()
        loss = loss_sum / n_tokens
        metrics = {
            'loss': loss, 'perplexity': math.exp(loss), 'num_tokens': n_tokens,
            'num_known_tokens': int(known.sum()), 'unk_target_rate': 1 - known.double().mean().item(),
            'top1_accuracy_all_tokens': hit1.double().mean().item(),
            'top5_accuracy_all_tokens': hit5.any(dim=1).double().mean().item(),
            'top1_accuracy_known_tokens': hit1[known].double().mean().item(),
            'top5_accuracy_known_tokens': hit5[known].any(dim=1).double().mean().item(),
        }
        for average in ('micro', 'macro', 'weighted'):
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets.squeeze(1).numpy(), top5[:, 0].numpy(), average=average, zero_division=0)
            metrics.update({f'token_precision_{average}': float(precision),
                            f'token_recall_{average}': float(recall), f'token_f1_{average}': float(f1)})
        return metrics

    def generate(self, prompt, temperature=0.0, max_new_tokens=40, seed=0):
        '''Continue a joke from its starting words. temperature 0 = greedy, >0 = sampling.
        Returns the token list (prompt words + generated words, without <eos>).'''
        device = next(self.parameters()).device
        prompt_tokens = tokenize_joke(prompt)
        generator = torch.Generator().manual_seed(seed)
        generated = []
        self.eval()
        with torch.no_grad():
            ids = [self.vocab.get(w, self.UNK_ID) for w in prompt_tokens]
            logits, hidden = self(torch.tensor([ids], device=device))  # read the starting words
            for _ in range(max_new_tokens):
                scores = logits[0, -1].float().cpu()
                scores[[self.PAD_ID, self.UNK_ID]] = float('-inf')     # never emit <pad>/<unk>
                if temperature > 0:
                    next_id = torch.multinomial(torch.softmax(scores / temperature, dim=-1), 1,
                                                generator=generator).item()
                else:
                    next_id = scores.argmax().item()
                if next_id == self.EOS_ID:
                    break
                generated.append(self.idx2word[next_id])
                logits, hidden = self(torch.tensor([[next_id]], device=device), hidden)
        return prompt_tokens + generated

    def run(self):
        device = self.device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(device)
        self.vocab = self.data['vocab']
        self.idx2word = {i: w for w, i in self.vocab.items()}
        print(f'Training {self.method_name} ({self.rnn_type}) on {device}...')
        self.method_start_time = time.time()
        self.fit(self.data['train']['X'], self.data['val']['X'], device)
        self.method_training_time = time.time() - self.method_start_time
        return {'val': self.evaluate(self.data['val']['X'], device),
                'test': self.evaluate(self.data['test']['X'], device)}
