'''
Concrete MethodModule class: vanilla RNN for IMDb binary sentiment classification
'''

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from local_code.base_class.method import method


class Method_RNN_Classification(method, nn.Module):
    PAD_ID = 0

    def __init__(self, mName, mDescription, vocab_size, embed_dim=64, hidden_dim=128, num_classes=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.PAD_ID)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)  # vanilla (tanh) RNN
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        '''x: [batch, seq_len] word ids, right-padded with PAD_ID -> logits [batch, num_classes]'''
        lengths = (x != self.PAD_ID).sum(dim=1).clamp(min=1)         # real length of each review
        embedded = self.embedding(x)                                  # [batch, seq_len, embed_dim]
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)                                     # h_n: [1, batch, hidden_dim]
        return self.fc(h_n[-1])                                       # [batch, num_classes]

    # fit / test / run (training + evaluation) are left for task 4-3
