'''
Concrete IO classes for the stage 4 datasets: IMDb reviews (classification) and jokes (generation)
'''

import csv
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_ROOT / 'data' / 'stage_4_data'
PAD, UNK, EOS = '<pad>', '<unk>', '<eos>'  # ids 0, 1, 2


def tokenize_review(text):
    '''Lowercase, drop the <br /> tags, keep word tokens.'''
    return re.findall(r"[a-z0-9']+", text.lower().replace('<br />', ' '))


def tokenize_joke(text):
    '''Lowercase; words and punctuation marks are separate tokens so jokes can be regenerated.'''
    text = text.replace('’', "'").replace('“', '"').replace('”', '"')
    return re.findall(r"[a-z0-9']+|[^\sa-z0-9']", text.lower())


def detokenize(tokens):
    '''Inverse of tokenize_joke (approximate): join with spaces, glue punctuation to the previous word.'''
    return re.sub(r" ([.,!?;:])", r"\1", ' '.join(tokens))


def build_vocab(token_lists, specials, max_size=None, min_freq=1):
    '''word -> id, built from the given (training) token lists only; specials get the first ids.'''
    counts = Counter(token for tokens in token_lists for token in tokens)
    words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
    return {w: i for i, w in enumerate(list(specials) + words)}


def pad_sequences(id_lists, length):
    '''Right-pad id lists with 0 (PAD) to a [N, length] int64 array.'''
    out = np.zeros((len(id_lists), length), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        out[i, :len(ids)] = ids
    return out


class IMDB_Dataset_Loader(dataset):
    # limit: use only the first `limit` files per class of each split (for smoke tests)
    def __init__(self, dName=None, dDescription=None, data_folder=DATA_FOLDER,
                 max_vocab=20000, max_len=300, limit=None):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = Path(data_folder) / 'text_classification'
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.limit = limit

    def _read_split(self, split):
        texts, labels = [], []
        for label, name in ((0, 'neg'), (1, 'pos')):
            folder = Path(self.dataset_source_folder_path) / split / name
            for path in sorted(folder.glob('*.txt'))[:self.limit]:
                texts.append(path.read_text(encoding='utf-8'))
                labels.append(label)
        return texts, np.asarray(labels, dtype=np.int64)

    def _encode(self, token_lists, vocab):
        # keep the LAST max_len words (the ending of a review usually states the verdict)
        id_lists = [[vocab.get(t, vocab[UNK]) for t in tokens][-self.max_len:] or [vocab[UNK]]
                    for tokens in token_lists]
        return pad_sequences(id_lists, self.max_len)

    def load(self):
        print(f'loading IMDb reviews from {self.dataset_source_folder_path}...')
        train_texts, y_train = self._read_split('train')
        test_texts, y_test = self._read_split('test')
        train_tokens = [tokenize_review(t) for t in train_texts]
        test_tokens = [tokenize_review(t) for t in test_texts]
        vocab = build_vocab(train_tokens, (PAD, UNK), max_size=self.max_vocab)  # training data only
        return {
            'train': {'X': self._encode(train_tokens, vocab), 'y': y_train},
            'test': {'X': self._encode(test_tokens, vocab), 'y': y_test},
            'vocab': vocab,
        }


class Jokes_Dataset_Loader(dataset):
    def __init__(self, dName=None, dDescription=None, data_folder=DATA_FOLDER,
                 seed=0, val_frac=0.1, test_frac=0.1, min_freq=2):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = Path(data_folder) / 'text_generation'
        self.dataset_source_file_name = 'data'
        self.seed = seed
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.min_freq = min_freq

    def load(self):
        source_path = Path(self.dataset_source_folder_path) / self.dataset_source_file_name
        print(f'loading jokes from {source_path}...')
        with source_path.open(newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # header: "ID","Joke"
            jokes = [row[1] for row in reader]

        # drop repeated jokes (same words) so a held-out joke can never also be a training joke
        seen, unique_jokes = set(), []
        for joke in jokes:
            key = tuple(tokenize_joke(joke))
            if key not in seen:
                seen.add(key)
                unique_jokes.append(joke)
        self.num_duplicates_removed = len(jokes) - len(unique_jokes)
        jokes = unique_jokes

        random.Random(self.seed).shuffle(jokes)  # fixed split, identical for every architecture
        n_test = round(len(jokes) * self.test_frac)
        n_val = round(len(jokes) * self.val_frac)
        raw = {
            'test': jokes[:n_test],
            'val': jokes[n_test:n_test + n_val],
            'train': jokes[n_test + n_val:],
        }
        tokens = {split: [tokenize_joke(j) for j in js] for split, js in raw.items()}
        # vocabulary from training jokes only; rarer words (and every unseen word) become <unk>
        vocab = build_vocab(tokens['train'], (PAD, UNK, EOS), min_freq=self.min_freq)
        length = max(len(t) for ts in tokens.values() for t in ts) + 1  # +1 for <eos>
        data = {'vocab': vocab, 'num_duplicates_removed': self.num_duplicates_removed}
        for split in raw:
            id_lists = [[vocab.get(w, vocab[UNK]) for w in t] + [vocab[EOS]] for t in tokens[split]]
            data[split] = {'X': pad_sequences(id_lists, length), 'text': raw[split], 'tokens': tokens[split]}
        return data
