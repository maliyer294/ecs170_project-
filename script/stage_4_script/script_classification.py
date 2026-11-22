'''
Stage 4 (tasks 4-3 / 4-5): IMDb sentiment classification with RNN, LSTM and GRU.

    python -m script.stage_4_script.script_classification --model all
    python -m script.stage_4_script.script_classification --model rnn --smoke
'''

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from local_code.stage_4_code.Dataset_Loader import DATA_FOLDER, IMDB_Dataset_Loader
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_4_code.Result_Utils import get_device, save_curve_plot, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_FOLDER = PROJECT_ROOT / 'result' / 'stage_4_result' / 'classification'
MODELS = ('rnn', 'lstm', 'gru')


def save_results(name, model, predictions, configuration):
    folder = RESULT_FOLDER / name
    folder.mkdir(parents=True, exist_ok=True)
    true_y, pred_y = predictions['true_y'], predictions['pred_y']

    with (folder / 'test_metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('metric', 'average', 'class', 'value'))
        writer.writerow(('accuracy', 'overall', '', accuracy_score(true_y, pred_y)))
        for average in ('binary', 'macro', 'weighted'):  # binary = positive class (label 1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_y, pred_y, average=average, zero_division=0)
            for metric, value in (('precision', precision), ('recall', recall), ('f1', f1)):
                writer.writerow((metric, average, '', float(value)))
        per_class = precision_recall_fscore_support(true_y, pred_y, labels=[0, 1], average=None, zero_division=0)
        for label, values in zip(('negative', 'positive'), zip(*per_class)):
            for metric, value in zip(('precision', 'recall', 'f1', 'support'), values):
                writer.writerow((metric, 'per_class', label, float(value)))

    with (folder / 'training_history.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('epoch', 'train_loss', 'train_accuracy'))
        for epoch, (loss, acc) in enumerate(zip(model.train_loss_history, model.train_accuracy_history), 1):
            writer.writerow((epoch, loss, acc))

    save_curve_plot(folder / 'training_loss.png', {name.upper(): model.train_loss_history},
                    f'IMDb {name.upper()} training loss', 'Mean training loss')
    with (folder / 'configuration.json').open('w') as f:
        json.dump(configuration, f, indent=2)
        f.write('\n')


def save_comparison():
    '''Combine whichever model result folders exist into comparison.csv and comparison_loss.png.'''
    rows, losses = [], {}
    for name in MODELS:
        folder = RESULT_FOLDER / name
        if not (folder / 'test_metrics.csv').exists():
            continue
        with (folder / 'test_metrics.csv').open() as f:
            metrics = {(r['metric'], r['average']): float(r['value']) for r in csv.DictReader(f) if not r['class']}
        with (folder / 'training_history.csv').open() as f:
            history = list(csv.DictReader(f))
        configuration = json.loads((folder / 'configuration.json').read_text())
        losses[name.upper()] = [float(r['train_loss']) for r in history]
        rows.append({
            'model': name.upper(),
            'test_accuracy': metrics[('accuracy', 'overall')],
            'precision_positive': metrics[('precision', 'binary')],
            'recall_positive': metrics[('recall', 'binary')],
            'f1_positive': metrics[('f1', 'binary')],
            'f1_macro': metrics[('f1', 'macro')],
            'final_train_loss': losses[name.upper()][-1],
            'final_train_accuracy': float(history[-1]['train_accuracy']),
            'training_time_sec': configuration['training_time_sec'],
        })
    if rows:
        with (RESULT_FOLDER / 'comparison.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_curve_plot(RESULT_FOLDER / 'comparison_loss.png', losses,
                        'IMDb classification: training loss', 'Mean training loss')


def main():
    parser = argparse.ArgumentParser(description='Train/evaluate stage 4 IMDb classifiers.')
    parser.add_argument('--model', choices=(*MODELS, 'all'), default='all')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('auto', 'mps', 'cpu'), default='auto')
    parser.add_argument('--data_dir', default=str(DATA_FOLDER))
    parser.add_argument('--smoke', action='store_true', help='tiny subset, 1 epoch, nothing saved')
    args = parser.parse_args()

    limit = 100 if args.smoke else None  # reviews per class and split
    epochs = 1 if args.smoke else args.epochs
    device = get_device(args.device)

    set_seed(args.seed)
    data = IMDB_Dataset_Loader(data_folder=args.data_dir, limit=limit).load()
    print(f"train {data['train']['X'].shape}, test {data['test']['X'].shape}, vocab {len(data['vocab'])}")

    for name in (MODELS if args.model == 'all' else (args.model,)):
        set_seed(args.seed)  # identical seed for every architecture
        model = Method_RNN_Classification(name.upper(), f'IMDb {name.upper()} classifier',
                                          vocab_size=len(data['vocab']), rnn_type=name.upper())
        model.max_epoch = epochs
        model.device = device
        model.data = data
        predictions = model.run()

        accuracy = accuracy_score(predictions['true_y'], predictions['pred_y'])
        print(f'{name.upper()} test accuracy: {accuracy:.4f}')
        if args.smoke:
            print('logits check: predictions', predictions['pred_y'].shape, 'smoke run - results not saved')
            continue
        configuration = {
            'task': 'IMDb binary sentiment classification', 'rnn_type': name.upper(), 'seed': args.seed,
            'device': str(device), 'epochs': epochs, 'optimizer': 'Adam', 'learning_rate': model.learning_rate,
            'batch_size': model.batch_size, 'grad_clip_norm': model.clip_norm, 'loss_function': 'CrossEntropyLoss',
            'embed_dim': model.embedding.embedding_dim, 'hidden_dim': model.rnn.hidden_size,
            'vocab_size': len(data['vocab']), 'max_len': data['train']['X'].shape[1],
            'truncation': 'keep last max_len tokens', 'embeddings': 'randomly initialised, trained from scratch',
            'train_size': int(len(data['train']['y'])), 'test_size': int(len(data['test']['y'])),
            'training_time_sec': model.method_training_time, 'torch_version': torch.__version__,
            'num_parameters': sum(p.numel() for p in model.parameters()),
        }
        save_results(name, model, predictions, configuration)
    if not args.smoke:
        save_comparison()


if __name__ == '__main__':
    main()
