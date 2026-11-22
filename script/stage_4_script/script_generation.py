'''
Stage 4 (tasks 4-4 / 4-5): joke generation from three starting words with RNN, LSTM and GRU.

    python -m script.stage_4_script.script_generation --model all
    python -m script.stage_4_script.script_generation --model rnn --smoke
'''

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch

from local_code.stage_4_code.Dataset_Loader import DATA_FOLDER, Jokes_Dataset_Loader, detokenize
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Result_Utils import get_device, save_curve_plot, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_FOLDER = PROJECT_ROOT / 'result' / 'stage_4_result' / 'generation'
MODELS = ('rnn', 'lstm', 'gru')
CUSTOM_PROMPTS = ('what did the', 'my dog is', 'the professor said')  # not taken from a specific joke
SAMPLING_TEMPERATURE = 0.8
PROMPT_WORDS = 3


def build_prompts(data):
    '''Same prompts for every architecture: first 3 tokens of 5 training jokes, 5 test jokes, plus custom ones.'''
    prompts = []
    for group, split in (('train', 'train'), ('heldout', 'test')):
        for text, tokens in list(zip(data[split]['text'], data[split]['tokens']))[:5]:
            prompts.append({'group': group, 'prompt': ' '.join(tokens[:PROMPT_WORDS]), 'original': text,
                            'original_tokens': tokens})
    prompts += [{'group': 'custom', 'prompt': p, 'original': None, 'original_tokens': None} for p in CUSTOM_PROMPTS]
    return prompts


def unigram_prf(generated, original):
    '''Word-overlap precision / recall / F1 of a generated continuation against the original one.
    precision = share of generated words found in the original, recall = share of original words generated.'''
    overlap = sum((Counter(generated) & Counter(original)).values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision, recall = overlap / len(generated), overlap / len(original)
    return precision, recall, 2 * precision * recall / (precision + recall)


def generate_samples(model, data, seed):
    train_jokes = {tuple(t) for t in data['train']['tokens']}
    samples = []
    for i, p in enumerate(build_prompts(data)):
        sample = {k: p[k] for k in ('group', 'prompt', 'original')}
        for mode, temperature in (('greedy', 0.0), ('sampled', SAMPLING_TEMPERATURE)):
            tokens = model.generate(p['prompt'], temperature=temperature, seed=seed + i)
            sample[mode] = detokenize(tokens)
            sample[f'{mode}_copies_a_training_joke'] = tuple(tokens) in train_jokes
            if p['original_tokens']:  # compare only the continuation, not the three given words
                scores = unigram_prf(tokens[PROMPT_WORDS:], p['original_tokens'][PROMPT_WORDS:])
                for metric, value in zip(('precision', 'recall', 'f1'), scores):
                    sample[f'{mode}_unigram_{metric}_vs_original'] = value
        samples.append(sample)

    summary = {}
    for group in ('train', 'heldout', 'custom'):
        rows = [s for s in samples if s['group'] == group]
        for mode in ('greedy', 'sampled'):
            summary[f'{group}_{mode}_copies_a_training_joke'] = sum(s[f'{mode}_copies_a_training_joke'] for s in rows)
            if group != 'custom':
                for metric in ('precision', 'recall', 'f1'):
                    summary[f'{group}_{mode}_mean_unigram_{metric}'] = sum(
                        s[f'{mode}_unigram_{metric}_vs_original'] for s in rows) / len(rows)
    return samples, summary


def save_results(name, model, metrics, samples, summary, configuration):
    folder = RESULT_FOLDER / name
    folder.mkdir(parents=True, exist_ok=True)

    with (folder / 'test_metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('split', 'metric', 'value'))
        for split in ('val', 'test'):
            for metric, value in metrics[split].items():
                writer.writerow((split, metric, value))
    with (folder / 'training_history.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('epoch', 'train_loss', 'val_loss'))
        for epoch, (tr, va) in enumerate(zip(model.train_loss_history, model.val_loss_history), 1):
            writer.writerow((epoch, tr, va))
    save_curve_plot(folder / 'training_loss.png',
                    {'train loss': model.train_loss_history, 'validation loss': model.val_loss_history},
                    f'Jokes {name.upper()} next-word loss', 'Cross-entropy per word')

    (folder / 'generated_samples.json').write_text(
        json.dumps({'summary': summary, 'samples': samples}, indent=2) + '\n')
    lines = [f'{name.upper()} generated jokes (greedy = most likely word each step, '
             f'sampled = temperature {SAMPLING_TEMPERATURE}); text is lowercased by the tokenizer\n']
    for s in samples:
        lines += [f"[{s['group']} prompt] {s['prompt']}",
                  f"  original : {s['original'] or '(none: custom prompt)'}",
                  f"  greedy   : {s['greedy']}", f"  sampled  : {s['sampled']}", '']
    (folder / 'generated_samples.txt').write_text('\n'.join(lines))
    with (folder / 'configuration.json').open('w') as f:
        json.dump(configuration, f, indent=2)
        f.write('\n')


def save_comparison():
    '''Combine whichever model result folders exist into comparison.csv and comparison plots.'''
    rows, train_curves, val_curves = [], {}, {}
    for name in MODELS:
        folder = RESULT_FOLDER / name
        if not (folder / 'test_metrics.csv').exists():
            continue
        with (folder / 'test_metrics.csv').open() as f:
            metrics = {(r['split'], r['metric']): float(r['value']) for r in csv.DictReader(f)}
        with (folder / 'training_history.csv').open() as f:
            history = list(csv.DictReader(f))
        configuration = json.loads((folder / 'configuration.json').read_text())
        summary = json.loads((folder / 'generated_samples.json').read_text())['summary']
        train_curves[name.upper()] = [float(r['train_loss']) for r in history]
        val_curves[name.upper()] = [float(r['val_loss']) for r in history]
        row = {'model': name.upper(), 'best_epoch': configuration['best_epoch'], 'epochs_run': len(history),
               'num_parameters': configuration['num_parameters']}
        for key in ('loss', 'perplexity', 'top1_accuracy_all_tokens', 'top1_accuracy_known_tokens',
                    'top5_accuracy_all_tokens', 'top5_accuracy_known_tokens', 'unk_target_rate',
                    'token_precision_macro', 'token_recall_macro', 'token_f1_macro',
                    'token_precision_weighted', 'token_recall_weighted', 'token_f1_weighted'):
            row[f'test_{key}'] = metrics[('test', key)]
        for group in ('train', 'heldout'):  # greedy continuations vs the original jokes (3 starting words)
            for metric in ('precision', 'recall', 'f1'):
                row[f'{group}_prompts_greedy_unigram_{metric}'] = summary[f'{group}_greedy_mean_unigram_{metric}']
        row['train_prompts_greedy_copies'] = summary['train_greedy_copies_a_training_joke']
        row['training_time_sec'] = configuration['training_time_sec']
        rows.append(row)
    if rows:
        with (RESULT_FOLDER / 'comparison.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_curve_plot(RESULT_FOLDER / 'comparison_train_loss.png', train_curves,
                        'Jokes generation: training loss', 'Cross-entropy per word')
        save_curve_plot(RESULT_FOLDER / 'comparison_val_loss.png', val_curves,
                        'Jokes generation: validation loss', 'Cross-entropy per word')


def main():
    parser = argparse.ArgumentParser(description='Train/evaluate stage 4 joke generators.')
    parser.add_argument('--model', choices=(*MODELS, 'all'), default='all')
    parser.add_argument('--epochs', type=int, default=40, help='maximum epochs (early stopping on validation loss)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('auto', 'mps', 'cpu'), default='auto')
    parser.add_argument('--data_dir', default=str(DATA_FOLDER))
    parser.add_argument('--smoke', action='store_true', help='2 epochs, nothing saved')
    args = parser.parse_args()

    epochs = 2 if args.smoke else args.epochs
    device = get_device(args.device)

    set_seed(args.seed)
    data = Jokes_Dataset_Loader(data_folder=args.data_dir, seed=args.seed).load()
    print(f"train {data['train']['X'].shape}, val {data['val']['X'].shape}, "
          f"test {data['test']['X'].shape}, vocab {len(data['vocab'])}")

    for name in (MODELS if args.model == 'all' else (args.model,)):
        set_seed(args.seed)  # identical seed for every architecture
        model = Method_RNN_Generation(name.upper(), f'Jokes {name.upper()} generator',
                                      vocab_size=len(data['vocab']), rnn_type=name.upper())
        model.max_epoch = epochs
        model.device = device
        model.data = data
        metrics = model.run()
        samples, summary = generate_samples(model, data, args.seed)
        test = metrics['test']
        print(f"{name.upper()} test loss {test['loss']:.4f} perplexity {test['perplexity']:.1f} "
              f"top-1 all tokens {test['top1_accuracy_all_tokens']:.4f} known tokens "
              f"{test['top1_accuracy_known_tokens']:.4f} token F1 macro {test['token_f1_macro']:.4f} "
              f"weighted {test['token_f1_weighted']:.4f}")
        if args.smoke:
            for s in samples[:3]:
                print(' ', s['prompt'], '->', s['greedy'], '|', s['sampled'])
            print('smoke run - results not saved')
            continue
        configuration = {
            'task': 'jokes next-word generation', 'rnn_type': name.upper(), 'seed': args.seed,
            'device': str(device), 'max_epochs': epochs, 'best_epoch': model.best_epoch,
            'early_stopping_patience': model.patience, 'optimizer': 'Adam', 'learning_rate': model.learning_rate,
            'batch_size': model.batch_size, 'grad_clip_norm': model.clip_norm, 'loss_function': 'CrossEntropyLoss',
            'embed_dim': model.embedding.embedding_dim, 'hidden_dim': model.rnn.hidden_size,
            'dropout': model.dropout.p, 'vocab_size': len(data['vocab']), 'vocab_min_freq': 2,
            'duplicate_jokes_removed_before_split': data['num_duplicates_removed'],
            'split': {'train': len(data['train']['text']), 'val': len(data['val']['text']),
                      'test': len(data['test']['text'])},
            'sampling_temperature': SAMPLING_TEMPERATURE, 'max_new_tokens': 40,
            'model_selection': 'weights from the epoch with the lowest validation loss',
            'training_time_sec': model.method_training_time, 'torch_version': torch.__version__,
            'num_parameters': sum(p.numel() for p in model.parameters()),
        }
        save_results(name, model, metrics, samples, summary, configuration)
    if not args.smoke:
        save_comparison()


if __name__ == '__main__':
    main()
