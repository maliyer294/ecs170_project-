import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.script_data_loader import (
    CIFAR_Dataset_Loader,
    MNIST_Dataset_Loader,
    ORL_Dataset_Loader,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_FOLDER = PROJECT_ROOT / 'result' / 'stage_3_result'
EXPERIMENTS = {
    'ORL': (ORL_Dataset_Loader, Method_CNN_ORL),
    'MNIST': (MNIST_Dataset_Loader, Method_CNN_MNIST),
    'CIFAR': (CIFAR_Dataset_Loader, Method_CNN_CIFAR),
}


def save_results(dataset_name, model, predictions):
    model_id = getattr(model, 'model_id', 'model_1')
    configuration = getattr(
        model,
        'configuration',
        {
            'dataset': dataset_name,
            'model_id': model_id,
            'optimizer': 'Adam',
            'learning_rate': model.learning_rate,
            'loss_function': 'CrossEntropyLoss',
            'epochs': model.max_epoch,
            'batch_size': model.batch_size,
        },
    )
    result_folder = RESULT_FOLDER / dataset_name.lower() / model_id
    result_folder.mkdir(parents=True, exist_ok=True)
    class_labels = range(model.num_classes)
    accuracy = accuracy_score(predictions['true_y'], predictions['pred_y'])
    per_class_metrics = precision_recall_fscore_support(
        predictions['true_y'],
        predictions['pred_y'],
        labels=class_labels,
        average=None,
        zero_division=0,
    )
    averages = {}
    for average in ('macro', 'weighted', 'micro'):
        precision, recall, f1, _ = precision_recall_fscore_support(
            predictions['true_y'],
            predictions['pred_y'],
            labels=class_labels,
            average=average,
            zero_division=0,
        )
        averages[average] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

    metrics_path = result_folder / 'test_metrics.csv'
    with metrics_path.open('w', newline='') as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(('metric', 'average', 'class', 'value'))
        writer.writerow(('accuracy', 'overall', '', accuracy))
        for average, values in averages.items():
            for metric, value in values.items():
                writer.writerow((metric, average, '', value))
        precision, recall, f1, support = per_class_metrics
        for class_index, values in enumerate(zip(precision, recall, f1, support)):
            for metric, value in zip(
                ('precision', 'recall', 'f1', 'support'), values
            ):
                writer.writerow((metric, 'per_class', class_index, value))

    predictions_path = result_folder / 'test_predictions.pkl'
    with predictions_path.open('wb') as prediction_file:
        pickle.dump(predictions, prediction_file)

    loss_path = result_folder / 'training_loss.csv'
    with loss_path.open('w', newline='') as loss_file:
        writer = csv.writer(loss_file)
        writer.writerow(('epoch', 'mean_training_loss'))
        writer.writerows(
            (epoch, loss)
            for epoch, loss in enumerate(model.train_loss_history, start=1)
        )

    plot_path = result_folder / 'training_loss.png'
    figure, axis = plt.subplots()
    try:
        axis.plot(
            range(1, len(model.train_loss_history) + 1),
            model.train_loss_history,
            marker='o',
            linewidth=1.5,
        )
        axis.set_title(f'{dataset_name} CNN Training Loss')
        axis.set_xlabel('Training epoch')
        axis.set_ylabel('Mean training loss')
        axis.grid(True, linestyle='--', alpha=0.5)
        figure.tight_layout()
        figure.savefig(plot_path, dpi=150)
    finally:
        plt.close(figure)

    configuration_path = result_folder / 'configuration.json'
    with configuration_path.open('w') as configuration_file:
        json.dump(configuration, configuration_file, indent=2)
        configuration_file.write('\n')

    checkpoint_path = result_folder / 'model_state.pt'
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'configuration': configuration,
        },
        checkpoint_path,
    )

    print(f'{dataset_name} test accuracy: {accuracy:.4f}')
    print('Saved metrics to:', metrics_path)
    print('Saved predictions to:', predictions_path)
    print('Saved training loss to:', loss_path, 'and', plot_path)
    print('Saved configuration to:', configuration_path)
    print('Saved model checkpoint to:', checkpoint_path)


def run_experiment(dataset_name, model_number='1'):
    loader_class, model_class = EXPERIMENTS[dataset_name]
    model_id = f'model_{model_number}'
    if dataset_name != 'ORL' and model_number != '1':
        raise ValueError(
            'Models 2 and 3 are currently implemented for ORL only.'
        )

    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(2)

    loaded_data = loader_class().load()
    model = (
        model_class(model_id=model_id)
        if dataset_name == 'ORL'
        else model_class()
    )
    model.data = loaded_data
    predictions = model.run()
    save_results(dataset_name, model, predictions)


def main():
    parser = argparse.ArgumentParser(
        description='Train a stage-three CNN and save its test results.'
    )
    parser.add_argument(
        '--dataset',
        choices=(*EXPERIMENTS, 'ALL'),
        default='ALL',
        help='Dataset to train; ALL runs ORL, MNIST, and CIFAR sequentially.',
    )
    parser.add_argument(
        '--model',
        choices=('1', '2', '3'),
        default='1',
        help='Model configuration number; ORL currently supports models 1, 2, and 3.',
    )
    args = parser.parse_args()
    if args.model != '1' and args.dataset != 'ORL':
        parser.error('Models 2 and 3 are currently implemented for ORL only.')
    datasets = EXPERIMENTS if args.dataset == 'ALL' else (args.dataset,)
    for dataset_name in datasets:
        print(f'\n=== Running {dataset_name} CNN model {args.model} ===')
        run_experiment(dataset_name, args.model)


if __name__ == '__main__':
    main()
