import csv
from pathlib import Path

import numpy as np
import torch

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FOLDER = PROJECT_ROOT / 'data' / 'stage_2_data'
# separate folder so the original saved results are never overwritten
RESULT_FOLDER = PROJECT_ROOT / 'result' / 'stage_2_result' / 'logged_run'


if __name__ == '__main__':
    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('digit images', '')
    data_obj.dataset_source_folder_path = DATA_FOLDER
    loaded_data = data_obj.load()

    method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj.data = loaded_data
    result = method_obj.run()

    evaluator = Evaluate_Accuracy('test classification metrics', '')
    evaluator.data = result
    metrics = evaluator.evaluate()
    print('Test accuracy:', metrics['accuracy'])
    for average in ('macro', 'weighted', 'micro'):
        print(f'{average.title()} metrics:', metrics[average])

    metrics_path = RESULT_FOLDER / 'mlp_test_metrics.csv'
    RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
    with metrics_path.open('w', newline='') as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(('metric', 'average', 'value'))
        writer.writerow(('accuracy', 'overall', metrics['accuracy']))
        for average in ('macro', 'weighted', 'micro'):
            for metric in ('precision', 'recall', 'f1'):
                writer.writerow((metric, average, metrics[average][metric]))
    print('Saved evaluation metrics to:', metrics_path)

    loss_path = RESULT_FOLDER / 'mlp_train_loss.csv'
    with loss_path.open('w', newline='') as loss_file:
        writer = csv.writer(loss_file)
        writer.writerow(('epoch', 'train_loss'))
        for epoch, loss in enumerate(method_obj.train_loss_history, start=1):
            writer.writerow((epoch, repr(loss)))
    print('Saved training loss history to:', loss_path)

    result_saver = Result_Saver('prediction results', '')
    result_saver.result_destination_folder_path = RESULT_FOLDER
    result_saver.result_destination_file_name = 'mlp_test_predictions'
    result_saver.data = result
    result_saver.save()
