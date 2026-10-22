'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        metrics = {'accuracy': accuracy_score(true_y, pred_y)}

        for average in ('macro', 'weighted', 'micro'):
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_y,
                pred_y,
                labels=range(10),
                average=average,
                zero_division=0,
            )
            metrics[average] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        return metrics
        