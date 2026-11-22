'''
Small helpers shared by the stage 4 scripts: seeding, device choice, convergence plots
'''

import random

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name='auto'):
    if name == 'auto':
        return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(name)


def save_curve_plot(path, curves, title, ylabel):
    '''curves: {label: list of per-epoch values}'''
    figure, axis = plt.subplots()
    try:
        for label, values in curves.items():
            axis.plot(range(1, len(values) + 1), values, marker='o', linewidth=1.5, label=label)
        axis.set_title(title)
        axis.set_xlabel('Training epoch')
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle='--', alpha=0.5)
        if len(curves) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)
