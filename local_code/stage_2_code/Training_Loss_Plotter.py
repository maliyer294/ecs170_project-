from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def save_training_loss_plot(loss_values: Sequence[float], output_path: Path) -> None:
    if not loss_values:
        raise ValueError('Cannot plot training loss: no epoch losses were recorded.')

    epochs = range(1, len(loss_values) + 1)
    figure, axis = plt.subplots()
    try:
        axis.plot(epochs, loss_values, marker='o', linewidth=1.5)
        axis.set_title('MLP Training Loss by Epoch')
        axis.set_xlabel('Training epoch')
        axis.set_ylabel('Mean training loss')
        axis.grid(True, linestyle='--', alpha=0.5)
        figure.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
    finally:
        plt.close(figure)
