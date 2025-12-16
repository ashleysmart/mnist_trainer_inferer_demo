import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
plt.set_loglevel("INFO")

class ResultsVisualizer:
    """
    Handles all visualization and plotting tasks

    Basically its just a couple wrappers for of plt and sns code that I
    plots training history, confusion matrix, per-class accuracy
    """

    @staticmethod
    def plot_training_loss(metrics, output_dir=None):
        """Plot training and validation metrics over epochs"""
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'training_loss.png')

        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))

        epochs = range(1, len(metrics['train_losses']) + 1)

        # Loss plot
        ax1.plot(epochs, metrics['train_losses'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, metrics['val_losses'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()

        logger.info(f"Training loss-epochs plot saved to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_training_accuracy(metrics, output_dir=None):
        """Plot training and validation metrics over epochs"""
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'training_accuracy.png')

        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))

        epochs = range(1, len(metrics['train_losses']) + 1)

        # Accuracy plot
        ax1.plot(epochs, metrics['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, metrics['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Training & Validation Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(output_dir, 'training_accuracy.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Training accuracy-epochs plot saved to {save_path}")
        return save_path

    @staticmethod
    def plot_confusion_matrix(cm, output_dir=None):
        """Plot confusion matrix heatmap

        Ideallly this is lights up on the diagonal only...
        """
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'confusion_matrix.png')

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(10), yticklabels=range(10),
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path is None:
            save_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Confusion matrix saved to {save_path}")
        return save_path

    @staticmethod
    def plot_per_class_accuracy(per_class_acc, output_dir=None):
        """Plot per-class accuracy bar chart"""

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'per_class_accuracy.png')

        classes = sorted(per_class_acc.keys())
        accuracies = [per_class_acc[c] for c in classes]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(classes)), accuracies, color='steelblue', alpha=0.8)

        # Color bars below threshold in red
        threshold = 0.90
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            if acc < threshold:
                bar.set_color('crimson')

        plt.axhline(y=threshold, color='red', linestyle='--',
                   label=f'Threshold ({threshold}%)', linewidth=2)
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        plt.title('Per-Class Accuracy')
        plt.xticks(range(len(classes)), [c.split('_')[1] for c in classes])
        plt.ylim([0.0, 1.05])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        if save_path is None:
            save_path = os.path.join(output_dir, 'per_class_accuracy.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Per-class accuracy plot saved to {save_path}")
        return save_path
