import torch
import torch.nn as nn
import torch.optim as optim

import json
import time
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

from .model_card import ModelCardGenerator
from .visualizer import ResultsVisualizer
from .exporter import ModelExporter
from .evaluator import ModelEvaluator

import logging
logger = logging.getLogger(__name__)

class Pipeline:
    """End-to-end ML Pipeline for MNIST"""

    def __init__(self,
                 config    = None,
                 model     = None,
                 criterion = None,
                 optimizer = None,
                 dataset   = None):

        # Hyperparameters storage
        self.config = config

        seed = config.get('seed')
        if not seed:
            seed = np.random.randint(1, 1000000)
        torch.manual_seed(seed)

        logger.info(f"Using Seed: {seed}")

        output_dir =     config.get('output_dir', 'output')
        self.log_dir =   os.path.join(output_dir,    "logs")
        self.model_dir = os.path.join(output_dir,    "models")
        self.card_dir =  os.path.join(output_dir,    "model_card")
        self.plot_dir =  os.path.join(self.card_dir, "plots")

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.model   = model.to(self.device)
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer

        # Initialize sub components
        self.exporter = \
            ModelExporter.make_exporter(config.get('export_format', 'all'))

        # Metrics storage
        self.metrics = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'test_accuracy': None,
            'per_class_accuracy': {},
            'training_time': None,
            'model_size_mb': None,
            'timestamp': datetime.datetime.now().isoformat()
        }

    def validate_model(self, data_loader):
        """Evaluate model on a dataset"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        criterion = nn.NLLLoss()

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                loss = criterion(output, target)
                total_loss += loss.item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        total_loss /= len(data_loader)
        accuracy = correct / total
        return accuracy, total_loss

    def train_model(self):
        """Train the model"""
        logger.info("Training model...")

        # get the training parameters
        epochs        = self.config.get('epochs')
        learning_rate = self.config.get('learning_rate')

        # TODO.. maybe this needs a config option?
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()

        start_time = time.time()

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0

            for data, target in self.dataset.train_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            train_loss /= len(self.dataset.train_loader)
            train_acc = correct / total

            # Validation phase
            val_acc, val_loss = self.validate_model(self.dataset.val_loader)

            self.metrics['train_losses'].append(train_loss)
            self.metrics['train_accuracies'].append(train_acc)
            self.metrics['val_losses'].append(val_loss)
            self.metrics['val_accuracies'].append(val_acc)

            logger.info(
                f'Epoch {epoch+1}/{epochs}: '
                f'Train Loss: {train_loss:.4f}, '
                f'Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val Acc: {val_acc:.2f}%')

        self.metrics['training_time'] = time.time() - start_time
        logger.info(f"Training completed in {self.metrics['training_time']:.2f} seconds")

    def evaluate_confusion(self):
        """Detailed evaluation on test set"""
        logger.info("Evaluating model on test set...")

        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.dataset.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Overall accuracy
        self.metrics['test_accuracy'] = float(np.mean(
            np.array(all_preds) == np.array(all_targets)
        ))

        # confusion matrix and per-class accuracy
        cm = confusion_matrix(all_targets, all_preds)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(per_class_acc):
            self.metrics['per_class_accuracy'][f'class_{i}'] = float(acc)

        # Classification report (sklearn)
        report = classification_report(all_targets,
                                       all_preds,
                                       target_names=[str(i) for i in range(10)])

        logger.info(f"Test Accuracy: {self.metrics['test_accuracy']:.2f}%")
        logger.info("Classification Report:")
        logger.info(f"\n {report}")

        return cm

    def measure_inference_time(self, num_samples=100):
        """Measure average inference time"""
        self.model.eval()
        sample_data = torch.randn(1, 1, 28, 28).to(self.device)

        # Warm-up
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(sample_data)

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_samples):
                start = time.time()
                _ = self.model(sample_data)
                times.append((time.time() - start) * 1000)  # Convert to ms

        avg_time = float(np.mean(times))
        self.metrics['avg_inference_time_ms'] = avg_time
        logger.info(f"Average inference time: {avg_time:.2f} ms")
        return avg_time

    def calculate_model_size(self):
        """Calculate model size"""
        model_path = os.path.join(self.model_dir, 'temp_model.pth')
        torch.save(self.model.state_dict(), model_path)

        file_size_bytes = os.path.getsize(model_path)
        file_size_megabytes = file_size_bytes / (1024 * 1024)
        os.remove(model_path)
        self.metrics['model_size_mb'] = file_size_megabytes
        logger.info(f"Model size: {file_size_megabytes:.2f} MB")
        return file_size_megabytes

    def check_release_criteria(self):
        """Evaluate if model meets release criteria"""
        logger.info("" + "="*60)
        logger.info("MODEL RELEASE DECISION")
        logger.info("="*60)

        evaluators = [
            ModelEvaluator.min_test_accuracy,
            ModelEvaluator.min_per_class_accuracy,
            ModelEvaluator.max_model_size,
            ModelEvaluator.max_inference_time
        ]

        release_criteria = {}
        for evaluator in evaluators:
            result = evaluator(self.metrics, self.config)
            if result is not None:
                release_criteria.update(result)

        # Final decision
        all_release_criteria = [v.get("success", False) for v in release_criteria.values()]
        all_release_criteria = all(all_release_criteria)

        if all_release_criteria:
            logger.info("RECOMMENDATION:    -- APPROVE FOR RELEASE --")
            logger.info("All criteria met. Model is ready for production deployment.")
            decision = True
        else:
            logger.info("RECOMMENDATION:    xx DO NOT RELEASE xx")
            logger.info("Model does not meet all release criteria. Further improvements needed.")
            decision = False

        # Save decision
        self.metrics['release_approved'] = decision
        self.metrics['release_criteria'] = release_criteria

        return all_release_criteria

    def generate_visualizations(self, confusion_mat):
        """Generate all visualizations"""
        logger.info("Generating visualizations...")

        ResultsVisualizer.plot_training_loss(self.metrics, self.plot_dir)
        ResultsVisualizer.plot_training_accuracy(self.metrics, self.plot_dir)
        ResultsVisualizer.plot_confusion_matrix(confusion_mat, self.plot_dir)
        ResultsVisualizer.plot_per_class_accuracy(self.metrics['per_class_accuracy'], self.plot_dir)

    def log_metrics(self):
        """Log all metrics to JSON file"""
        log_file = os.path.join(self.log_dir, "metrics.json")

        with open(log_file, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'config': self.config
            }, f, indent=4)

        logger.info(f"Metrics logged to {log_file}")

    def save_model_artifacts(self, base_name=None):
        """Save model in multiple formats and generate model card"""
        # Export model in multiple formats
        self.exporter(self.model, self.model_dir, base_name=base_name)

        # Generate model card
        card_name = base_name or 'model'
        card_path = os.path.join(self.card_dir, f'{card_name}_card.md')
        ModelCardGenerator.generate_model_card(
            self.model,
            self.metrics,
            self.config,
            self.dataset.get_info(),
            card_path
        )

        logger.info(f"✓ Model card generated at: {card_path}")

    def __call__(self):
        """Run the complete ML pipeline"""

        # Train model
        logger.info("="*60)
        logger.info("PIPELINE - TRAINING")
        logger.info("="*60)

        self.train_model()

        # Evaluate model
        logger.info("="*60)
        logger.info("PIPELINE - EVALULATION")
        logger.info("="*60)

        confusion_mat = self.evaluate_confusion()
        self.measure_inference_time()
        self.calculate_model_size()

        # Check the release criteria
        release_approved = self.check_release_criteria()

        # Generate visualizations
        logger.info("="*60)
        logger.info("PIPELINE - GENERATING REPORTS")
        logger.info("="*60)

        if not self.config.get("skip_visualizations"):
            self.generate_visualizations(confusion_mat)
        else:
            logger.info(" >>>> Skipping visualizations (--skip-visualizations flag set)")

        # Log metrics
        self.log_metrics()

        # Save model artifacts
        logger.info("="*60)
        logger.info("PIPELINE - SAVING")
        logger.info("="*60)
        if not self.config.get("skip_export"):
            self.save_model_artifacts()
        else:
            logger.info(" >>>> Skipping model export (--skip-export flag set)")

        # Complete and summary
        logger.info("" + "="*60)
        logger.info("PIPELINE - COMPLETED")
        logger.info("="*60)
        logger.info(f"Releasable: {'-- APPROVED --' if release_approved else 'xx REJECTED xx'}")
