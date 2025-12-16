import torch
import datetime

import logging
logger = logging.getLogger(__name__)

class ModelCardGenerator:
    """Generates model cards documenting model details and performance"""

    # --- Markdown String Templates (now in lower_snake_case) ---

    # Template for the start of the model card (Model Details section)
    model_details_template = """# Model Card: MNIST Digit Classifier

## Model Details

**Model Name:**    {model_id}
**Training Date:** {timestamp}
**Framework:**     PyTorch {pytorch_version}

### Architecture
{model_summary}

### Training Details

**Hyperparameters:**

- Epochs:        {epochs}
- Batch Size:    {batch_size}
- Learning Rate: {learning_rate}
- Optimizer: Adam
- Loss Function: Negative Log Likelihood

**Training Data:**

- Dataset:            {dataset_id}
- Training Samples:   {train_samples}
- Validation Samples: {val_samples}
- Test Samples:       {test_samples}

**Training Time:**    {training_time:.2f} seconds

## Performance Metrics

### Overall Performance

- **Test Accuracy:**          {test_accuracy:.2f}%
- **Model Size:**             {model_size_mb:.2f} MB
- **Average Inference Time:** {avg_inference_time_ms:.2f} ms

### Per-Class Performance

"""

    # Template for the Per-Class Performance lines (used in a loop)
    per_class_metric_template = "- **{class_key}:** {acc:.2f}%\n"

    # Template for the Training History and Release Criteria Evaluation section
    training_history_and_release_template = """
- **Max:** {max_class_acc:.2f}%
- **Min:** {min_class_acc:.2f}%

!(per_class_accuracy)[plots/per_class_accuracy.png]
!(confusion_matrix)[plots/confusion_matrix.png]

### Training History

!(training_accuracy)[plots/training_accuracy.png]
!(training_loss)[plots/training_loss.png]

- **Final Training Accuracy:** {final_train_acc:.2f}%
- **Final Validation Accuracy:** {final_val_acc:.2f}%
- **Best Validation Accuracy:** {best_val_acc:.2f}%

## Release Criteria Evaluation

| Criteria                       | Required   | Actual     | Status | Notes
|--------------------------------|------------|------------|--------|-------
"""

    # Template for the Release Criteria Table (conditional, only if 'release_criteria' is present)
    release_criteria_table_row_template = "| {label} | {required} | {actual} | {status} | {notes}\n"

    release_criteria_final_template = """
**Release Decision:** {release_approved}
"""

    # Template for the final section of the model card
    footer = """
---
*This model card was automatically generated.*
"""

    @staticmethod
    def generate_model_card(model, metrics, config, dataset, save_path):
        """Generate a comprehensive model card in markdown format"""

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        model_summary = \
            "\n - " + \
            "\n - ".join(model.get_summary())

        # 1. Build the Model Details section
        card = ModelCardGenerator.model_details_template.format(
            model_id              = model.get_name(),
            dataset_id            = config.get("dataset_id"),
            pytorch_version       = torch.__version__,
            timestamp             = timestamp,
            model_summary         = model_summary,
            epochs                = config.get('epochs', 'N/A'),
            batch_size            = config.get('batch_size', 'N/A'),
            learning_rate         = config.get('learning_rate', 'N/A'),
            train_samples         = dataset.get('train_samples', 'N/A'),
            val_samples           = dataset.get('val_samples', 'N/A'),
            test_samples          = dataset.get('test_samples', 'N/A'),
            training_time         = metrics['training_time'],
            test_accuracy         = metrics['test_accuracy'],
            model_size_mb         = metrics['model_size_mb'],
            avg_inference_time_ms = metrics['avg_inference_time_ms']
        )

        # 2. Add Per-Class Performance metrics (loop)
        for class_key in sorted(metrics['per_class_accuracy'].keys()):
            acc = metrics['per_class_accuracy'][class_key]
            # Assumes class_name is like 'class_1', 'class_2', etc.
            card += ModelCardGenerator.per_class_metric_template.format(
                class_key=class_key,
                acc=acc)

        # Per class summary
        min_class_acc = min(metrics['per_class_accuracy'].values())
        max_class_acc = max(metrics['per_class_accuracy'].values())

        # 3. Add Training History section
        card += ModelCardGenerator.training_history_and_release_template.format(
            min_class_acc=min_class_acc,
            max_class_acc=max_class_acc,
            final_train_acc=metrics['train_accuracies'][-1],
            final_val_acc=metrics['val_accuracies'][-1],
            best_val_acc=max(metrics['val_accuracies'])
        )

        # 4. Conditionally add Release Criteria Table
        if 'release_criteria' in metrics:
            criteria = metrics['release_criteria']
            def status_formatter(success):
                return ' PASS ' if success else ' FAIL '

            def column_formatter(value,length):
                return f"{str(value): <{length}}"

            def column_num_formatter(value,length):
                return f"{str(value): <{length}.6}"

            for entry in criteria.values():
                formatted_entry = {
                    "label":    column_formatter(entry.get("label"), 30),
                    "required": column_formatter(entry.get("required"), 10),
                    "actual":   column_num_formatter(entry.get("actual"),10),
                    "notes":    entry.get("notes"),
                    "status":   status_formatter(entry.get('success')),
                }
                card += ModelCardGenerator.release_criteria_table_row_template \
                    .format(**formatted_entry)

            # final
            release_approved = metrics.get('release_approved')
            if release_approved is None:
                release_approved = "?? PENDING ??"
            elif release_approved:
                release_approved = "-- APPROVED --"
            else:
                release_approved = "xx REJECTED xx"
            card += ModelCardGenerator.release_criteria_final_template \
                .format(release_approved=release_approved)

        # 5. Add Contact and Footer
        card += ModelCardGenerator.footer

        # 6. Save the card
        with open(save_path, 'w') as f:
            f.write(card)

        logger.info(f"Model card saved to {save_path}")
        return save_path