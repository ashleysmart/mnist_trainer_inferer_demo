# MNIST Training Pipeline - Usage Guide

![Mnist_Demo](https://github.com/user-attachments/assets/63e035b0-ba9c-4cdc-9283-3a660d55d0e7)

## Summary

The MNIST training pipeline
- First the intention is that the models are trained locally and then deployed

- A discussion of the architecture and explanation of my design choices is available at [doc/Design.md]
- The spec as it was given to me is at [doc/Spec.md]
- Some research notes are at [doc/Research.md]
  - In particular the SOTA seems to say that hintons capsules networks are the top rank for single models.
  - There is an instruction below on how to reconfigure the pipeline to train a capsule network

## Limitations and incomplete stuff in the task

- My intention was not to focus as much as i did on training..
- the CI/CD isnt complete to my liking

- My intention for the CI/CD was to use Nvidia trition.
  - However i ran into trouble with seting up the account(s) and realized it would be complex for a consumer to do as well
  - ideally models should be stored in a model zoo repo.. that a triton like docker image can consume them from
  - on check in the CI/CD should build that model zoo update and release for use
  - you dont actually need to build a new docker image every time the nvida trition image can mount the repo
  - The current deployment is a docker image that runs a flask server to test the model with..
    - this is sub par and over engineered

## How to use

### build the docker

Create the docker image to reproduce the system without issues

```bash
bin/build
```

The docker image is used
- to ensure portability of the build and check changes locally before pushing to a repo
- It should pickup your machines GPU if you have one and use it

### Confirm the basic operation

Run the release tests. This uses the docker image

```bash
bin/test
```

These tests are
- just a quick set of end2end tests
    - this due to time limits..
    - there should be more tests in a real prod system, one for each model, loss, etc

## Recommended Usage Examples

### Train the basic cnn network on MNIST

```bash
bin/train --model-id mnist_cnn_v1 --epochs 5
```

### Train the basic cnn network on FashionMNIST

```bash
bin/train --model-id mnist_cnn_v1 --epochs 5 --dataset-id fashion_mnist_v1
```

### Train capsule network on MNIST

```bash
bin/train --model-id mnist_capsule_v1 --epochs 20 --criterion-id capsule_loss_v1 --max-model-size-mb 30
```

## Command Line Arguments

You can always use `--help` to check the latest options and confirm this documentation is upto date

### Model/Dataset Settings
- `-q` or `--quiet`: Silence the logging output (default: False)

### Model/Dataset Settings
- `--model-id`:   Model architecture used  (default: 'mnist_cnn_v1')
- `--dataset-id`: Dataset name used to train/evel (default: 'mnist_v1')

### Directory Settings
- `--data-dir`:      Directory for MNIST data (default: from .env or './data')
- `--output-dir`:    Directory for the output (default: from .env or './output_<date>_<time>')
- `--export_format`: Format(s) to save model as: `all`, `pytorch`, `onnx`, `torchscript` (repeatable) (default: `all`)

### Training parameters
- `--criterion-id`:  The loss function(s) to use (default: `nll`) or `mse`
- `--optimizer-id`:  The optimizer to use (default: `adam`)
- `--epochs`:        Number of training epochs (default: `5`)
- `--batch-size`:    Training batch size (default: `64`)
- `--learning-rate`: Learning rate for optimizer (default: `0.001`)

### Release Criteria
- `--min-test-accuracy`: Minimum test accuracy for release, 0-1.0 (default: `0.97`)
- `--min-per-class-accuracy`: Minimum per-class accuracy, 0-1.0 (default: `0.90`)
- `--max-model-size-mb`: Maximum model size in MB (default: `10.0`)
- `--max-inference-time-ms`: Maximum inference time in ms (default: `50.0`)

### Execution Options
- `--skip-visualizations`: Skip generating visualization plots (default: `False`)
- `--skip-export`: Skip exporting model in multiple formats (default: `False`)

## Output Structure

After running the pipeline, you'll get:

```
output/
├── pipeline.log                    # training log
├── logs/
│   └── metrics.json                # summary of model metrics
├── models/
│   ├── mnist_model.pth             # PyTorch format
│   ├── mnist_model.onnx            # ONNX format
│   ├── mnist_model.pt              # TorchScript format
└── model_card
    ├── mnist_model_card.md         # Model documentation
    └── plots/
        ├── confusion_matrix.png    # confusion matrix plot
        ├── per_class_accuracy.png  # per class accuracy graph
        ├── training_accuracy.png   # training graph (accuracy)
        └── training_loss.png       # training graph (loss)
```
