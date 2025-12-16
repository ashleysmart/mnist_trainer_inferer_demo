from dotenv import load_dotenv

import argparse
import os
import datetime

import logging
logger = logging.getLogger(__name__)

def load_config_from_env():
    """Load configuration from .env file"""

    # Load .env file if it exists
    load_dotenv()

    output_dir = os.getenv('OUTPUT_DIR', './output') \
        + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # generate defaults from environment variables
    config = {
        # Model/Data settings
        'model_id':   os.getenv('MODEL_ID', 'mnist_cnn_v0'),
        'dataset_id': os.getenv('DATASET_ID', 'mnist_data_v0'),

        # Directory settings
        'data_dir':      os.getenv('DATA_DIR', './data'),
        'output_dir':    output_dir,
        'export_format': os.getenv('EXPORT_FORMAT', 'all'),

        # Training hyperparameters
        'criterion_id': os.getenv('CRITERION_ID', 'nnl'),
        'optimizer_id': os.getenv('OPTIMIZER_ID', 'adam'),

        'seed':          int(os.getenv('SEED')),
        'epochs':        int(os.getenv('EPOCHS', '5')),
        'batch_size':    int(os.getenv('BATCH_SIZE', '64')),
        'learning_rate': float(os.getenv('LEARNING_RATE', '0.001')),

        # Release criteria
        'min_test_accuracy':      float(os.getenv('MIN_TEST_ACCURACY', '0.97')),
        'min_per_class_accuracy': float(os.getenv('MIN_PER_CLASS_ACCURACY', '0.90')),
        'max_model_size_mb':      float(os.getenv('MAX_MODEL_SIZE_MB', '10.0')),
        'max_inference_time_ms':  float(os.getenv('MAX_INFERENCE_TIME_MS', '50.0')),
    }

    return config

def parse_args():
    """Parse command line arguments"""

    # Load defaults from .env
    default_config = load_config_from_env()

    # allow the user to overloaded any parameters
    parser = argparse.ArgumentParser(
        description='MNIST Training Pipeline - Train and evaluate digit classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-q', '--quiet', type=bool,
                        default=False,
                        help='Silence the logging output')

    # model and dataset arguments
    dir_group = parser.add_argument_group('Model/Dataset Settings')
    dir_group.add_argument('--model-id', type=str,
                           default=default_config['model_id'],
                           help='The model architecture to use')
    dir_group.add_argument('--dataset-id', type=str,
                           default=default_config['dataset_id'],
                          help='The dataset to use to train/eval the model')

    # Directory arguments
    dir_group = parser.add_argument_group('Directory Settings')
    dir_group.add_argument('--data-dir', type=str, default=default_config['data_dir'],
                          help='Directory for MNIST data')
    dir_group.add_argument('--output-dir', type=str, default=default_config['output_dir'],
                          help='Directory for logs')
    dir_group.add_argument('--export_format',nargs='+', default=default_config['export_format'],
                           help='format(s) to save model as: all, pytorch, onnx, torchscript (repeatable)')

    # Training hyperparameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--seed', type=int,
                           default=default_config.get('seed'),
                           help='The seed to use for repeating the training run')
    train_group.add_argument('--criterion-id', type=str,
                           default=default_config['criterion_id'],
                           help='The loss function(s) to use')
    train_group.add_argument('--optimizer-id', type=str,
                           default=default_config['optimizer_id'],
                           help='The optimizer to use')
    train_group.add_argument('--epochs', type=int, default=default_config['epochs'],
                            help='Number of training epochs')
    train_group.add_argument('--batch-size', type=int, default=default_config['batch_size'],
                            help='Training batch size')
    train_group.add_argument('--learning-rate', type=float, default=default_config['learning_rate'],
                            help='Learning rate for optimizer')

    # Release criteria
    criteria_group = parser.add_argument_group('Release Criteria')
    criteria_group.add_argument('--min-test-accuracy', type=float,
                               default=default_config['min_test_accuracy'],
                               help='Minimum test accuracy for release (0-1)')
    criteria_group.add_argument('--min-per-class-accuracy', type=float,
                               default=default_config['min_per_class_accuracy'],
                               help='Minimum per-class accuracy for release (0-1)')
    criteria_group.add_argument('--max-model-size-mb', type=float,
                               default=default_config['max_model_size_mb'],
                               help='Maximum model size in MB')
    criteria_group.add_argument('--max-inference-time-ms', type=float,
                               default=default_config['max_inference_time_ms'],
                               help='Maximum inference time in milliseconds')

    # Execution options
    exec_group = parser.add_argument_group('Execution Options')
    exec_group.add_argument('--skip-visualizations', action='store_true',
                           help='Skip generating visualization plots')
    exec_group.add_argument('--skip-export', action='store_true',
                           help='Skip exporting model in multiple formats')

    args = parser.parse_args()

    # return final hyperparameters as a dictionary
    return vars(args)

def setup_logger(output_dir, verbose):
    global logger
    log_file_path = os.path.join(output_dir, 'pipeline.log')

    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s > %(message)s')

    log_level = logging.DEBUG if verbose else logging.INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.DEBUG)   # Log everything
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)

def main():
    # Parse arguments (with .env defaults)
    args = parse_args()
    verbose = not args.get('quiet', False)

    output_dir = args.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)

    # setup logger
    # --- 3. Configure Logging (Happens AFTER Argparse) ---
    setup_logger(output_dir, verbose)

    # import other modules (after logger setup)
    from . import models
    from . import datasets
    from . import criterion
    from . import optimizers
    from .pipeline import Pipeline

    # give the log a header with hyperparam summary
    logger.info( "="*60)
    logger.info( "PIPELINE - CONFIGURATION")
    logger.info( "="*60)
    logger.info(f"Model: {args.get('model_id')}")
    logger.info(f"Dataset: {args.get('dataset_id')}")
    logger.info( "Locations:")
    logger.info(f" - Data directory: {args.get('data_dir')}")
    logger.info(f" - Output directory: {args.get('output_dir')}")
    logger.info( "Training: ")
    logger.info(f" - Criterion: {args.get('criterion_id')}")
    logger.info(f" - Optimizer: {args.get('optimizer_id')}")
    logger.info(f" - Epochs: {args.get('epochs')}")
    logger.info(f" - Batch size: {args.get('batch_size')}")
    logger.info(f" - Learning rate: {args.get('learning_rate')}")
    logger.info( "Release Criteria:")
    logger.info(f" - Min test accuracy: {args.get('min_test_accuracy')*100:.1f}%")
    logger.info(f" - Min per-class accuracy: {args.get('min_per_class_accuracy')*100:.1f}%")
    logger.info(f" - Max model size: {args.get('max_model_size_mb')} MB")
    logger.info(f" - Max inference time: {args.get('max_inference_time_ms')} ms")

    # Setup model and dataset
    model = models.make_model(args.get('model_id'))
    dataset = datasets.make_dataset(
        args.get('dataset_id'),
        data_dir=args.get('data_dir'),
        batch_size=args.get('batch_size'))
    criteria = criterion.make_criterion(args.get('criterion_id'))
    optimizer = optimizers.make_optimizer(
        args.get('optimizer_id'),
        args,
        model.parameters())

    dataset_info = dataset.get_info()
    logger.info( "Dataset size: ")
    logger.info(f" - Training samples:   {dataset_info.get('train_samples')}")
    logger.info(f" - Validation samples: {dataset_info.get('val_samples')}")
    logger.info(f" - Test samples:       {dataset_info.get('test_samples')}")
    logger.info("="*60)

    # Initialize pipeline
    pipeline = Pipeline(
        model=model,
        dataset=dataset,
        criterion=criteria,
        optimizer=optimizer,
        config=args)

    # run the training/eval/etc
    pipeline()
