import logging
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles the automatic evaluations of the model

    Design notes:
     - The return for each of the exporter should be the a dict of the
        key -> values that way the user can disable/enable checks.
    """

    @staticmethod
    def print_eval(label, actual, required, success):
        status = "-- PASS --" if success else "xx FAIL xx"
        logger.info(f"  {status} - {label}: {actual:.4f} - (Required: {required})")

    @staticmethod
    def min_test_accuracy(metrics,
                          config):
        if 'min_test_accuracy' not in config:
            return None

        label    = "Min test accuracy"
        actual   = metrics['test_accuracy']
        required = config['min_test_accuracy']
        success  = bool(actual >= required)
        notes    = None

        ModelEvaluator.print_eval(label, actual, required, success)

        return {
            "min_test_accuracy": {
                "label": label,
                "required": required,
                "actual": actual,
                "success": success,
                "notes": notes
            }
        }

    @staticmethod
    def min_per_class_accuracy(metrics,
                               config):
        if 'min_per_class_accuracy' not in config:
            return None

        worst_class = min(metrics['per_class_accuracy'].items(),
                        key=lambda x: x[1])

        label    = "Min per class accuracy"
        actual   = float(worst_class[1])
        required = config['min_per_class_accuracy']
        success  = bool(actual >= required)
        notes    = f"Worst performing class: {worst_class[0]} at {worst_class[1]:.2f}"

        ModelEvaluator.print_eval(label, actual, required, success)

        return {
            "min_per_class_accuracy": {
                "label": label,
                "required": required,
                "actual": actual,
                "success": success,
                "notes": notes
            }}

    @staticmethod
    def max_model_size(metrics,
                       config):
        if 'max_model_size_mb' not in config:
            return None

        label    = "Max model size"
        actual   = metrics['model_size_mb']
        required = config['max_model_size_mb']
        success  = bool(actual <= required)
        notes    = None

        ModelEvaluator.print_eval(label, actual, required, success)

        return {
            "max_model_size_mb": {
                "label": label,
                "required": required,
                "actual": actual,
                "success": success,
                "notes": notes
            }}

    @staticmethod
    def max_inference_time(metrics,
                               config):
        if 'max_inference_time_ms' not in config:
            return None

        label    = "Max ave inference time ms"
        actual   = metrics['avg_inference_time_ms']
        required = config['max_inference_time_ms']
        success  = bool(actual <= required)
        notes    = None

        ModelEvaluator.print_eval(label, actual, required, success)

        return {
            "max_inference_time_ms": {
                "label": label,
                "required": required,
                "actual": actual,
                "success": success,
                "notes": notes
            }}
