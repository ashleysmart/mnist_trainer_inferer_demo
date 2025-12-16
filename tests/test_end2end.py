# test_script_execution.py (Modified)
import tempfile
import subprocess
import sys
import os

# Get the path to the script being tested
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "main.py")

def test_end2end_mnsit_epoch1():
    """
    Tests successful trainer execution and captures/checks log output and model is made
    """

    with tempfile.TemporaryDirectory() as temp_path:
        # Construct the command list
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            '--epochs', str(1),
            '--output-dir', temp_path
        ]

        # Run the script, capturing output
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        # Check for successful exit code (0)
        assert result.returncode == 0, \
            f"Script failed with exit code {result.returncode}.\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"


        # Check for expected log output
        logs = f"{result.stdout}\n{result.stderr}"
        expected_log_items = [
            "PIPELINE - CONFIGURATION",
            "PIPELINE - TRAINING",
            "Epoch 1/1: Train Loss:",
            "PIPELINE - EVALULATION",
            "RECOMMENDATION:    -- APPROVE FOR RELEASE --",
            "PIPELINE - GENERATING REPORTS",
            "PIPELINE - SAVING",
            "Releasable: -- APPROVED --",
        ]
        for entry in expected_log_items:
            assert entry in logs, \
                f"Missing expected log message: '{entry}'"

        # Check for artifacts
        expected_files = [
            "logs/metrics.json",
            "models/model.onnx",
            "models/model.pt",
            "models/model.pth",
            "model_card/plots/confusion_matrix.png",
            "model_card/plots/per_class_accuracy.png",
            "model_card/plots/training_accuracy.png",
            "model_card/plots/training_loss.png",
            "pipeline.log",
        ]
        for file in expected_files:
            expected_file = os.path.join(temp_path, file)
            assert os.path.exists(expected_file), \
                f"Expected file not found: {expected_file}"


def test_end2end_fashion_mnsit_epoch1():
    """
    Tests successful trainer execution and captures/checks log output and model is made
    """

    with tempfile.TemporaryDirectory() as temp_path:
        # Construct the command list
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            '--epochs', str(1),
            '--output-dir', temp_path,
            '--dataset-id', 'fashion_mnist_v1'
        ]

        # Run the script, capturing output
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        # Check for successful exit code (0)
        assert result.returncode == 0, \
            f"Script failed with exit code {result.returncode}.\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

        # Check for expected log output
        logs = f"{result.stdout}\n{result.stderr}"
        expected_log_items = [
            "PIPELINE - CONFIGURATION",
            "PIPELINE - TRAINING",
            "Epoch 1/1: Train Loss:",
            "PIPELINE - EVALULATION",
            "RECOMMENDATION:    xx DO NOT RELEASE xx",
            "PIPELINE - GENERATING REPORTS",
            "PIPELINE - SAVING",
            "Releasable: xx REJECTED xx",
        ]
        for entry in expected_log_items:
            assert entry in logs, \
                f"Missing expected log message: '{entry}'"

        # Check for artifacts
        expected_files = [
            "logs/metrics.json",
            "models/model.onnx",
            "models/model.pt",
            "models/model.pth",
            "model_card/plots/confusion_matrix.png",
            "model_card/plots/per_class_accuracy.png",
            "model_card/plots/training_accuracy.png",
            "model_card/plots/training_loss.png",
            "pipeline.log",
        ]
        for file in expected_files:
            expected_file = os.path.join(temp_path, file)
            assert os.path.exists(expected_file), \
                f"Expected file not found: {expected_file}"