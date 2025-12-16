import torch

import os
import logging
logger = logging.getLogger(__name__)

def flatten(iterable):
    if iterable is None:
        return
    elif isinstance(iterable,dict):
        yield from flatten(iterable.items())
    else:
        for item in iterable:
            if isinstance(item, list) or \
                isinstance(item,tuple) or \
                isinstance(item,set) or \
                isinstance(item,dict):
                yield from flatten(item)
            else:
                yield item

class ModelExporter:
    """Handles exporting models to multiple formats

    Supported formats:
    - Native PyTorch (.pth)
    - TorchScript (.pt)
    - ONNX (.onnx)

    Design notes:
     - The return for each of the exporter should be the a dict of the
        type(s) -> path(s) that way the user can swap or combine exporters
        or allow the user to select which formats to export.
    """

    @staticmethod
    def export_pytorch(model,
                       output_dir,
                       base_name=None):
        """Export model in native PyTorch format"""
        os.makedirs(output_dir, exist_ok=True)
        base_name = base_name or 'model'
        filename = f'{base_name}.pth'
        model_path = os.path.join(output_dir, filename)

        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': str(model)
        }, model_path)

        logger.info(f"✓ PyTorch model saved to {model_path}")
        return { 'pytorch': model_path }

    @staticmethod
    def export_onnx(model,
                    output_dir,
                    base_name=None):
        """Export model to ONNX format"""
        os.makedirs(output_dir, exist_ok=True)
        base_name = base_name or 'model'
        filename = f'{base_name}.onnx'
        model_path = os.path.join(output_dir, filename)

        # Create dummy input
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 1, 28, 28).to(device)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            dynamo=False  # Use legacy exporter
        )

        logger.info(f"✓ ONNX model saved to {model_path}")
        return { 'onnx': model_path }

    @staticmethod
    def export_torchscript(model,
                           output_dir,
                           base_name=None):
        """Export model as TorchScript format"""
        os.makedirs(output_dir, exist_ok=True)
        base_name = base_name or 'model'
        filename = f'{base_name}.pt'
        model_path = os.path.join(output_dir, filename)

        # Trace the model
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        traced_model = torch.jit.trace(model, dummy_input)

        # Save traced model
        traced_model.save(str(model_path))

        logger.info(f"✓ TorchScript model saved to {model_path}")
        return { 'torchscript': model_path }

    @staticmethod
    def make_exporter(types):
        """Factory method to create exporter or a combine of exporters based on type(s)"""
        if types is None:
            return None

        if isinstance(types, list):
            types = list(flatten(types))
            if len(types) == 0:
                return None
            elif len(types) == 1:
                types = types[0]
                # fall out to the string handler
            else:
                # create a combined exporter that calls each in sequence
                def combined_exporter(model, output_dir, base_name=None):
                    paths = {}
                    for t in types:
                        exporter = ModelExporter.make_exporter(t)
                        result = exporter(model, output_dir, base_name)
                        if isinstance(result, dict):
                            paths.update(result)
                        else:
                            paths[t] = result
                    return paths
                return combined_exporter

        if isinstance(types, str):
            type_str = types.lower()
            if type_str == 'all' or \
                type_str is None or \
                type_str == '*':
                return ModelExporter.make_exporter(['pytorch',
                                                    'onnx',
                                                    'torchscript'])
            elif type_str == 'pytorch' or \
                type_str == 'pt':
                return ModelExporter.export_pytorch
            elif type_str == 'onnx':
                return ModelExporter.export_onnx
            elif type_str == 'torchscript' or \
                type_str == 'ts':
                return ModelExporter.export_torchscript

        raise ValueError(f"Unsupported exporter type: {type_str}")
