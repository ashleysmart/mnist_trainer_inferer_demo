from .mnist_cnn_v1 import MnistCnnV1
from .mnist_capsule_v1 import MnistCapsuleV1

model_registry = [
    MnistCnnV1,
    MnistCapsuleV1
]

def make_model(model_id):
    """Factory method to create model instances based on model_id"""
    for model_cls in model_registry:
        if model_id == model_cls.get_name():
            return model_cls()

    raise ValueError(f"Unknown model_id: {model_id}")