from torch import nn
from .capsule_loss_v1 import CapsuleLossV1

criterion_registry = [
    CapsuleLossV1
]

def make_criterion(criterion_id):
    """Factory method to create loss function based on criterion_id"""
    if criterion_id == "nll":
        return nn.NLLLoss()
    elif criterion_id == "mse":
        return nn.MSELoss()

    for criterion_cls in criterion_registry:
        if criterion_id == criterion_cls.get_name():
            return criterion_cls()

    raise ValueError(f"Unknown criterion_id: {criterion_id}")