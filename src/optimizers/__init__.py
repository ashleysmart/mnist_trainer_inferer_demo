import torch.optim as optim

optimizer_registry = [
]

def make_optimizer(optimizer_id, config, model_parameters):
    """Factory method to create loss function based on optimizer_id"""
    learning_rate = config.get('learning_rate')

    if optimizer_id == "adam":
        return optim.Adam(
            model_parameters,
            lr=learning_rate)

    for optimizer_cls in optimizer_registry:
        if optimizer_id == optimizer_cls.get_name():
            return optimizer_cls(
                model_parameters,
                config)

    raise ValueError(f"Unknown optimizer_id: {optimizer_id}")