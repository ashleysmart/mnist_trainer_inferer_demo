from .mnist import MnistDataV1
from .fashion_mnist import FashionMnistDataV1

dataset_registry = [
    MnistDataV1,
    FashionMnistDataV1
]

def make_dataset(dataset_id, data_dir=None, batch_size=64):
    """Factory method to load dataset based on dataset_id"""
    for dataset_cls in dataset_registry:
        if dataset_id == dataset_cls.get_name():
            return dataset_cls(
                data_dir=data_dir,
                batch_size=batch_size)

    raise ValueError(f"Unknown dataset_id: {dataset_id}")