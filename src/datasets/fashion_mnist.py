from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import logging
logger = logging.getLogger(__name__)

class FashionMnistDataV1:
    @staticmethod
    def get_name():
        """Get a name of the dataset"""
        return "fashion_mnist_v1"

    def __init__(self, data_dir=None, batch_size=64):
        """Load FashionMNIST dataset"""
        logger.info("Loading Fashion MNIST dataset...")

        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        self.train_dataset = datasets.FashionMNIST(data_dir, train=True,  transform=transform, download=True)
        self.test_dataset  = datasets.FashionMNIST(data_dir, train=False, transform=transform, download=True)

        # Split training data for validation
        train_size = int(0.9 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        self.train_subset, self.val_subset = random_split(
            self.train_dataset, [train_size, val_size]
        )

        self.train_loader = DataLoader(self.train_subset, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_subset,   batch_size=batch_size, shuffle=False)
        self.test_loader  = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        self.batch_size = batch_size

    def get_info(self):
        return {
            'batch_size': self.batch_size,
            'train_samples': len(self.train_subset),
            'val_samples': len(self.val_subset),
            'test_samples': len(self.test_dataset)
        }