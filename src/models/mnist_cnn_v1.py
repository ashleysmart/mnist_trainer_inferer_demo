import torch.nn as nn

class MnistCnnV1(nn.Module):
    """CNN for MNIST classification"""

    @staticmethod
    def get_name():
        """Get a name of the model"""
        return "mnist_cnn_v1"

    def get_summary(self):
        """Get a summary of the model """

        input_shape_str = "x".join([str(x) for x in self.input_shape])
        channels = self.input_shape[0]

        return [
            f"Input: {input_shape_str} image(s)",
            f"Conv Layer 1 (nn.Conv2d): Input Channels={channels}, Output Channels=32, Kernel Size=3x3, Stride=1",
            "Activation: ReLU",
            "Conv Layer 2 (nn.Conv2d): Input Channels=32, Output Channels=64, Kernel Size=3x3, Stride=1",
            "Activation: ReLU",
            "Pooling (nn.MaxPool2d): Kernel Size=2x2",
            "Dropout (nn.Dropout): p=0.25 (Applied after pooling)",
            "Flatten: Converts (64, 12, 12) to 9216",
            "FC Layer 1 (nn.Linear): Input Units=9216, Output Units=128",
            "Activation: ReLU",
            "Dropout (nn.Dropout): p=0.5 (Applied after FC1)",
            f"FC Layer 2 (nn.Linear): Input Units=128, Output Units={self.class_count} (Output Layer)",
            "Activation: Log Softmax (Output)"
        ]

    def __init__(self, input_shape=[1, 28, 28], class_count=10):
        super().__init__()

        self.input_shape = input_shape
        self.class_count = class_count

        # Define the Convolutional block (conv + pool + dropout)
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25), # Dropout after pooling
            nn.Flatten(1) # Flattens all dimensions except the batch dimension (dim 0)
        )

        # Define the Fully Connected block (fc + dropout)
        self.classifier = nn.Sequential(
            # Flatten is applied implicitly or explicitly in forward before this
            nn.Linear(9216, 128), # 9216 = 64 * 12 * 12 (output size after 2 Conv and 1 MaxPool)
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout after FC1
            nn.Linear(128, class_count),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # Pass through the feature extraction block
        x = self.features(x)

        # Pass through the classification block
        return self.classifier(x)
