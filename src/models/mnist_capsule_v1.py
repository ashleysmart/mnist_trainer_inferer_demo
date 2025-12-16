import torch
import torch.nn as nn

from torch.autograd import Variable

def squash(x, dim=-1, eps=1e-8):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + eps)

class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels, num_capsule_channels, dim_capsule, kernel_size, stride=1):
        super().__init__()
        self.dim_capsule          = dim_capsule
        self.num_capsule_channels = num_capsule_channels
        self.kernel_size          = kernel_size
        self.stride               = stride
        self.conv = nn.Conv2d(in_channels, num_capsule_channels * dim_capsule, kernel_size, stride)

    def forward(self, x):
        x = self.conv(x)
        x = x.view((x.size(0), -1, self.dim_capsule))
        x = squash(x)
        return x

    def __repr__(self):
        s = '{}: dim_capsule={}, num_capsule_channels={}, kernel_size={}, stride={}'
        return s.format(
            self.__class__.__name__,
            self.dim_capsule,
            self.num_capsule_channels,
            self.kernel_size,
            self.stride)

class DigitalCapsule(nn.Module):
    def __init__(self, prev_num_capsules, prev_dim_capsule, num_capsules, dim_capsule, num_routing_iters=3):
        super().__init__()
        self.prev_num_capsules = prev_num_capsules
        self.prev_dim_capsule  = prev_dim_capsule
        self.num_capsules      = num_capsules
        self.dim_capsule       = dim_capsule
        self.num_routing_iters = num_routing_iters
        self.W = torch.nn.Parameter(torch.randn(num_capsules, prev_num_capsules, prev_dim_capsule, dim_capsule))

    def forward(self, x):
        u_hat = torch.matmul(x[:, None, :, None, :], self.W[None, :, :, :, :])
        b = Variable(torch.zeros(*u_hat.size()))
        if torch.cuda.is_available():
            b = b.cuda()
        for i in range(1, self.num_routing_iters + 1):
            c = nn.functional.softmax(b, dim=2)
            v = squash((c * u_hat).sum(dim=2, keepdim=True))
            if i is not self.num_routing_iters:
                db = (v * u_hat).sum(dim=-1, keepdim=True)
                b = b + db
        return v.squeeze()

    def __repr__(self):
        s = '{}: prev_num_capsules={}, prev_dim_capsule={}, num_capsules={}, dim_capsule={}, num_routing_iters={}'
        return s.format(
            self.__class__.__name__,
            self.prev_num_capsules,
            self.prev_dim_capsule,
            self.num_capsules,
            self.dim_capsule,
            self.num_routing_iters)

class MnistCapsuleV1(nn.Module):
    """
    Jeff Hintons - Capsule Network for MNIST classification

    Refer:
     - https://arxiv.org/abs/1710.09829
     - https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b
    """

    @staticmethod
    def get_name():
        """Get a name of the model"""
        return "mnist_capsule_v1"

    def get_summary(self):
        """Get a summary of the model """

        input_shape_str = "x".join([str(x) for x in self.input_shape])
        channels = self.input_shape[0]

        return [
            f"Input: {input_shape_str} image(s)",
            f"Conv Layer 1 (nn.Conv2d): Input Channels={channels}, Output Channels=256, Kernel Size=9x9, Stride=1",
            "Activation: ReLU",
            str(self.primary_capsule),
            str(self.digit_capsule),
            f"Activation: Square Softmax Outputs={self.class_count}"
        ]

    def __init__(self, input_shape=[1, 28, 28], class_count=10):
        super().__init__()

        self.class_count = class_count
        self.input_shape = input_shape
        self.digital_capsules_count = 1
        self.conv1 = torch.nn.Conv2d(input_shape[0], 256, 9)
        self.primary_capsule = PrimaryCapsule(256, 32, 8, 9, 2)
        self.digit_capsule = DigitalCapsule(1, 8, class_count, 16)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x), inplace=True)
        x = self.primary_capsule(x)
        v = self.digit_capsule(x)
        class_probs = nn.functional.softmax(torch.sqrt((v ** 2).sum(dim=-1)), dim=-1)
        return class_probs