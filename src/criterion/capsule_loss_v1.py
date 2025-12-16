import torch
from torch import nn

def norm_plus_ep(tensor=None, dim=-1, epsilon=1e-7, keep_dims=False):
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=keep_dims)
    return torch.sqrt(squared_norm + epsilon)

class CapsuleLossV1(nn.Module):
    # https://arxiv.org/pdf/1710.09829 - section 3 - Margin loss for digit existence

    @staticmethod
    def get_name():
        """Get a name of the loss"""
        return "capsule_loss_v1"

    def __init__(self,
                 m_plus       = 0.9,
                 m_minus      = 0.1,
                 lambda_minus = 0.5,
                 device       = None):
        super().__init__()

        self.device = device

        # Hyperparameters for loss function
        self.m_plus       = m_plus
        self.m_minus      = m_minus
        self.lambda_minus = lambda_minus

    def forward(self, actual, labels):
        # https://arxiv.org/pdf/1710.09829
        # now offically Jeff Hinton used a "total loss function" which has 2 parts
        # - the margin loss
        # - the reconstruction loss (when its in an auto-encoder form)
        # i will only focus on the margin loss
        #
        # Margin loss for digit existence
        # Lk =
        #     Tk * max(0, m_plus − ||vk||)^2 +            (a)
        #     λ * (1 − Tk) * max(0, ||vk|| − m_neg)^2     (b)
        #
        # - Tk = labels
        #    - as hinton put it "where Tk = 1 iff a digit of class k is present3"
        #    - so in english that the reads "labels", 1 if correct 0 if incorrect
        # - vk = raw outputs.. ||vk|| is  normalized
        # - m_plus = 0.9
        # - m_neg = 0.1
        #    - ".. and m+ = 0.9 and m− = 0.1...""
        # - λ = 0.5
        #    -  "... use λ = 0.5..."
        #
        #  - (a) "present" basically this is the loss for the classes its supposed to be seen
        #  - (b) "absent"  basically this is the loss for the other classes
        #

        output_norm = norm_plus_ep(actual, dim=-2, keep_dims=True)

        zero = torch.zeros(output_norm.size()).to(self.device)

        # (a) - the present error (correct digit)
        present_error = torch.square((torch.maximum(zero, self.m_plus - output_norm)))
        present_error = torch.reshape(present_error, shape=(-1, 10))
        present_error = labels * present_error

        # (b) - the absent error (incorrect digit)
        #absent_error = torch.square((torch.max(zero, output_norm - self.m_minus)))
        absent_error = torch.square((torch.maximum(zero, output_norm - self.m_minus)))
        absent_error = torch.reshape(absent_error, shape=(-1, 10))
        absent_error = self.lambda_minus * (1. - labels) * absent_error

        # final function
        margin_loss = present_error + absent_error
        margin_loss = torch.sum(margin_loss, dim=-1, keepdim=True)
        margin_loss = torch.mean(margin_loss)

        return margin_loss