import torch
import torch.nn as nn

# Code from https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/uncertainty/evidence
class MaximumLikelihoodLoss(nn.Module):
    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        """
        :param evidence: Tensor with shape [batch_size, n_classes]
        :param target: Tensor with shape [batch_size, n_classes]
        """
        alpha = evidence + 1.
        strength = alpha.sum(dim=-1)
        loss = (target * (strength.log()[:, None] - alpha.log())).sum(dim=-1)

        # Mean loss over the batch
        return loss.mean()


class CrossEntropyBayesRisk(nn.Module):
    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        """
        :param evidence: Tensor with shape [batch_size, n_classes]
        :param target: Tensor with shape [batch_size, n_classes]
        """
        alpha = evidence + 1.
        strength = alpha.sum(dim=-1)

        loss = (target * (torch.digamma(strength)[:, None] - torch.digamma(alpha))).sum(dim=-1)

        # Mean loss over the batch
        return loss.mean()


class SquaredErrorBayesRisk(nn.Module):
    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        """
        :param evidence: Tensor with shape [batch_size, n_classes]
        :param target: Tensor with shape [batch_size, n_classes]
        """
        alpha = evidence + 1.
        strength = alpha.sum(dim=-1)
        p = alpha / strength[:, None]

        err = (target - p) ** 2
        var = p * (1 - p) / (strength[:, None] + 1)
        loss = (err + var).sum(dim=-1)

        # Mean loss over the batch
        return loss.mean()


class KLDivergenceLoss(nn.Module):
    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        """
        :param evidence: Tensor with shape [batch_size, n_classes]
        :param target: Tensor with shape [batch_size, n_classes]
        """
        alpha = evidence + 1.
        n_classes = evidence.shape[-1]

        # Remove non-misleading evidence
        alpha_tilde = target + (1 - target) * alpha
        strength_tilde = alpha_tilde.sum(dim=-1)

        first = (torch.lgamma(alpha_tilde.sum(dim=-1))
                 - torch.lgamma(alpha_tilde.new_tensor(float(n_classes)))
                 - (torch.lgamma(alpha_tilde)).sum(dim=-1))
        second = ((alpha_tilde - 1) *
                  (torch.digamma(alpha_tilde) - torch.digamma(strength_tilde)[:, None])
                  ).sum(dim=-1)
        loss = first + second

        # Mean loss over the batch
        return loss.mean()

